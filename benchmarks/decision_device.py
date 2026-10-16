#!/usr/bin/env python3
"""Decision-model device benchmark: run the EXACT router decision model
(vllm-sr/Vela-2.0-0.3B, the same 0.3B encoder the Semantic Router uses to pick
the tier) on CPU or on the AMD MI300X, and measure decision throughput.

Same model, same code (transformers reference path, fp32), the ONLY variable is
the device. This is the apples-to-apples "decision model on CPU vs GPU" the booth
toggle asks for. Nothing is simulated: it prints real per-decision latency or the
real error.

Note for honesty: the production Semantic Router classifies on CPU in ~1 ms using
a compiled, optimized runtime, not this transformers fp32 path. This benchmark
isolates the raw model forward on each device so the GPU acceleration is visible.

Emits JSONL progress to stdout (one JSON object per line) so the dashboard can
render a live progress bar:
  {"ev":"start","device":...,"total":N}
  {"ev":"load","device":...,"load_s":...}
  {"ev":"item","device":...,"done":k,"total":N,"tier":...,"got":...,"ms":...}
  {"ev":"summary","device":...,"avg_ms":...,"p50_ms":...,"dps":...,"wall_s":...,"n":...}
"""
import argparse, json, os, statistics, sys, time

REPO = os.environ.get("DECISION_REPO", "vllm-sr/Vela-2.0-0.3B")

# A small, representative, length-varied set across the three routing tiers.
PROMPTS = [
    ("hi thanks!", "trivial"),
    ("What is the capital of France?", "trivial"),
    ("Convert 10 km to miles.", "trivial"),
    ("Who wrote Romeo and Juliet and when?", "standard"),
    ("Summarize the main causes of World War I.", "standard"),
    ("Explain how a hash map achieves average O(1) lookups.", "standard"),
    ("Prove step by step that there are infinitely many primes.", "hard"),
    ("Derive the time complexity of mergesort from its recurrence.", "hard"),
    ("Design a rate limiter and analyze its worst-case behavior under bursty load.", "hard"),
]

# The tier decision, expressed as a SystemOne choice question (trivial/standard/hard).
QUESTIONS = {
    "tier": {
        "type": "choice",
        "instructions": "How hard is this user request to answer well?",
        "criteria": {
            "trivial": "a trivial factual or greeting question a tiny model answers instantly",
            "standard": "a moderate question needing a mid-size model",
            "hard": "a hard reasoning, math, or multi-step proof needing the largest model",
        },
    },
}


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "gpu"], required=True)
    ap.add_argument("--n", type=int, default=int(os.environ.get("DECISION_N", "6")))
    args = ap.parse_args()

    items = (PROMPTS * ((args.n // len(PROMPTS)) + 1))[: args.n]
    emit({"ev": "start", "device": args.device, "total": len(items), "model": REPO})

    import torch
    from transformers import AutoModel

    dev = "cpu"
    if args.device == "gpu":
        if not torch.cuda.is_available():
            emit({"ev": "error", "device": "gpu", "error": "no ROCm/CUDA device visible to torch"})
            sys.exit(2)
        dev = "cuda"

    t0 = time.time()
    m = AutoModel.from_pretrained(REPO, trust_remote_code=True)
    if dev != "cpu":
        m = m.to(dev)
    load_s = time.time() - t0
    emit({"ev": "load", "device": args.device, "load_s": round(load_s, 2)})

    # Warmup: exclude first-call graph/kernel compile from the measured window.
    for _ in range(3):
        m.system_one(state=items[0][0], questions=QUESTIONS)
    if dev != "cpu":
        torch.cuda.synchronize()

    lats, rows = [], []
    wall0 = time.time()
    for k, (text, want) in enumerate(items, 1):
        t = time.time()
        r = m.system_one(state=text, questions=QUESTIONS)
        if dev != "cpu":
            torch.cuda.synchronize()
        ms = (time.time() - t) * 1000.0
        lats.append(ms)
        got = r["answers"]["tier"]["choice"]
        rows.append({"tier": want, "got": got, "ms": round(ms, 1)})
        emit({"ev": "item", "device": args.device, "done": k, "total": len(items),
              "tier": want, "got": got, "ms": round(ms, 1)})
    wall_s = time.time() - wall0

    avg = statistics.mean(lats)
    p50 = statistics.median(lats)
    dps = len(lats) / wall_s if wall_s > 0 else None
    emit({"ev": "summary", "device": args.device,
          "avg_ms": round(avg, 2), "p50_ms": round(p50, 2),
          "dps": round(dps, 2) if dps else None,
          "wall_s": round(wall_s, 2), "n": len(lats),
          "load_s": round(load_s, 2), "rows": rows,
          "model": REPO})


if __name__ == "__main__":
    main()
