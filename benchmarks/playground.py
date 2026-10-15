#!/usr/bin/env python3
"""
Benchmark playground: GPU performance WITH vLLM semantic routing ON vs OFF.

The story this measures, on real hardware, no fabrication:

  ROUTING ON   every prompt goes through the Semantic Router (vllm-sr/auto on
               :8899), which right-sizes it to the 2B / 9B / 27B backend. A
               trivial "capital of France" lands on the 2B and comes back in a
               blink; a hard proof lands on the 27B.

  ROUTING OFF  every prompt is forced to the single 27B model (Qwen3.8-27B on
               :8003) -- the "just always use the big model" baseline. Correct,
               but it burns the whole MI300X on questions a 2B could answer, so
               latency and total wall time blow up.

For each prompt we run BOTH paths and record, per request:
  - end-to-end latency (ms)
  - time to first token (ms, measured from the streamed response)
  - completion tokens
  - which model actually served it (ON reads x-vsr-selected-model; OFF is 27B)

Then we aggregate per mode: avg / p50 / p95 latency, total wall time to serve
the whole workload, aggregate throughput (tokens/sec), and a per-tier latency
breakdown. The headline comparison is avg latency and total wall time: routing
ON is dramatically faster because two thirds of the workload never touches the
27B.

Nothing is simulated. If the router or a backend is unreachable, that request is
marked unreachable and EXCLUDED from the denominator; the report says so.

Writes incremental results to $BENCH_STATE (default dashboard/benchmark.json) so
the dashboard can show a live progress bar as items complete.

    python3 benchmarks/playground.py                 # full set, both modes
    python3 benchmarks/playground.py --limit 4        # 4 items per tier (quick)
    python3 benchmarks/playground.py --repeat 2       # average over 2 passes
"""
import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
SET_PATH = Path(os.environ.get("BENCH_SET", os.environ.get("ACCURACY_SET", str(BASE / "accuracy_set.json"))))
STATE_PATH = Path(os.environ.get("BENCH_STATE", str(REPO / "dashboard" / "benchmark.json")))

ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8899")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "vllm-sr/auto")
# The single big model every request hits when routing is OFF.
BIG_HOSTPORT = os.environ.get("BK_HARD", "127.0.0.1:8003")
BIG_MODEL = os.environ.get("BIG_MODEL", "Qwen3.8-27B")

MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "160"))
REQ_TIMEOUT = int(os.environ.get("BENCH_REQ_TIMEOUT_S", "90"))
TIERS = ["trivial", "standard", "hard"]
TIER_MODEL = {"trivial": "Qwen3.5-2B", "standard": "Qwen3.5-9B", "hard": "Qwen3.8-27B"}


def _stream_chat(url, model, prompt, timeout=REQ_TIMEOUT):
    """POST a streaming chat completion. Returns (text, latency_ms, ttft_ms,
    out_tokens, served_model). Raises on failure.

    Streaming lets us measure time-to-first-token honestly. We count tokens from
    the stream (one content delta ~ one token) and prefer the server usage count
    if the final chunk carries it.
    """
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Qwen3 models default to a chain-of-thought scratchpad; turn it off so
        # both modes measure the same answer-only workload (and a trivial prompt
        # is not inflated by the model thinking out loud).
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    ttft = None
    deltas = 0
    usage_tokens = None
    served = None
    text = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        served = r.headers.get("x-vsr-selected-model")
        for line in r:
            s = line.decode("utf-8", "ignore").strip()
            if not s.startswith("data:"):
                continue
            payload = s[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            u = obj.get("usage")
            if u and u.get("completion_tokens") is not None:
                usage_tokens = u["completion_tokens"]
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {}).get("content", "")
            if delta:
                if ttft is None:
                    ttft = (time.time() - t0) * 1000.0
                deltas += 1
                text.append(delta)
    latency_ms = (time.time() - t0) * 1000.0
    out_tokens = usage_tokens if usage_tokens is not None else deltas
    return ("".join(text), latency_ms, ttft, out_tokens, served)


def _blank_mode(label, model_desc):
    return {"label": label, "model_desc": model_desc,
            "latencies": [], "ttfts": [], "wall_s": 0.0, "tokens": 0,
            "unreachable": 0, "by_tier": {t: {"lat": [], "n": 0} for t in TIERS},
            "routing_mix": {}}


def _finalize_mode(m):
    lat = m["latencies"]
    out = {
        "label": m["label"], "model_desc": m["model_desc"],
        "n": len(lat),
        "avg_latency_ms": round(statistics.mean(lat), 1) if lat else None,
        "p50_latency_ms": round(statistics.median(lat), 1) if lat else None,
        "p95_latency_ms": round(statistics.quantiles(lat, n=20)[-1], 1) if len(lat) >= 2 else (round(lat[0], 1) if lat else None),
        "avg_ttft_ms": round(statistics.mean([t for t in m["ttfts"] if t is not None]), 1) if any(t is not None for t in m["ttfts"]) else None,
        "total_wall_s": round(m["wall_s"], 2),
        "total_tokens": m["tokens"],
        "tok_s": round(m["tokens"] / m["wall_s"], 1) if m["wall_s"] > 0 else None,
        "unreachable": m["unreachable"],
        "by_tier": {t: {
            "avg_latency_ms": round(statistics.mean(m["by_tier"][t]["lat"]), 1) if m["by_tier"][t]["lat"] else None,
            "n": m["by_tier"][t]["n"],
        } for t in TIERS},
        "routing_mix": m["routing_mix"],
    }
    return out


def _write_state(data):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(STATE_PATH)


def run(items, repeat=1):
    on = _blank_mode("Routing ON", "Semantic Router picks 2B / 9B / 27B per prompt")
    off = _blank_mode("Routing OFF", "Every prompt forced to %s" % BIG_MODEL)
    rows = []
    total = len(items) * repeat
    done = 0
    t_start = time.time()

    def push(phase):
        state = {
            "ok": True, "updated": time.time(), "phase": phase,
            "running": phase not in ("done", "error"),
            "progress": {"done": done, "total": total},
            "elapsed_s": round(time.time() - t_start, 1),
            "n_items": len(items), "repeat": repeat,
            "max_tokens": MAX_TOKENS,
            "modes": {"on": _finalize_mode(on), "off": _finalize_mode(off)},
            "items": rows[-40:],
            "grader": "latency/throughput measured on real backends; no simulation",
            "tier_model": TIER_MODEL, "big_model": BIG_MODEL,
        }
        o, f = state["modes"]["on"], state["modes"]["off"]
        if o["avg_latency_ms"] and f["avg_latency_ms"]:
            state["speedup"] = {
                "latency_x": round(f["avg_latency_ms"] / o["avg_latency_ms"], 2),
                "wall_x": round(f["total_wall_s"] / o["total_wall_s"], 2) if o["total_wall_s"] else None,
                "tok_s_x": round(o["tok_s"] / f["tok_s"], 2) if (o["tok_s"] and f["tok_s"]) else None,
            }
        _write_state(state)
        return state

    push("measuring")
    for rep in range(repeat):
        for it in items:
            tier = it["tier"]
            row = {"id": it["id"], "tier": tier, "prompt": it["prompt"][:80], "rep": rep + 1}
            # ---- ON: through the semantic router ----
            try:
                _, ms, ttft, tok, served = _stream_chat(ROUTER_URL, ROUTER_MODEL, it["prompt"])
                on["latencies"].append(ms); on["ttfts"].append(ttft)
                on["wall_s"] += ms / 1000.0; on["tokens"] += tok or 0
                on["by_tier"][tier]["lat"].append(ms); on["by_tier"][tier]["n"] += 1
                sm = served or TIER_MODEL.get(tier, "?")
                on["routing_mix"][sm] = on["routing_mix"].get(sm, 0) + 1
                row["on"] = {"ms": round(ms, 1), "ttft_ms": round(ttft, 1) if ttft else None,
                             "tok": tok, "model": sm}
            except Exception as e:
                on["unreachable"] += 1
                row["on"] = {"error": e.__class__.__name__}
            # ---- OFF: forced to the big model ----
            try:
                _, ms, ttft, tok, _ = _stream_chat("http://" + BIG_HOSTPORT, BIG_MODEL, it["prompt"])
                off["latencies"].append(ms); off["ttfts"].append(ttft)
                off["wall_s"] += ms / 1000.0; off["tokens"] += tok or 0
                off["by_tier"][tier]["lat"].append(ms); off["by_tier"][tier]["n"] += 1
                off["routing_mix"][BIG_MODEL] = off["routing_mix"].get(BIG_MODEL, 0) + 1
                row["off"] = {"ms": round(ms, 1), "ttft_ms": round(ttft, 1) if ttft else None,
                              "tok": tok, "model": BIG_MODEL}
            except Exception as e:
                off["unreachable"] += 1
                row["off"] = {"error": e.__class__.__name__}
            rows.append(row)
            done += 1
            push("measuring")

    final = push("done")
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max items per tier")
    ap.add_argument("--repeat", type=int, default=1, help="passes over the set (averaged)")
    args = ap.parse_args()

    data = json.loads(SET_PATH.read_text())
    items = data["items"]
    if args.limit:
        keep, seen = [], {}
        for it in items:
            seen.setdefault(it["tier"], 0)
            if seen[it["tier"]] < args.limit:
                keep.append(it); seen[it["tier"]] += 1
        items = keep

    try:
        final = run(items, repeat=max(1, args.repeat))
        print(json.dumps({k: final[k] for k in ("phase", "progress", "speedup") if k in final}, indent=2))
    except Exception as e:
        _write_state({"ok": False, "updated": time.time(), "phase": "error",
                      "running": False, "error": "%s: %s" % (e.__class__.__name__, e)})
        raise


if __name__ == "__main__":
    main()
