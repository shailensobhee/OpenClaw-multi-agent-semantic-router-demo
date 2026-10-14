#!/usr/bin/env python3
"""
Accuracy harness for the booth demo (real measurement, no judge model, no fabrication).

Two things it measures against benchmarks/accuracy_set.json:

  (1) ANSWER ACCURACY per backend  -- Option 1
      Send EVERY benchmark item DIRECTLY to each of the three Qwen backends
      (2B :8001, 9B :8002, 27B :8003), grade the output deterministically, and
      report accuracy% per model, broken down by tier. This is the "right-sizing"
      story: the 2B aces trivial but fails hard reasoning; the 27B nails hard.

  (2) ROUTING ACCURACY of Vela 2.0 -- Option 3
      Send every item THROUGH the Semantic Router (model=vllm-sr/auto, max_tokens=1
      so only the decision matters) and read the x-vsr-selected-decision header.
      Compare the router's chosen tier to the item's human tier label. This scores
      the decision model, not the answers.

Grading is exact/contains/numeric/fraction only -- fully reproducible. If a backend
or the router is down, the item is marked unreachable and EXCLUDED from the
denominator (never counted as right or wrong), and the report says so honestly.

Writes results to $ACCURACY_STATE (default dashboard/accuracy.json) so the Flask
dashboard can serve them. Run on a schedule or on demand:

    python3 benchmarks/accuracy.py                 # full run, all backends + routing
    python3 benchmarks/accuracy.py --answers-only
    python3 benchmarks/accuracy.py --routing-only
    python3 benchmarks/accuracy.py --limit 3       # smoke test: 3 items per tier
"""
import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
REPO = BASE.parent
SET_PATH = Path(os.environ.get("ACCURACY_SET", str(BASE / "accuracy_set.json")))
STATE_PATH = Path(os.environ.get("ACCURACY_STATE", str(REPO / "dashboard" / "accuracy.json")))

ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8899")
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "vllm-sr/auto")

# tier -> (backend host:port, served-model-name shown on the board)
BACKENDS = {
    "trivial":  {"tier_label": "2B",  "hostport": os.environ.get("BK_TRIVIAL",  "127.0.0.1:8001"), "model": "Qwen3.5-2B"},
    "standard": {"tier_label": "9B",  "hostport": os.environ.get("BK_STANDARD", "127.0.0.1:8002"), "model": "Qwen3.5-9B"},
    "hard":     {"tier_label": "27B", "hostport": os.environ.get("BK_HARD",     "127.0.0.1:8003"), "model": "Qwen3.8-27B"},
}
# the three backends, in board order (every backend answers every item)
FLEET = [
    {"key": "2B",  "hostport": BACKENDS["trivial"]["hostport"],  "model": "Qwen3.5-2B",  "owns": "trivial"},
    {"key": "9B",  "hostport": BACKENDS["standard"]["hostport"], "model": "Qwen3.5-9B",  "owns": "standard"},
    {"key": "27B", "hostport": BACKENDS["hard"]["hostport"],     "model": "Qwen3.8-27B", "owns": "hard"},
]

# map a router decision string -> our tier vocabulary. Decisions look like
# "hard-to-27b", "trivial-to-2b", "default-route". Robust to naming drift.
def decision_to_tier(decision: str):
    if not decision:
        return None
    d = decision.lower()
    if "hard" in d or "27b" in d or "32b" in d:
        return "hard"
    if "trivial" in d or "2b" in d:
        return "trivial"
    if "default" in d or "standard" in d or "9b" in d or "7b" in d:
        return "standard"
    return None


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"[^a-z0-9/\u00b2\u2082 ]+", "", s).strip()


def _last_number(s: str):
    nums = re.findall(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
    return float(nums[-1]) if nums else None


def grade(item: dict, output: str) -> bool:
    g = item["grader"]
    ans = item["answer"]
    out = output or ""
    if g == "numeric":
        n = _last_number(out)
        return n is not None and any(abs(n - float(a)) < 1e-6 for a in ans)
    if g == "fraction":
        n = _last_number(out)
        no = _norm(out)
        hit_frac = any(isinstance(a, str) and _norm(a) in no for a in ans)
        hit_dec = n is not None and any(isinstance(a, (int, float)) and abs(n - float(a)) < 1e-3 for a in ans)
        return hit_frac or hit_dec
    if g == "exact":
        no = _norm(out)
        toks = no.split()
        last = toks[-1] if toks else no
        return any(_norm(str(a)) == no or _norm(str(a)) == last for a in ans)
    # default: contains
    no = _norm(out)
    return any(_norm(str(a)) in no for a in ans)


def ask_backend(hostport: str, model: str, prompt: str, timeout=60):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 160, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"http://{hostport}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    txt = d["choices"][0]["message"]["content"]
    return txt, (time.time() - t0) * 1000.0, d.get("usage", {}).get("completion_tokens")


def ask_router_decision(prompt: str, timeout=30):
    body = json.dumps({
        "model": ROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(f"{ROUTER_URL}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        hdr = {k.lower(): v for k, v in r.headers.items()}
        r.read()
    return hdr.get("x-vsr-selected-decision"), hdr.get("x-vsr-selected-model")


def run(items, do_answers=True, do_routing=True):
    tiers = ["trivial", "standard", "hard"]
    # ---- (1) answer accuracy: every backend answers every item ----
    answers = {}
    if do_answers:
        for bk in FLEET:
            per_tier = {t: {"right": 0, "total": 0, "unreachable": 0} for t in tiers}
            lat = []
            for it in items:
                tier = it["tier"]
                per_tier[tier]["total"] += 1
                try:
                    out, ms, _ = ask_backend(bk["hostport"], bk["model"], it["prompt"])
                    lat.append(ms)
                    if grade(it, out):
                        per_tier[tier]["right"] += 1
                except Exception:
                    per_tier[tier]["unreachable"] += 1
                    per_tier[tier]["total"] -= 1  # exclude from denominator, never guess
            graded = sum(per_tier[t]["total"] for t in tiers)
            correct = sum(per_tier[t]["right"] for t in tiers)
            answers[bk["key"]] = {
                "model": bk["model"], "owns_tier": bk["owns"],
                "overall_pct": round(100.0 * correct / graded, 1) if graded else None,
                "graded": graded, "correct": correct,
                "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
                "by_tier": {t: {
                    "pct": round(100.0 * per_tier[t]["right"] / per_tier[t]["total"], 1) if per_tier[t]["total"] else None,
                    "right": per_tier[t]["right"], "total": per_tier[t]["total"],
                    "unreachable": per_tier[t]["unreachable"],
                } for t in tiers},
            }
    # ---- (3) routing accuracy: router tier vs human label ----
    routing = {}
    if do_routing:
        right = 0; graded = 0; unreachable = 0
        conf = {t: {"right": 0, "total": 0} for t in tiers}
        mistakes = []
        for it in items:
            try:
                decision, model = ask_router_decision(it["prompt"])
                picked = decision_to_tier(decision)
                if picked is None:
                    unreachable += 1
                    continue
                graded += 1
                conf[it["tier"]]["total"] += 1
                if picked == it["tier"]:
                    right += 1
                    conf[it["tier"]]["right"] += 1
                elif len(mistakes) < 12:
                    mistakes.append({"id": it["id"], "prompt": it["prompt"][:70],
                                     "expected": it["tier"], "router_chose": picked,
                                     "decision": decision, "model": model})
            except Exception:
                unreachable += 1
        routing = {
            "overall_pct": round(100.0 * right / graded, 1) if graded else None,
            "right": right, "graded": graded, "unreachable": unreachable,
            "by_tier": {t: {
                "pct": round(100.0 * conf[t]["right"] / conf[t]["total"], 1) if conf[t]["total"] else None,
                "right": conf[t]["right"], "total": conf[t]["total"],
            } for t in tiers},
            "mistakes": mistakes,
        }
    return answers, routing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers-only", action="store_true")
    ap.add_argument("--routing-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max items per tier (smoke test)")
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

    do_answers = not args.routing_only
    do_routing = not args.answers_only
    t0 = time.time()
    answers, routing = run(items, do_answers, do_routing)
    state = {
        "ok": True,
        "updated": time.time(),
        "duration_s": round(time.time() - t0, 1),
        "set_version": data.get("version"),
        "n_items": len(items),
        "answers": answers,     # Option 1
        "routing": routing,     # Option 3
        "grader": "deterministic (exact / contains / numeric / fraction) -- no judge model",
    }
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
