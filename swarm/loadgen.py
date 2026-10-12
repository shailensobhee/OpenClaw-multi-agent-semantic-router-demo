#!/usr/bin/env python3
"""GPU saturation load generator for the ai-Pulse 2026 booth.

Separate from the OpenClaw agent demo. Keeps a fixed number of chat completions
ALWAYS in flight against the Semantic Router (:8899), so the MI300X stays busy
and the booth can show "one GPU, pinned". Honest: every request is a real
completion; it writes live stats to loadgen.json for the dashboard.

Toggle-able: responds to a control file so the booth can turn saturation on/off
without restarting. Env:
  LOADGEN_URL         router OpenAI endpoint (default http://127.0.0.1:8899/v1)
  LOADGEN_MODEL       model name (default vllm-sr/auto)
  LOADGEN_CONCURRENCY in-flight requests to maintain (default 24)
  LOADGEN_MAX_TOKENS  completion length per request (default 128)
  LOADGEN_STATE       stats file (default /opt/openclaw-swarm/loadgen.json)
  LOADGEN_CONTROL     control file; if it contains 'off', generator idles
"""
import json, os, time, threading, urllib.request, random
from pathlib import Path

URL = os.environ.get("LOADGEN_URL", "http://127.0.0.1:8899/v1").rstrip("/")
MODEL = os.environ.get("LOADGEN_MODEL", "vllm-sr/auto")
CONC = int(os.environ.get("LOADGEN_CONCURRENCY", "24"))
MAXTOK = int(os.environ.get("LOADGEN_MAX_TOKENS", "128"))
STATE = Path(os.environ.get("LOADGEN_STATE", "/opt/openclaw-swarm/loadgen.json"))
CONTROL = Path(os.environ.get("LOADGEN_CONTROL", "/opt/openclaw-swarm/loadgen.control"))

# A spread of prompts so routing hits all three tiers and prefill varies.
PROMPTS = [
    "Summarize in one sentence why on-prem AI matters.",
    "Prove step by step that the sum of the first n odd numbers is n squared.",
    "Analyze the trade-off between model size and latency for agent workloads.",
    "Say hello and wish the ai-Pulse 2026 booth visitors a great day.",
    "Design an algorithm to schedule many agents on one GPU; explain the reasoning.",
    "Give one crisp fact about the AMD Instinct MI300X memory capacity.",
]

_lock = threading.Lock()
_stats = {"completed": 0, "inflight": 0, "errors": 0, "gen_tokens": 0,
          "started": time.time(), "last_latency_ms": 0.0, "enabled": True}


def _enabled():
    try:
        if CONTROL.exists() and CONTROL.read_text().strip().lower() == "off":
            return False
    except Exception:
        pass
    return True


def _one_request():
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": random.choice(PROMPTS)}],
        "max_tokens": MAXTOK, "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(URL + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=120))
        dt = (time.time() - t0) * 1000.0
        ct = int((r.get("usage") or {}).get("completion_tokens", 0) or 0)
        with _lock:
            _stats["completed"] += 1
            _stats["gen_tokens"] += ct
            _stats["last_latency_ms"] = round(dt, 0)
    except Exception:
        with _lock:
            _stats["errors"] += 1


def _worker():
    while True:
        if not _enabled():
            with _lock:
                _stats["enabled"] = False
            time.sleep(0.5)
            continue
        with _lock:
            _stats["enabled"] = True
            _stats["inflight"] += 1
        _one_request()
        with _lock:
            _stats["inflight"] -= 1


def _publisher():
    prev_tok, prev_t = 0, time.time()
    while True:
        time.sleep(1.0)
        with _lock:
            now = time.time()
            dt = now - prev_t
            tps = round((_stats["gen_tokens"] - prev_tok) / dt, 1) if dt > 0 else 0.0
            prev_tok, prev_t = _stats["gen_tokens"], now
            out = {
                "ok": True,
                "enabled": _stats["enabled"],
                "concurrency_target": CONC,
                "inflight": _stats["inflight"],
                "completed": _stats["completed"],
                "errors": _stats["errors"],
                "gen_tok_s": tps,
                "last_latency_ms": _stats["last_latency_ms"],
                "uptime_s": round(now - _stats["started"], 1),
                "updated": now,
            }
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(out))
        tmp.replace(STATE)


def main():
    STATE.parent.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_publisher, daemon=True).start()
    workers = [threading.Thread(target=_worker, daemon=True) for _ in range(CONC)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()


if __name__ == "__main__":
    main()
