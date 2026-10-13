#!/usr/bin/env python3
"""
OpenClaw REAL swarm orchestrator for the ai-Pulse 2026 booth.

Drives N real OpenClaw agents concurrently against the LOCAL vLLM on the MI300X.
Every agent is a distinct persona that performs a real LLM task via
`openclaw agent exec --json`. We capture REAL token usage, latency and output
text from the agent-exec JSON envelope and publish it to state.json, which the
booth dashboard renders live.

No external APIs, no simulation. If vLLM is down, agents error honestly and the
dashboard shows it.

Task (self-contained, needs no network): given a rotating booth TOPIC, each
persona produces its own real contribution (an angle, a critique, a one-liner,
a French translation, a risk, a hashtag ...). The merge persona then summarises.
"""
import concurrent.futures as cf
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path

STATE = Path(os.environ.get("SWARM_STATE", "/opt/openclaw-swarm/state.json"))
MODEL = os.environ.get("SWARM_MODEL", "vllm/Qwen3.5-9B")
STATE_ROOT = Path(os.environ.get("SWARM_STATE_ROOT", "/opt/openclaw-swarm/agent-state"))
MAX_WORKERS = int(os.environ.get("SWARM_CONCURRENCY", "8"))
MODELS_SIDECAR = Path(os.environ.get("SWARM_MODELS_SIDECAR", "/opt/openclaw-swarm/agent-models.json"))
TURN_TIMEOUT = int(os.environ.get("SWARM_TURN_TIMEOUT", "150"))

TOPICS = [
    "running local LLMs on a single AMD Instinct MI300X at the ai-Pulse 2026 booth",
    "why data sovereignty matters for enterprises adopting on-prem AI",
    "multi-agent AI systems coordinating on one GPU",
    "ROCm as an open software stack for AI inference",
    "the 192 GB HBM3 advantage of the MI300X for big-context agents",
]

# persona id, display, role, task template ({topic} injected)
PERSONAS = [
    ("scout", "Scout", "trend hunter", "In ONE punchy sentence, surface the most exciting angle about: {topic}."),
    ("analyst", "Analyst", "sentiment", "In ONE sentence, give a crisp analytical take on: {topic}."),
    ("critic", "Critic", "fact-check", "In ONE sentence, raise a fair, skeptical question about: {topic}."),
    ("poet", "Poet", "copywriter", "Write ONE short, catchy booth tagline (max 10 words) about: {topic}."),
    ("translator", "Translator", "FR/EN", "Translate this idea into ONE natural French sentence: {topic}."),
    ("qa", "QA", "verifier", "In ONE sentence, state how you would verify a claim about: {topic}."),
    ("coder", "Coder", "snippets", "In ONE sentence, name a concrete code/tooling step a developer takes for: {topic}."),
    ("summarizer", "Summarizer", "digest", "In ONE sentence, give the elevator-pitch summary of: {topic}."),
    ("researcher", "Researcher", "sources", "In ONE sentence, name what evidence would prove the value of: {topic}."),
    ("strategist", "Strategist", "ranking", "In ONE sentence, give the top strategic reason to care about: {topic}."),
    ("debater", "Debater", "counterpoint", "In ONE sentence, give the strongest counterpoint about: {topic}."),
    ("visionary", "Visionary", "ideation", "In ONE sentence, paint the 3-year future of: {topic}."),
]

ROSTER = [dict(id=p[0], name=p[1], role=p[2], task=p[3]) for p in PERSONAS]


def _now():
    return time.time()


def _write(state):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(STATE)


def _clean_final(text):
    """Strip Harmony-style reasoning channel scaffolding from Muse Glimmer output.

    Glimmer emits chain-of-thought then the real answer after an
    `assistant to=user` (or `assistant` / `final`) channel marker. Keep only
    the user-facing answer; if no marker is present, return text unchanged.
    """
    if not text:
        return text
    t = text
    # Preferred: content after the LAST 'assistant to=user' marker.
    m = list(re.finditer(r"assistant\s+to=user\s*", t))
    if m:
        t = t[m[-1].end():]
    else:
        # Fallbacks for other channel spellings.
        m2 = list(re.finditer(r"(?:^|\n)\s*(?:assistant|final)\b[:>]?\s*", t))
        if m2:
            t = t[m2[-1].end():]
    # Drop any leading 'to=self ...' preamble line if it survived.
    t = re.sub(r"^\s*to=self\b.*?(?:\n|$)", "", t, count=1)
    return t.strip()


def run_agent(persona, topic):
    """Run ONE real OpenClaw agent turn against local vLLM. Return real metrics."""
    # Ground every prompt to the AMD/ROCm context so the model never drifts to
    # competitor hardware. Kept terse on purpose: the OpenClaw system scaffold is
    # ~8.4K tokens and 95% prefix-cached, so a short, STABLE guard string reuses
    # that cache instead of forcing re-prefill. Keep this wording constant.
    guard = "Context: AMD Instinct MI300X + ROCm only; never mention NVIDIA/CUDA. "
    prompt = guard + persona["task"].format(topic=topic)
    sdir = STATE_ROOT / persona["id"]
    sdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Per-agent API key so the model-tap proxy can attribute the REAL routed
    # model to THIS agent. The router ignores the key value (local auth); the
    # tap reads the `local-vllm.<id>` suffix to tag the selected model.
    _base_key = os.environ.get("VLLM_API_KEY", "local-vllm").split(".")[0]
    env["VLLM_API_KEY"] = f"{_base_key}.{persona['id']}"
    t0 = _now()
    try:
        proc = subprocess.run(
            ["openclaw", "agent", "exec",
             "--model", MODEL, "--local-model-lean", "--json",
             "--thinking", os.environ.get("SWARM_THINKING", "off"),
             "--state-dir", str(sdir),
             "--timeout", str(TURN_TIMEOUT), prompt],
            capture_output=True, text=True, timeout=TURN_TIMEOUT + 20, env=env,
            cwd="/tmp")
        raw = proc.stdout
        env_ok, final, usage = False, "", {}
        d = None
        # stdout is the JSON envelope (diagnostics go to stderr). Parse robustly.
        try:
            d = json.loads(raw.strip())
        except Exception:
            # fallback: slice from first top-level brace to end
            i = raw.find("{")
            if i >= 0:
                try:
                    d = json.loads(raw[i:])
                except Exception:
                    d = None
        if d is not None:
            env_ok = bool(d.get("ok"))
            final = d.get("final") or ""
            if not final and d.get("payloads"):
                final = " ".join(p.get("text", "") for p in d["payloads"] if p.get("text"))
            final = _clean_final(final)
            usage = d.get("usage", {}) or {}
        dt = _now() - t0
        out_tok = int(usage.get("output", 0) or 0)
        in_tok = int(usage.get("input", 0) or 0)
        tps = round(out_tok / dt, 1) if dt > 0 and out_tok else 0.0
        return {
            "id": persona["id"], "ok": env_ok, "final": final.strip()[:400],
            "in_tok": in_tok, "out_tok": out_tok, "latency_s": round(dt, 1),
            "tok_s": tps, "ts": _now(),
        }
    except subprocess.TimeoutExpired:
        return {"id": persona["id"], "ok": False, "final": "(timeout)",
                "in_tok": 0, "out_tok": 0, "latency_s": round(_now() - t0, 1),
                "tok_s": 0.0, "ts": _now()}
    except Exception as e:
        return {"id": persona["id"], "ok": False, "final": f"(error: {e.__class__.__name__})",
                "in_tok": 0, "out_tok": 0, "latency_s": round(_now() - t0, 1),
                "tok_s": 0.0, "ts": _now()}


def main():
    """Continuous rolling swarm: keep the GPU fed with NO barrier and NO sleep.

    The old design fired all 12 agents, waited for the SLOWEST (barrier), slept
    2s, then repeated, leaving the GPU idle ~52% of the time (measured). Here we
    keep a rolling pool: the moment any agent finishes we immediately resubmit it
    with the next topic, so there is always work in flight. `round` now means a
    full lap in which every persona has produced at least once (for the booth UI).
    """
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    results = {p["id"]: {"id": p["id"], "name": p["name"], "role": p["role"],
                         "status": "thinking", "final": "", "out_tok": 0, "in_tok": 0,
                         "tok_s": 0.0, "latency_s": 0.0, "ts": 0} for p in ROSTER}
    PERSONA = {p["id"]: p for p in ROSTER}
    cumulative_tok = 0
    cumulative_calls = 0
    # per-persona topic cursor so each agent rotates through topics independently
    topic_idx = {p["id"]: i for i, p in enumerate(ROSTER)}
    completed_total = 0
    started = _now()

    def _topic_for(pid):
        t = TOPICS[topic_idx[pid] % len(TOPICS)]
        topic_idx[pid] += 1
        return t

    def _publish():
        # live aggregate throughput = sum of agents currently producing
        agg = round(sum(results[p["id"]]["tok_s"] for p in ROSTER
                        if results[p["id"]]["status"] in ("thinking", "done")), 1)
        # rolling "round" = full laps completed by the whole roster
        round_no = 1 + completed_total // len(ROSTER)
        # Merge the REAL routed model per agent, captured by the model-tap proxy.
        # Best-effort: a missing/old sidecar must never break publishing.
        models = {}
        try:
            models = json.loads(MODELS_SIDECAR.read_text())
        except Exception:
            models = {}
        for p in ROSTER:
            rec = models.get(p["id"]) if isinstance(models, dict) else None
            if isinstance(rec, dict) and rec.get("model"):
                results[p["id"]]["model"] = rec["model"]
                results[p["id"]]["decision"] = rec.get("decision")
        # current headline topic = the most recently assigned one
        state = {"round": round_no, "topic": results.get("_topic", ""),
                 "model": MODEL, "updated": _now(), "cloud_calls": 0,
                 "agg_tok_s": agg,
                 "cumulative_tokens": cumulative_tok,
                 "cumulative_calls": cumulative_calls,
                 "uptime_s": round(_now() - started, 1),
                 "agents": [results[p["id"]] for p in ROSTER]}
        _write(state)

    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        # prime the pool: one in-flight task per persona
        inflight = {}
        for p in ROSTER:
            tp = _topic_for(p["id"])
            results["_topic"] = tp
            results[p["id"]]["status"] = "thinking"
            inflight[ex.submit(run_agent, p, tp)] = p["id"]
        _publish()

        # rolling completion: as each finishes, record + immediately resubmit
        while inflight:
            done, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
            for fut in done:
                pid = inflight.pop(fut)
                r = fut.result()
                base = results[pid]
                base.update({"status": "done" if r["ok"] else "error",
                             "final": r["final"], "out_tok": r["out_tok"],
                             "in_tok": r["in_tok"], "tok_s": r["tok_s"],
                             "latency_s": r["latency_s"], "ts": r["ts"]})
                cumulative_tok += r["out_tok"]
                cumulative_calls += 1
                completed_total += 1
                _publish()
                # resubmit the SAME persona on the next topic, no sleep, no barrier
                tp = _topic_for(pid)
                results["_topic"] = tp
                p = PERSONA[pid]
                # brief 'done' flash, then back to thinking on next publish
                results[pid]["status"] = "thinking"
                inflight[ex.submit(run_agent, p, tp)] = pid


if __name__ == "__main__":
    main()
