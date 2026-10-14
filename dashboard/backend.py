#!/usr/bin/env python3
"""
OpenClaw Swarm booth backend (:7000).

Serves the swarm wall-display page AND exposes REAL telemetry so booth visitors
can see exactly what is running:
  GET /                -> the demo page (index.html)
  GET /api/models      -> live model list from the local vLLM (/v1/models)
  GET /api/gpu         -> live rocm-smi (VRAM used/total, utilisation)
  GET /api/health      -> is vLLM actually up and serving?
  GET /api/logs/stream -> Server-Sent Events: live `docker logs -f vllm`

Nothing here is simulated. If vLLM is down, the endpoints say so honestly.
"""
import json
import os
import re
import subprocess
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

BASE = Path(__file__).resolve().parent
VLLM_URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
VLLM_CONTAINER = os.environ.get("VLLM_CONTAINER", "vllm")

app = Flask(__name__)


@app.route("/")
def index():
    # No-cache so the booth wall display always pulls the freshest UI after a deploy.
    resp = send_from_directory(BASE, "index.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/health")
def health():
    import urllib.request
    try:
        with urllib.request.urlopen(f"{VLLM_URL}/health", timeout=3) as r:
            up = r.status == 200
    except Exception:
        up = False
    return jsonify({"vllm_up": up, "vllm_url": VLLM_URL, "ts": time.time()})


@app.route("/api/models")
def models():
    """Show the REAL backend fleet the router chooses between, not the router's
    virtual aliases.

    The router's own /v1/models only advertises the meta-models (vllm-sr/auto,
    auto, MoM) which hold no weights, so listing those tells a booth visitor
    nothing. Instead we probe each real vLLM backend directly (loopback ports the
    fleet publishes) and report the actual Qwen model + its routing tier. We also
    surface the router aliases separately so the "how" is still visible.
    """
    import urllib.request

    # tier label -> (loopback port, human role). Override via FLEET_BACKENDS
    # as "name=host:port=role,..." if the fleet changes.
    fleet = os.environ.get(
        "FLEET_BACKENDS",
        "2B=127.0.0.1:8001=fast / trivial,"
        "9B=127.0.0.1:8002=standard,"
        "27B=127.0.0.1:8003=hard / reasoning")

    def _probe(hostport):
        try:
            with urllib.request.urlopen(f"http://{hostport}/v1/models", timeout=3) as r:
                data = json.loads(r.read().decode())
            d = data.get("data", [])
            if d:
                return d[0].get("id"), d[0].get("max_model_len")
        except Exception:
            pass
        return None, None

    details, served = [], []
    for item in [x for x in fleet.split(",") if x.strip()]:
        parts = item.split("=")
        tier = parts[0].strip()
        hostport = parts[1].strip() if len(parts) > 1 else ""
        role = parts[2].strip() if len(parts) > 2 else ""
        mid, mml = _probe(hostport)
        up = mid is not None
        details.append({"id": mid or tier, "tier": tier, "role": role,
                        "max_model_len": mml, "up": up,
                        "owned_by": "vLLM (local)"})
        if up:
            served.append(mid)

    # router virtual aliases (the "how"), best-effort, non-fatal.
    aliases = []
    try:
        with urllib.request.urlopen(f"{ROUTER_URL}/v1/models", timeout=3) as r:
            aliases = [m.get("id") for m in json.loads(r.read().decode()).get("data", [])]
    except Exception:
        pass

    ok = len(served) > 0
    return jsonify({"ok": ok, "served_by": "vLLM (local) via Semantic Router",
                    "models": served, "details": details,
                    "router_aliases": aliases,
                    "error": None if ok else "no backends reachable"}), 200


def _amdgpu_card_path():
    """Find the amdgpu card sysfs dir (the one with gpu_busy_percent). Cached."""
    import glob
    for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
        if os.path.exists(os.path.join(dev, "gpu_busy_percent")):
            return dev
    return None


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _rocm_smi():
    """Return real HBM + compute util + power + temp for the MI300X.

    Primary source is amdgpu SYSFS, not `rocm-smi`: on this host's ROCm build
    `rocm-smi --showuse`/`--showtemp` crash with an assertion (exit 134, core
    dump) in GetGPUMetricsFormat1, and because the old code bundled every flag
    into ONE call, that crash took HBM and compute down with it. Sysfs nodes
    (gpu_busy_percent, mem_info_vram_*, hwmon temp/power) never core-dump, so we
    read those directly. rocm-smi is used only as a best-effort fallback for any
    field sysfs cannot provide, each in its OWN isolated call so one failure
    cannot poison the others.
    """
    dev = _amdgpu_card_path()
    if not dev:
        return {"ok": False, "error": "amdgpu sysfs not found"}

    used = _read_int(os.path.join(dev, "mem_info_vram_used"))
    total = _read_int(os.path.join(dev, "mem_info_vram_total"))
    busy = _read_int(os.path.join(dev, "gpu_busy_percent"))

    # temperature + power from hwmon (junction preferred), microdegree/microwatt.
    temp_c = power_w = None
    import glob
    for h in glob.glob(os.path.join(dev, "hwmon", "hwmon*")):
        # pick junction temp if labelled, else first temp sensor
        best = None
        for lbl in glob.glob(os.path.join(h, "temp*_label")):
            try:
                if open(lbl).read().strip().lower() == "junction":
                    best = lbl.replace("_label", "_input")
                    break
            except Exception:
                pass
        if best is None:
            inp = sorted(glob.glob(os.path.join(h, "temp*_input")))
            best = inp[0] if inp else None
        if best:
            v = _read_int(best)
            if v is not None:
                temp_c = round(v / 1000.0, 1)
        for pf in ("power1_average", "power1_input"):
            v = _read_int(os.path.join(h, pf))
            if v is not None:
                power_w = round(v / 1e6, 1)
                break
        if temp_c is not None or power_w is not None:
            break

    # best-effort power fallback via an ISOLATED rocm-smi call (never bundled).
    if power_w is None:
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showpower", "--json"],
                stderr=subprocess.DEVNULL, timeout=6).decode()
            card = next(iter(json.loads(out).values()), {})
            for k, val in card.items():
                if "power" in k.lower():
                    try:
                        power_w = round(float(str(val).strip()), 1)
                    except Exception:
                        pass
        except Exception:
            pass

    if used is None or total is None:
        return {"ok": False, "error": "amdgpu vram sysfs unreadable"}

    return {"ok": True,
            "vram_used_gb": round(used / 1e9, 1),
            "vram_total_gb": round(total / 1e9, 1),
            "util_pct": int(busy) if busy is not None else 0,
            "power_w": power_w,
            "temp_c": temp_c,
            "source": "amdgpu-sysfs"}


@app.route("/api/gpu")
def gpu():
    return jsonify(_rocm_smi())


def _vllm_counter():
    """Read vLLM's cumulative generation_tokens_total counter (Prometheus).

    The swarm routes through the Semantic Router across SEVERAL co-resident vLLM
    backends, and Envoy (:8899) does NOT expose vLLM's Prometheus /metrics. So we
    scrape each backend's own /metrics and SUM the fleet. Set VLLM_METRICS_URLS to
    a comma-separated list to override; default = the three Qwen backends.
    """
    import urllib.request
    urls = os.environ.get(
        "VLLM_METRICS_URLS",
        "http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003",
    ).split(",")
    gen_total = 0.0
    running_total = 0.0
    any_ok = False
    for base in urls:
        base = base.strip().rstrip("/")
        if not base:
            continue
        try:
            raw = urllib.request.urlopen(base + "/metrics", timeout=4).read().decode()
        except Exception:
            continue
        for line in raw.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith("vllm:generation_tokens_total"):
                gen_total += float(line.rsplit(" ", 1)[1]); any_ok = True
            elif line.startswith("vllm:num_requests_running"):
                running_total += float(line.rsplit(" ", 1)[1])
    if not any_ok:
        return None, 0.0
    return gen_total, running_total


@app.route("/api/vllm_throughput")
def vllm_throughput():
    """OPTION 1: true live server-side throughput = rate of vLLM's generation
    token counter sampled over a short interval. Non-zero whenever the GPU is
    actually generating, independent of the swarm's per-turn accounting."""
    g0, _ = _vllm_counter()
    if g0 is None:
        return jsonify({"ok": False, "tok_s": None, "error": "vllm /metrics unreachable"})
    t0 = time.time()
    time.sleep(1.0)
    g1, running = _vllm_counter()
    dt = time.time() - t0
    tps = round((g1 - g0) / dt, 1) if (g1 is not None and dt > 0) else 0.0
    if tps < 0:
        tps = 0.0
    return jsonify({"ok": True, "tok_s": tps, "requests_running": int(running)})


SWARM_STATE = Path(os.environ.get("SWARM_STATE", str(BASE / "state.json")))


@app.route("/api/swarm")
def swarm():
    """Live state written by the real OpenClaw swarm orchestrator (swarm.py)."""
    try:
        data = json.loads(SWARM_STATE.read_text())
        # freshness: if the orchestrator stopped writing, say so
        age = time.time() - data.get("updated", 0)
        data["stale"] = age > 90
        data["age_s"] = round(age, 1)
        return jsonify({"ok": True, **data})
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "swarm not started yet", "agents": []})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{e.__class__.__name__}", "agents": []})


LOADGEN_STATE = Path(os.environ.get("LOADGEN_STATE", str(BASE / "loadgen.json")))
LOADGEN_CONTROL = Path(os.environ.get("LOADGEN_CONTROL", str(BASE / "loadgen.control")))


# Router endpoint (Envoy OpenAI entrypoint). Defaults to VLLM_URL, which in the
# booth deployment already points at the Semantic Router on :8899.
ROUTER_URL = os.environ.get("ROUTER_URL", VLLM_URL)
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "vllm-sr/auto")

# Sample prompts for the routing visualiser, one per decision the router makes.
# These are illustrative booth prompts; the actual decision always comes from
# the live router headers, never hard-coded here.
ROUTE_SAMPLES = [
    {"label": "Trivial", "prompt": "hello there, thanks!"},
    {"label": "Standard", "prompt": "Summarize what AMD ROCm is in one sentence."},
    {"label": "Translation", "prompt": "Translate 'good morning' into natural French."},
    {"label": "Hard reasoning",
     "prompt": "Prove step by step that there are infinitely many primes, and analyze the argument."},
    {"label": "Code task",
     "prompt": "Refactor and optimize this loop; explain the algorithmic complexity change."},
]


@app.route("/api/route_probe")
def route_probe():
    """Send ONE prompt through the Semantic Router and report its REAL routing
    decision, captured from the router's x-vsr-* response headers. This is the
    data source for the live routing visualiser. Nothing is simulated: if the
    router is down, this says so.

    Query params:
      prompt=<text>   custom prompt (optional)
      sample=<int>    index into ROUTE_SAMPLES (optional; default rotates)
    """
    import urllib.request
    prompt = request.args.get("prompt", "").strip()
    if not prompt:
        try:
            idx = int(request.args.get("sample", int(time.time() // 3) % len(ROUTE_SAMPLES)))
        except ValueError:
            idx = 0
        idx = idx % len(ROUTE_SAMPLES)
        prompt = ROUTE_SAMPLES[idx]["prompt"]

    body = json.dumps({
        "model": ROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,          # we only care about the ROUTING decision, not the text
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{ROUTER_URL}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            hdr = {k.lower(): v for k, v in r.headers.items()}
            payload = json.loads(r.read().decode() or "{}")
        decision = hdr.get("x-vsr-selected-decision")
        model = hdr.get("x-vsr-selected-model") or payload.get("model")
        return jsonify({
            "ok": True,
            "prompt": prompt,
            "decision": decision,                               # e.g. hard-to-32b
            "model": model,                                     # e.g. Qwen3.8-27B
            "confidence": hdr.get("x-vsr-selected-confidence"),
            "algorithm": hdr.get("x-vsr-selected-algorithm"),   # static / classifier
            "route_latency_ms": hdr.get("x-vsr-routing-latency-ms"),
            "roundtrip_ms": round((time.time() - t0) * 1000, 1),
            "source": "x-vsr headers (live router)",
        })
    except Exception as e:
        return jsonify({"ok": False, "prompt": prompt,
                        "error": f"{e.__class__.__name__}: {e}",
                        "router_url": ROUTER_URL}), 200


@app.route("/api/route_samples")
def route_samples():
    """The list of sample prompts the visualiser can cycle through."""
    return jsonify({"ok": True, "samples": ROUTE_SAMPLES, "model": ROUTER_MODEL})


@app.route("/api/loadgen")
def loadgen():
    """Live stats from the GPU-saturation load generator (loadgen.py)."""
    try:
        data = json.loads(LOADGEN_STATE.read_text())
        age = time.time() - data.get("updated", 0)
        data["stale"] = age > 10
        data["age_s"] = round(age, 1)
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({"ok": False, "error": "loadgen not started"})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{e.__class__.__name__}"})


@app.route("/api/loadgen/toggle", methods=["POST", "GET"])
def loadgen_toggle():
    """Turn GPU saturation on/off without restarting the service."""
    want = request.args.get("state", "").lower()
    if want not in ("on", "off"):
        # flip based on current control file
        cur = ""
        try:
            cur = LOADGEN_CONTROL.read_text().strip().lower()
        except Exception:
            cur = "on"
        want = "off" if cur != "off" else "on"
    try:
        LOADGEN_CONTROL.write_text("off" if want == "off" else "on")
        return jsonify({"ok": True, "state": want})
    except Exception as e:
        return jsonify({"ok": False, "error": f"{e.__class__.__name__}"})


@app.route("/api/logs/stream")
def logs_stream():
    """Stream real `docker logs -f` from the live inference fleet as SSE.

    The swarm now routes across SEVERAL co-resident vLLM backends plus the
    Semantic Router, so we multiplex each container's logs into one stream with a
    short name prefix. Set VLLM_LOG_CONTAINERS to a comma-separated list to
    override; default = the three Qwen backends + the router (routing decisions).
    Only containers that actually exist are attached, so a missing one never
    spams "container not found".
    """
    import threading, queue, shutil
    names = os.environ.get(
        "VLLM_LOG_CONTAINERS",
        "qwen-05b,qwen-7b,qwen-32b,vllm-sr-router-container",
    ).split(",")
    # short, booth-friendly labels
    label = {
        "qwen-05b": "2B", "qwen-7b": "9B", "qwen-32b": "27B",
        "vllm-sr-router-container": "router",
    }

    def _exists(name):
        try:
            out = subprocess.check_output(
                ["docker", "ps", "--format", "{{.Names}}"],
                stderr=subprocess.DEVNULL, timeout=5).decode().split()
            return name in out
        except Exception:
            return False

    q = queue.Queue(maxsize=2000)
    procs = []
    attached = []

    def _pump(name, tag):
        try:
            p = subprocess.Popen(
                ["docker", "logs", "-f", "--tail", "20", name],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                universal_newlines=True)
            procs.append(p)
            for line in iter(p.stdout.readline, ""):
                line = line.rstrip("\n").replace("\r", "")
                if not line:
                    continue
                # Drop health/metrics access-log noise (the dashboard polls these
                # every second) so the booth sees MEANINGFUL inference lines:
                # chat completions, throughput stats, routing decisions.
                if ("/metrics" in line or "/health" in line
                        or "GET /v1/models" in line):
                    continue
                try:
                    q.put_nowait("[%s] %s" % (tag, line))
                except queue.Full:
                    pass
        except Exception:
            pass

    def generate():
        for name in [n.strip() for n in names if n.strip()]:
            tag = label.get(name, name)
            if _exists(name):
                attached.append(tag)
                threading.Thread(target=_pump, args=(name, tag), daemon=True).start()
        if attached:
            yield "event: hello\ndata: attached to live fleet: %s\n\n" % ", ".join(attached)
        else:
            yield "event: hello\ndata: no inference containers running yet\n\n"
        try:
            while True:
                try:
                    line = q.get(timeout=15)
                    yield "data: %s\n\n" % line
                except queue.Empty:
                    # heartbeat so the SSE connection + booth tile stay alive
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            for p in procs:
                try:
                    p.terminate()
                except Exception:
                    pass
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# =====================================================================
# Accuracy harness endpoints (Option 1 answer accuracy + Option 3 routing
# accuracy). The benchmark runner lives in benchmarks/accuracy.py and needs a
# clean GPU window (the swarm + loadgen pin the MI300X at 100%, which starves
# the backends), so the on-demand run pauses those services, measures, then
# resumes them. Results are cached to ACCURACY_STATE and served read-only.
# =====================================================================
import threading

ACCURACY_STATE = Path(os.environ.get("ACCURACY_STATE", str(BASE / "accuracy.json")))
ACCURACY_SET = Path(os.environ.get("ACCURACY_SET", str(BASE.parent / "benchmarks" / "accuracy_set.json")))
ACCURACY_SCRIPT = Path(os.environ.get("ACCURACY_SCRIPT", str(BASE.parent / "benchmarks" / "accuracy.py")))
# services paused for a clean measurement window (space-separated, overridable)
ACCURACY_PAUSE_UNITS = os.environ.get(
    "ACCURACY_PAUSE_UNITS",
    "booth-loadgen.service openclaw-swarm-runner.service openclaw-model-tap.service").split()

_accuracy_lock = threading.Lock()
_accuracy_running = {"on": False, "phase": "idle", "started": 0.0}


def _svc(action, units):
    """Best-effort systemctl; returns per-unit ok. Never raises."""
    import shutil
    if not shutil.which("systemctl"):
        return {u: False for u in units}
    out = {}
    for u in units:
        try:
            r = subprocess.run(["systemctl", action, u], capture_output=True, text=True, timeout=30)
            out[u] = (r.returncode == 0)
        except Exception:
            out[u] = False
    return out


def _run_accuracy_job():
    """Pause load, run the benchmark, resume load. Runs in a worker thread."""
    import time as _t
    try:
        _accuracy_running["phase"] = "pausing load"
        _svc("stop", ACCURACY_PAUSE_UNITS)
        # also flip the loadgen control file in case the service respawns fast
        try:
            LOADGEN_CONTROL.write_text("off")
        except Exception:
            pass
        _t.sleep(float(os.environ.get("ACCURACY_DRAIN_S", "22")))  # let inflight requests drain
        _accuracy_running["phase"] = "measuring"
        env = dict(os.environ)
        env["ACCURACY_STATE"] = str(ACCURACY_STATE)
        env["ACCURACY_SET"] = str(ACCURACY_SET)
        env.setdefault("ROUTER_URL", ROUTER_URL)
        subprocess.run([os.environ.get("PYTHON", "python3"), str(ACCURACY_SCRIPT)],
                       env=env, capture_output=True, text=True,
                       timeout=int(os.environ.get("ACCURACY_TIMEOUT_S", "600")))
    except Exception as e:
        try:
            ACCURACY_STATE.write_text(json.dumps({"ok": False, "updated": _t.time(),
                                                  "error": f"{e.__class__.__name__}: {e}"}, indent=2))
        except Exception:
            pass
    finally:
        _accuracy_running["phase"] = "resuming load"
        try:
            LOADGEN_CONTROL.write_text("on")
        except Exception:
            pass
        _svc("start", ACCURACY_PAUSE_UNITS)
        _accuracy_running["on"] = False
        _accuracy_running["phase"] = "idle"


@app.route("/api/accuracy")
def accuracy():
    """Latest cached accuracy results (Option 1 answers + Option 3 routing)."""
    running = _accuracy_running["on"]
    try:
        data = json.loads(ACCURACY_STATE.read_text())
        data["age_s"] = round(time.time() - data.get("updated", 0), 1)
    except FileNotFoundError:
        data = {"ok": False, "error": "no run yet", "answers": {}, "routing": {}}
    except Exception as e:
        data = {"ok": False, "error": f"{e.__class__.__name__}"}
    data["running"] = running
    data["phase"] = _accuracy_running["phase"]
    return jsonify(data)


@app.route("/api/accuracy/run", methods=["POST", "GET"])
def accuracy_run():
    """Kick off an on-demand accuracy run (pauses load, measures, resumes)."""
    with _accuracy_lock:
        if _accuracy_running["on"]:
            return jsonify({"ok": False, "error": "already running",
                            "phase": _accuracy_running["phase"]}), 200
        if not ACCURACY_SCRIPT.exists():
            return jsonify({"ok": False, "error": f"runner missing: {ACCURACY_SCRIPT}"}), 200
        _accuracy_running["on"] = True
        _accuracy_running["started"] = time.time()
        _accuracy_running["phase"] = "starting"
        threading.Thread(target=_run_accuracy_job, daemon=True).start()
    return jsonify({"ok": True, "started": True,
                    "note": "pausing swarm+loadgen, measuring ~1-3 min, then resuming"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "7000")), threaded=True)
