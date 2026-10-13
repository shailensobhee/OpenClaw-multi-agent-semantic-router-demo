#!/usr/bin/env python3
"""model_tap: transparent proxy between OpenClaw and the Semantic Router (:8899).

Purpose: OpenClaw masks the served model as the alias `vllm-sr/auto`. The router
returns the REAL model in the X-Vsr-Selected-Model response header. We cannot
re-probe to learn it (OpenClaw's ~8.7K-token scaffold changes routing). So we
read the agent id from the per-request bearer `local-vllm.<id>`, forward the
request to the router with the canonical key, and record the real selected model
per agent into a sidecar JSON.

HARD RULE: this sits in the live booth hot path. Capture is strictly
best-effort inside try/except. Forwarding ALWAYS happens. Any capture failure
must never affect the proxied response.
"""
import http.server, socketserver, urllib.request, json, os, threading, time
from pathlib import Path

ROUTER = os.environ.get("TAP_ROUTER", "http://127.0.0.1:8899")
CANON  = os.environ.get("TAP_CANON_KEY", "local-vllm")
SIDE   = Path(os.environ.get("TAP_SIDECAR", "/opt/openclaw-swarm/agent-models.json"))
PORT   = int(os.environ.get("TAP_PORT", "8898"))

_lock = threading.Lock()
_models = {}  # agentid -> {model, decision, ts}

def _persist():
    try:
        tmp = SIDE.with_suffix(".tmp")
        with _lock:
            snap = dict(_models)
        tmp.write_text(json.dumps(snap))
        tmp.replace(SIDE)
    except Exception:
        pass

class H(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def _proxy(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
        except Exception:
            n = 0
        body = self.rfile.read(n) if n else b""
        # agent id from bearer: local-vllm.<id>  (best-effort, never fatal)
        agent = None
        try:
            auth = self.headers.get("Authorization", "")
            tok = auth.split("Bearer ", 1)[-1].strip()
            if tok.startswith(CANON + ".") and len(tok) > len(CANON) + 1:
                agent = tok[len(CANON) + 1:]
        except Exception:
            agent = None
        req = urllib.request.Request(ROUTER + self.path, data=body, method=self.command)
        for k, v in self.headers.items():
            lk = k.lower()
            if lk in ("host", "content-length", "authorization", "connection"):
                continue
            req.add_header(k, v)
        req.add_header("Authorization", "Bearer " + CANON)  # canonical key upstream
        req.add_header("Content-Length", str(len(body)))
        try:
            r = urllib.request.urlopen(req, timeout=180)
            data = r.read()
            # best-effort capture: FIRST tagged call per turn = primary generation
            if agent:
                try:
                    sm = r.getheader("X-Vsr-Selected-Model")
                    dec = r.getheader("X-Vsr-Selected-Decision")
                    if sm:
                        with _lock:
                            _models[agent] = {"model": sm, "decision": dec, "ts": time.time()}
                        _persist()
                except Exception:
                    pass
            self.send_response(r.status)
            for k, v in r.getheaders():
                if k.lower() in ("transfer-encoding", "content-length", "connection"):
                    continue
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            msg = json.dumps({"error": {"message": "tap_upstream_error: " + str(e)}}).encode()
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
            except Exception:
                pass
    do_POST = _proxy
    do_GET = _proxy
    def log_message(self, *a):
        pass

class TS(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

if __name__ == "__main__":
    TS(("127.0.0.1", PORT), H).serve_forever()
