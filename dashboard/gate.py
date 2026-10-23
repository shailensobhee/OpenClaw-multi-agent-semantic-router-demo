#!/usr/bin/env python3
"""Password gate + reverse proxy for the ai-Pulse 2026 OpenClaw booth.

Fronts the dashboard (127.0.0.1:7000) on public :80. A single shared password
(GATE_PASSWORD) unlocks the whole site via a signed session cookie. Everything
behind the gate (/, /live, /playground, /api/*, SSE streams) is transparently
reverse-proxied, including streaming responses.

Brute-force protection: BAN_THRESHOLD wrong passwords from one client IP and
that IP is banned. A banned IP gets a bare 404 for EVERYTHING (login page,
assets, API) so the site looks like it does not exist. Bans + failure counts
persist to GATE_STATE_FILE so a restart does not reset them.

Behind Cloudflare, the real client IP is in CF-Connecting-IP (request.remote_addr
is only the CF edge), so bans key on that.

No external deps beyond Flask + requests (already in the dashboard venv).
"""
import os
import hmac
import hashlib
import time
import json
import threading
from flask import Flask, request, Response, redirect, make_response, stream_with_context, abort
import requests

app = Flask(__name__)

UPSTREAM = os.environ.get("GATE_UPSTREAM", "http://127.0.0.1:7000")
PASSWORD = os.environ.get("GATE_PASSWORD", "aipulse2026").strip()
SECRET = os.environ.get("GATE_SECRET", "ai-pulse-2026-openclaw-booth-secret").encode()
COOKIE = "booth_sess2"
OLD_COOKIES = ["booth_auth"]  # stale cookies from earlier gate versions, must be purged
MAXAGE = 60 * 60 * 24 * 7  # 7 days
BAN_THRESHOLD = int(os.environ.get("GATE_BAN_THRESHOLD", "3"))
STATE_FILE = os.environ.get("GATE_STATE_FILE", "/opt/openclaw-swarm/gate_bans.json")

# Hop-by-hop headers must not be forwarded.
HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-encoding",
    "content-length",
}

# ---- brute-force state (persisted) -----------------------------------------
_lock = threading.Lock()
_fails = {}       # ip -> consecutive wrong-password count
_banned = set()   # ips that hit the threshold


def _load_state():
    global _fails, _banned
    try:
        with open(STATE_FILE) as f:
            d = json.load(f)
        _fails = {k: int(v) for k, v in d.get("fails", {}).items()}
        _banned = set(d.get("banned", []))
    except (FileNotFoundError, ValueError, OSError):
        _fails, _banned = {}, set()


def _save_state():
    tmp = STATE_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"fails": _fails, "banned": sorted(_banned)}, f)
        os.replace(tmp, STATE_FILE)
    except OSError:
        pass


def _client_ip() -> str:
    # Behind Cloudflare: CF-Connecting-IP is the true client. Fall back through
    # X-Forwarded-For (first hop) then the socket peer.
    ip = request.headers.get("CF-Connecting-IP")
    if ip:
        return ip.strip()
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _is_banned(ip: str) -> bool:
    with _lock:
        return ip in _banned


def _record_fail(ip: str) -> bool:
    """Increment failures for ip; return True if this tips it into a ban."""
    with _lock:
        n = _fails.get(ip, 0) + 1
        _fails[ip] = n
        newly_banned = False
        if n >= BAN_THRESHOLD:
            if ip not in _banned:
                newly_banned = True
            _banned.add(ip)
        _save_state()
        return newly_banned


def _clear_fails(ip: str):
    with _lock:
        if ip in _fails:
            del _fails[ip]
            _save_state()


def _attempts_left(ip: str) -> int:
    with _lock:
        return max(0, BAN_THRESHOLD - _fails.get(ip, 0))


# ---- auth token ------------------------------------------------------------
def _sign(ts: str) -> str:
    return hmac.new(SECRET, ts.encode(), hashlib.sha256).hexdigest()


def _make_token() -> str:
    ts = str(int(time.time()))
    return f"{ts}.{_sign(ts)}"


def _valid(token: str) -> bool:
    try:
        ts, sig = token.split(".", 1)
    except (ValueError, AttributeError):
        return False
    if not hmac.compare_digest(sig, _sign(ts)):
        return False
    return (int(time.time()) - int(ts)) < MAXAGE


def _authed() -> bool:
    return _valid(request.cookies.get(COOKIE, ""))


def _dbg(event: str, **kw):
    # Debug tracing disabled in production (it logged raw cookies). Left as a
    # no-op so the call sites stay harmless; flip GATE_DEBUG=1 to re-enable.
    if os.environ.get("GATE_DEBUG") != "1":
        return
    try:
        import datetime
        line = {
            "t": datetime.datetime.utcnow().isoformat(),
            "event": event,
            "path": request.full_path,
            "ip": _client_ip(),
            "has_cookie": COOKIE in request.cookies,
            "cookie_valid": _valid(request.cookies.get(COOKIE, "")),
            "dup_count": request.headers.get("Cookie", "").count(COOKIE + "="),
            "ua": request.headers.get("User-Agent", "")[:80],
            "cf": request.headers.get("CF-Connecting-IP", ""),
            "xfp": request.headers.get("X-Forwarded-Proto", ""),
            **kw,
        }
        with open("/var/log/booth-gate-debug.log", "a") as f:
            f.write(json.dumps(line) + "\n")
    except Exception:
        pass


LOGIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenClaw demo</title>
<style>
  :root{--amd-red:#ED1C24;}
  *{box-sizing:border-box;}
  body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
    font-family:"Inter",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
    background:#05070a;color:#e8eef5;
    background-image:radial-gradient(900px 500px at 70% -10%,rgba(237,28,36,.18),transparent),
      radial-gradient(700px 500px at 0% 110%,rgba(46,120,220,.12),transparent);}
  .card{width:min(92vw,400px);background:rgba(13,18,24,.92);border:1px solid #1f2730;
    border-radius:16px;padding:34px 30px;box-shadow:0 20px 60px rgba(0,0,0,.55);}
  .brand{display:flex;align-items:center;gap:10px;margin-bottom:6px;}
  .brand .dot{width:11px;height:11px;border-radius:3px;background:var(--amd-red);}
  .brand b{font-size:13px;letter-spacing:.4px;color:#c6d2de;font-weight:700;}
  h1{font-size:20px;margin:14px 0 4px;line-height:1.25;}
  p.sub{margin:0 0 22px;color:#8a97a6;font-size:13px;line-height:1.5;}
  label{display:block;font-size:12px;color:#8a97a6;margin:0 0 7px;}
  input[type=password]{width:100%;padding:12px 13px;border-radius:9px;border:1px solid #273039;
    background:#0b1014;color:#e8eef5;font-size:15px;outline:none;}
  input[type=password]:focus{border-color:var(--amd-red);}
  button{margin-top:16px;width:100%;padding:12px;border:0;border-radius:9px;cursor:pointer;
    background:var(--amd-red);color:#fff;font-size:15px;font-weight:700;letter-spacing:.3px;}
  button:hover{background:#c91820;}
  .err{margin-top:14px;color:#ff6b6b;font-size:13px;min-height:16px;}
  .foot{margin-top:20px;color:#5c6773;font-size:11px;text-align:center;line-height:1.5;}
</style></head><body>
  <form class="card" method="POST" action="/__gate_login" autocomplete="off">
    <div class="brand"><span class="dot"></span><b>AMD INSTINCT&trade; MI300X</b></div>
    <h1>OpenClaw multi-agent demo</h1>
    <p class="sub">A live OpenClaw agent swarm routed across a Qwen model fleet on one MI300X. Enter the demo password to continue.</p>
    <label for="p">Password</label>
    <input id="p" name="password" type="password" autofocus autocomplete="off" placeholder="&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;&bull;">
    <input type="hidden" name="next" value="__NEXT__">
    <button type="submit">Enter demo</button>
    <div class="err">__ERR__</div>
    <div class="foot">AMD Instinct&trade; MI300X</div>
  </form>
</body></html>"""


def _no_store(resp: Response) -> Response:
    # Make sure neither Cloudflare nor the browser caches the gate, so a stale
    # login form can never be replayed against new server logic. Vary: Cookie so
    # any intermediate cache keys on auth state (an authed request must never be
    # served a cached no-cookie login page -> that is the post-login bounce).
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    resp.headers["CDN-Cache-Control"] = "no-store"
    resp.headers["Vary"] = "Cookie"
    return resp


def _login_page(err="", nxt="/"):
    html = LOGIN_HTML.replace("__ERR__", err).replace("__NEXT__", nxt)
    return _no_store(Response(html, status=200, mimetype="text/html"))


@app.before_request
def _block_banned():
    # A banned IP sees a bare 404 for everything: the site "does not exist".
    if _is_banned(_client_ip()):
        abort(404)


@app.route("/__gate_login", methods=["POST"])
def gate_login():
    ip = _client_ip()
    pw = request.form.get("password", "").strip()
    nxt = request.form.get("next", "/") or "/"
    if not nxt.startswith("/"):
        nxt = "/"
    if hmac.compare_digest(pw, PASSWORD):
        _clear_fails(ip)
        # Redirect to a one-shot cache-busted URL so a stale `/` sitting in the
        # browser's OWN disk cache (from the pre-no-store gate) cannot be served
        # instead of the authed dashboard -> that is the post-login bounce.
        sep = "&" if "?" in nxt else "?"
        target = f"{nxt}{sep}_a={int(time.time())}"
        resp = make_response(redirect(target))
        # Secure only if the client reached us over HTTPS (CF sets X-Forwarded-Proto).
        # A Secure cookie on a plain-HTTP hop would be silently dropped by the browser.
        secure = request.headers.get("X-Forwarded-Proto", "").lower() == "https"
        resp.set_cookie(COOKIE, _make_token(), max_age=MAXAGE, httponly=True,
                        samesite="Lax", secure=secure, path="/")
        # Purge stale cookies from earlier gate versions that Chrome keeps sending
        # and that shadow the fresh one (signed with an old secret -> never valid).
        for oc in OLD_COOKIES:
            resp.delete_cookie(oc, path="/")
            resp.delete_cookie(oc)
        _dbg("login_ok", secure=secure, target=target)
        return _no_store(resp)
    # wrong password
    _dbg("login_bad")
    newly_banned = _record_fail(ip)
    if newly_banned:
        abort(404)
    left = _attempts_left(ip)
    msg = "Incorrect password. Try again."
    if left <= 2:
        msg = f"Incorrect password. {left} attempt{'s' if left != 1 else ''} left before this IP is blocked."
    return _login_page(err=msg, nxt=nxt)


@app.route("/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def proxy(path):
    if not _authed():
        _dbg("proxy_unauthed")
        nxt = request.full_path if request.query_string else request.path
        if nxt.endswith("?"):
            nxt = nxt[:-1]
        return _login_page(nxt=nxt)

    url = f"{UPSTREAM}/{path}"
    fwd_headers = {k: v for k, v in request.headers if k.lower() not in ("host", "cookie")}
    try:
        upstream = requests.request(
            method=request.method,
            url=url,
            params=request.args,
            data=request.get_data(),
            headers=fwd_headers,
            cookies=request.cookies,
            stream=True,
            timeout=(10, 600),
            allow_redirects=False,
        )
    except requests.exceptions.RequestException as e:
        return Response(f"Upstream error: {e}", status=502, mimetype="text/plain")

    resp_headers = [(k, v) for k, v in upstream.raw.headers.items() if k.lower() not in HOP]
    # Prevent the browser/CDN from caching authed HTML pages and replaying them
    # (a cached `/` served without re-checking the cookie causes a post-login bounce).
    ctype = upstream.headers.get("Content-Type", "")
    if "text/html" in ctype.lower():
        resp_headers = [(k, v) for (k, v) in resp_headers
                        if k.lower() not in ("cache-control", "pragma", "expires", "vary", "cdn-cache-control")]
        resp_headers += [
            ("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0, private"),
            ("Pragma", "no-cache"),
            ("Expires", "0"),
            ("CDN-Cache-Control", "no-store"),
            ("Vary", "Cookie"),
        ]

    def generate():
        for chunk in upstream.iter_content(chunk_size=8192):
            if chunk:
                yield chunk

    return Response(stream_with_context(generate()), status=upstream.status_code, headers=resp_headers)


_load_state()

if __name__ == "__main__":
    port = int(os.environ.get("GATE_PORT", "80"))
    app.run(host="0.0.0.0", port=port, threaded=True)
