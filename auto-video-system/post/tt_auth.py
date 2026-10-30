#!/usr/bin/env python3
"""
tt_auth.py — one-time TikTok OAuth: turn a client key/secret into a refresh token.

Standard library only. No pip install. Requires Python 3.8+.

Usage:
    python3 tt_auth.py
    python3 tt_auth.py --port 8722

Reads tt_client.json (next to this script):

    { "client_key": "aw...", "client_secret": "..." }

Both values come from developers.tiktok.com -> your app -> Manage app ->
Basic information. Create that file yourself; nothing here writes it.

Writes tt_token.json (chmod 600) next to this script. That file holds the
refresh token — treat it like a password. Do not commit it, do not paste it.

BEFORE THIS WILL WORK, in the developer portal under Login Kit, register the
redirect URI exactly as printed by this script (default
http://127.0.0.1:8722/callback/). TikTok only allows localhost / 127.0.0.1
hosts for desktop apps, the port is mandatory, and the string must match
character for character including the trailing slash.

Token lifetimes, which differ from YouTube in a way that matters:
  - access_token  ~24 hours   (tt_post.py refreshes it automatically)
  - refresh_token ~365 days   (it does NOT last forever, and TikTok rotates it
                               on every refresh — tt_post.py rewrites this file
                               each time. Do not hand-copy a stale token around.)
"""

import argparse
import hashlib
import http.server
import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

AUTH_ENDPOINT = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_ENDPOINT = "https://open.tiktokapis.com/v2/oauth/token/"

# Comma-separated, not space-separated — TikTok differs from Google here.
DEFAULT_SCOPES = "user.info.basic,video.publish,video.upload"

DEFAULT_PORT = 8722
CALLBACK_PATH = "/callback/"

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT_PATH = os.path.join(HERE, "tt_client.json")
TOKEN_PATH = os.path.join(HERE, "tt_token.json")

SUCCESS_HTML = b"""<!doctype html><meta charset="utf-8">
<title>Authorized</title>
<body style="font:16px/1.5 -apple-system,system-ui,sans-serif;padding:3rem;max-width:32rem">
<h2>TikTok posting authorized.</h2>
<p>The refresh token was written to <code>tt_token.json</code>.</p>
<p>You can close this tab and go back to the terminal.</p>
</body>"""

FAILURE_HTML = b"""<!doctype html><meta charset="utf-8">
<title>Failed</title>
<body style="font:16px/1.5 -apple-system,system-ui,sans-serif;padding:3rem;max-width:32rem">
<h2>Authorization failed.</h2>
<p>Check the terminal for details, then run <code>tt_auth.py</code> again.</p>
</body>"""


class _Catcher(http.server.BaseHTTPRequestHandler):
    """Single-shot loopback listener for the OAuth redirect."""

    result = None

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.rstrip("/") != CALLBACK_PATH.rstrip("/"):
            self.send_response(404)
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        _Catcher.result = {k: v[0] for k, v in params.items()}
        ok = "code" in _Catcher.result
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(SUCCESS_HTML if ok else FAILURE_HTML)

    def log_message(self, *_args):
        pass  # keep the terminal clean


def die(msg):
    sys.exit(f"error: {msg}")


def load_client():
    if not os.path.exists(CLIENT_PATH):
        die(
            f"{CLIENT_PATH} not found.\n"
            "Create it with your app's credentials from\n"
            "  developers.tiktok.com -> your app -> Manage app -> Basic information\n\n"
            '  { "client_key": "aw...", "client_secret": "..." }\n\n'
            "Then: chmod 600 tt_client.json"
        )
    with open(CLIENT_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    key, secret = raw.get("client_key"), raw.get("client_secret")
    if not key or not secret:
        die("tt_client.json must contain both 'client_key' and 'client_secret'.")
    return key, secret


def port_is_free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def pkce_pair():
    """
    TikTok deviates from RFC 7636: their own desktop sample hex-encodes the
    SHA256 digest instead of base64url-encoding it. A base64url challenge is
    rejected. Do not "fix" this to match the RFC.
    """
    verifier = secrets.token_urlsafe(64)  # 86 chars of [A-Za-z0-9_-], within 43-128
    challenge = hashlib.sha256(verifier.encode("ascii")).hexdigest()
    return verifier, challenge


def post_form(url, fields):
    body = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Cache-Control": "no-cache",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        die(f"Token exchange failed (HTTP {exc.code}):\n{detail}")
    # TikTok returns HTTP 200 with an error body on some failures.
    if payload.get("error"):
        die(
            "Token exchange rejected: "
            f"{payload.get('error')} — {payload.get('error_description', '')}\n"
            f"log_id: {payload.get('log_id', 'n/a')}"
        )
    return payload


def write_token(payload, mode=0o600):
    fd = os.open(TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Mint a TikTok refresh token.")
    ap.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"loopback port for the redirect (default: {DEFAULT_PORT}). "
        "Must match the redirect URI registered in the developer portal.",
    )
    ap.add_argument(
        "--scopes",
        default=DEFAULT_SCOPES,
        help=f"comma-separated scopes (default: {DEFAULT_SCOPES})",
    )
    args = ap.parse_args()

    client_key, client_secret = load_client()
    redirect_uri = f"http://127.0.0.1:{args.port}{CALLBACK_PATH}"

    if not port_is_free(args.port):
        die(
            f"Port {args.port} is already in use. Free it, or pass --port with a\n"
            "different value — but then register that exact redirect URI in the\n"
            "developer portal too."
        )

    print(f"client_key   {client_key}")
    print(f"redirect_uri {redirect_uri}")
    print(f"scopes       {args.scopes}\n")
    print("If authorization fails with a redirect_uri error, the portal entry does")
    print("not match the line above exactly — including the trailing slash.\n")

    verifier, challenge = pkce_pair()
    state = secrets.token_urlsafe(24)

    auth_url = AUTH_ENDPOINT + "?" + urllib.parse.urlencode(
        {
            "client_key": client_key,
            "response_type": "code",
            "scope": args.scopes,
            "redirect_uri": redirect_uri,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )

    server = http.server.HTTPServer(("127.0.0.1", args.port), _Catcher)
    server.timeout = 300
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    print("Opening your browser to authorize. If it does not open, paste this URL:\n")
    print(auth_url + "\n")
    print("Log in as the TikTok account you want to post to, and approve the scopes.\n")
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    thread.join(timeout=310)
    server.server_close()

    result = _Catcher.result
    if not result:
        die("Timed out waiting for the browser redirect. Run it again.")
    if "error" in result:
        die(
            f"Authorization denied: {result['error']} "
            f"{result.get('error_description', '')}"
        )
    if result.get("state") != state:
        die("State mismatch — aborting. Run it again in a clean browser profile.")

    tokens = post_form(
        TOKEN_ENDPOINT,
        {
            "client_key": client_key,
            "client_secret": client_secret,
            "code": urllib.parse.unquote(result["code"]),
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
            "code_verifier": verifier,
        },
    )

    refresh = tokens.get("refresh_token")
    if not refresh:
        die(
            "TikTok did not return a refresh_token. Check that the app has the\n"
            "requested scopes approved, then run this again."
        )

    now = int(time.time())
    payload = {
        "client_key": client_key,
        "client_secret": client_secret,
        "refresh_token": refresh,
        "access_token": tokens.get("access_token"),
        "access_expires_at": now + int(tokens.get("expires_in", 0)),
        "refresh_expires_at": now + int(tokens.get("refresh_expires_in", 0)),
        "open_id": tokens.get("open_id"),
        "scope": tokens.get("scope", args.scopes),
        "token_uri": TOKEN_ENDPOINT,
    }
    write_token(payload)

    print(f"\nWrote {TOKEN_PATH} (chmod 600).")
    print(f"  open_id        {payload['open_id']}")
    print(f"  scopes granted {payload['scope']}")
    print(
        "  refresh valid  until "
        + time.strftime("%Y-%m-%d", time.localtime(payload["refresh_expires_at"]))
    )
    print("\nNext: python3 tt_post.py --file <video.mp4> --title \"...\"")
    print(
        "\nReminder: unlike the YouTube token, this one expires — about a year out.\n"
        "TikTok also issues a NEW refresh token on every refresh; tt_post.py\n"
        "rewrites tt_token.json each time, so always copy the current file rather\n"
        "than an old backup."
    )


if __name__ == "__main__":
    main()
