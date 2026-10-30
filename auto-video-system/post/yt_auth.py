#!/usr/bin/env python3
"""
yt_auth.py — one-time YouTube OAuth: turn client_secret.json into a refresh token.

Standard library only. No pip install. Requires Python 3.8+.

Usage:
    python3 yt_auth.py
    python3 yt_auth.py --client-secret ~/Downloads/youtube/client_secret_xxx.json

Writes yt_token.json (chmod 600) next to this script. That file holds the
refresh token — treat it like a password. Do not commit it, do not paste it.

If the OAuth consent screen is published ("In production" at
console.cloud.google.com/auth/audience), the refresh token does not expire.
If it is still in "Testing", it dies after 7 days. Publish it first.
"""

import argparse
import base64
import glob
import hashlib
import http.server
import json
import os
import secrets
import socket
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
DEFAULT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(HERE, "yt_token.json")

SUCCESS_HTML = b"""<!doctype html><meta charset="utf-8">
<title>Authorized</title>
<body style="font:16px/1.5 -apple-system,system-ui,sans-serif;padding:3rem;max-width:32rem">
<h2>YouTube upload authorized.</h2>
<p>The refresh token was written to <code>yt_token.json</code>.</p>
<p>You can close this tab and go back to the terminal.</p>
</body>"""

FAILURE_HTML = b"""<!doctype html><meta charset="utf-8">
<title>Failed</title>
<body style="font:16px/1.5 -apple-system,system-ui,sans-serif;padding:3rem;max-width:32rem">
<h2>Authorization failed.</h2>
<p>Check the terminal for details, then run <code>yt_auth.py</code> again.</p>
</body>"""


class _Catcher(http.server.BaseHTTPRequestHandler):
    """Single-shot loopback listener for the OAuth redirect."""

    result = None

    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/":
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


def find_client_secret(explicit):
    if explicit:
        path = os.path.expanduser(explicit)
        if not os.path.exists(path):
            sys.exit(f"No such file: {path}")
        return path
    matches = sorted(glob.glob(os.path.join(HERE, "client_secret*.json")))
    if not matches:
        sys.exit(
            "No client_secret*.json found next to this script.\n"
            "Download it from console.cloud.google.com/auth/clients (your Desktop\n"
            "app client -> download icon) and drop it in:\n  " + HERE
        )
    if len(matches) > 1:
        print(f"Multiple client_secret files found; using {os.path.basename(matches[0])}")
    return matches[0]


def load_client(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    node = raw.get("installed") or raw.get("web")
    if not node:
        sys.exit(
            f"{os.path.basename(path)} has neither an 'installed' nor a 'web' key.\n"
            "Make sure you created the client with application type 'Desktop app'."
        )
    if "web" in raw:
        print(
            "Warning: this looks like a 'Web application' client. A 'Desktop app'\n"
            "client is what this flow expects. Continuing, but if the redirect is\n"
            "rejected, create a Desktop app client instead."
        )
    cid, csecret = node.get("client_id"), node.get("client_secret")
    if not cid or not csecret:
        sys.exit("client_secret JSON is missing client_id or client_secret.")
    return cid, csecret


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post_form(url, fields):
    body = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        sys.exit(f"Token exchange failed ({exc.code}):\n{detail}")


def main():
    ap = argparse.ArgumentParser(description="Mint a YouTube refresh token.")
    ap.add_argument("--client-secret", help="path to client_secret*.json")
    ap.add_argument(
        "--scopes",
        default=" ".join(DEFAULT_SCOPES),
        help="space-separated OAuth scopes (default: youtube.upload only)",
    )
    ap.add_argument("--port", type=int, default=0, help="loopback port (default: random free port)")
    args = ap.parse_args()

    secret_path = find_client_secret(args.client_secret)
    client_id, client_secret = load_client(secret_path)
    print(f"Using client: {os.path.basename(secret_path)}")

    port = args.port or free_port()
    redirect_uri = f"http://127.0.0.1:{port}"

    verifier = base64.urlsafe_b64encode(os.urandom(64)).decode().rstrip("=")
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    )
    state = secrets.token_urlsafe(24)

    auth_url = AUTH_ENDPOINT + "?" + urllib.parse.urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": args.scopes,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }
    )

    server = http.server.HTTPServer(("127.0.0.1", port), _Catcher)
    server.timeout = 300
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    print("\nOpening your browser to authorize. If it does not open, paste this URL:\n")
    print(auth_url + "\n")
    print("Pick the Google account that owns the YouTube channel.")
    print('If you see "Google hasn\'t verified this app": Advanced -> Go to <app> (unsafe).')
    print("That warning is expected for a personal-use app and is safe here.\n")
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    thread.join(timeout=310)
    server.server_close()

    result = _Catcher.result
    if not result:
        sys.exit("Timed out waiting for the browser redirect. Run it again.")
    if "error" in result:
        sys.exit(f"Authorization denied: {result['error']}")
    if result.get("state") != state:
        sys.exit("State mismatch — aborting. Run it again in a clean browser profile.")

    tokens = post_form(
        TOKEN_ENDPOINT,
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": result["code"],
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        },
    )

    refresh = tokens.get("refresh_token")
    if not refresh:
        sys.exit(
            "Google did not return a refresh_token. This usually means the account\n"
            "already granted this client. Revoke it at\n"
            "https://myaccount.google.com/permissions and run this again."
        )

    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh,
        "token_uri": TOKEN_ENDPOINT,
        "scopes": args.scopes.split(),
    }
    fd = os.open(TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Wrote {TOKEN_PATH} (chmod 600).")
    print("\nNext: python3 yt_upload.py --file <video.mp4> --title \"...\"")
    print(
        "\nReminder: if the consent screen is still 'Testing', this refresh token\n"
        "dies in 7 days. Publish the app at console.cloud.google.com/auth/audience."
    )


if __name__ == "__main__":
    main()
