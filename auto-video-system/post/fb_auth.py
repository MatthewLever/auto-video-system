#!/usr/bin/env python3
"""
fb_auth.py - one-time Facebook Page token bootstrap for YoungSponge.

Mirrors yt_auth.py / tt_auth.py: stdlib-only Python 3, zero dependencies.

What it does
------------
1. Asks for your Meta app secret (hidden input) and a short-lived USER token
   copied out of the Graph API Explorer.
2. Exchanges the short-lived user token for a LONG-LIVED user token (~60 days).
3. Uses that to fetch the PAGE access token for the YoungSponge Page.
   A Page token derived from a long-lived user token does not expire.
4. Writes ~/publishing/fb_client.json and ~/publishing/fb_token.json, mode 600.

Nothing is printed that would reveal a secret or a token.

Usage
-----
    cd ~/publishing && python3 fb_auth.py

Re-run it any time the Page token stops working. It is idempotent.
"""

import getpass
import json
import os
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# --- constants -------------------------------------------------------------
# NOTE: os.path.abspath does not resolve symlinks, so if this file is symlinked
# into a sandbox dir, HERE becomes that dir and the sandbox gets its own creds.
# Same trick as the TikTok sandbox. Deliberate.
HERE = os.path.dirname(os.path.abspath(__file__))

CLIENT_PATH = os.path.join(HERE, "fb_client.json")
TOKEN_PATH = os.path.join(HERE, "fb_token.json")

GRAPH = "https://graph.facebook.com"
API_VERSION = "v26.0"

APP_ID = "YOUR_META_APP_ID"          # Meta app "YoungSponge Publisher"
PAGE_ID = "YOUR_PAGE_ID"          # FB Page "YoungSponge" 
# NB: OTHER_OBJECT_ID is a DIFFERENT object that also shows up under
# "Pages you manage". It is NOT the Page and every Graph call against it
# fails with code 100 / subcode 33. Do not "fix" the ID above to that one.

TIMEOUT = 30


# --- tiny http helper ------------------------------------------------------
def _get(path, params):
    url = "%s/%s/%s?%s" % (GRAPH, API_VERSION, path.lstrip("/"),
                           urllib.parse.urlencode(params))
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(body)
            err = parsed.get("error", {})
            raise SystemExit(
                "Graph API error on %s\n  message: %s\n  type: %s  code: %s  subcode: %s"
                % (path, err.get("message"), err.get("type"),
                   err.get("code"), err.get("error_subcode"))
            )
        except ValueError:
            raise SystemExit("HTTP %s on %s\n%s" % (e.code, path, body[:500]))


def _get_root(path, params):
    """Some endpoints (debug_token) live off the version-less root."""
    url = "%s/%s?%s" % (GRAPH, path.lstrip("/"), urllib.parse.urlencode(params))
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": {"message": e.read().decode("utf-8", "replace")[:300]}}


def _write_600(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)   # 0600 before it has a real name
    os.replace(tmp, path)                        # atomic
    return path


def _fmt_expiry(ts):
    if not ts:
        return "never (0 = does not expire)"
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(ts))


# --- main ------------------------------------------------------------------
def main():
    print(__doc__.split("Usage")[0].strip())
    print()
    print("App ID : %s" % APP_ID)
    print("Page ID: %s" % PAGE_ID)
    print()
    print("Step 1 of 2 - your Meta app secret.")
    print("  developers.facebook.com -> YoungSponge Publisher")
    print("  -> App settings -> Basic -> App secret -> Show")
    print("  (Facebook will ask for your password. That is why this step is manual.)")
    print()

    app_secret = getpass.getpass("  Paste app secret (input hidden): ").strip()
    if not app_secret:
        raise SystemExit("No app secret entered. Nothing written.")
    if len(app_secret) < 20:
        print("  ! That looks short for an app secret (%d chars). Continuing anyway."
              % len(app_secret))

    print()
    print("Step 2 of 2 - a short-lived USER access token.")
    print("  developers.facebook.com/tools/explorer -> app 'YoungSponge Publisher'")
    print("  -> the Access Token box at the top right -> copy button next to it")
    print("  It must carry pages_manage_posts. It expires ~2h after you mint it,")
    print("  so do this bit promptly.")
    print()

    short_token = getpass.getpass("  Paste short-lived user token (input hidden): ").strip()
    if not short_token:
        raise SystemExit("No token entered. Nothing written.")

    # --- exchange for a long-lived user token ------------------------------
    print()
    print("Exchanging for a long-lived user token ...")
    ll = _get("oauth/access_token", {
        "grant_type": "fb_exchange_token",
        "client_id": APP_ID,
        "client_secret": app_secret,
        "fb_exchange_token": short_token,
    })
    long_user_token = ll.get("access_token")
    if not long_user_token:
        raise SystemExit("No access_token in exchange response: %s" % ll)
    print("  ok - long-lived user token acquired (expires_in=%s)"
          % ll.get("expires_in", "?"))

    # --- fetch the Page token ---------------------------------------------
    # /me/accounts is NOT used on purpose: this Page is owned by a Business
    # portfolio, so it never appears there from a personal-login token.
    # Addressing the Page node directly is what works. Verified 2026-08-08.
    print("Fetching the Page access token ...")
    page = _get(PAGE_ID, {
        "fields": "access_token,name,fan_count",
        "access_token": long_user_token,
    })
    page_token = page.get("access_token")
    if not page_token:
        raise SystemExit("No access_token on the Page node. Response: %s" % page)
    print("  ok - Page: %s (%s followers)"
          % (page.get("name"), page.get("fan_count")))

    # --- inspect what we actually got --------------------------------------
    dbg = _get_root("debug_token", {
        "input_token": page_token,
        "access_token": long_user_token,
    }).get("data", {})

    scopes = dbg.get("scopes", [])
    expires_at = dbg.get("expires_at", 0)
    data_expires_at = dbg.get("data_access_expires_at", 0)

    _write_600(CLIENT_PATH, {"app_id": APP_ID, "app_secret": app_secret})
    _write_600(TOKEN_PATH, {
        "app_id": APP_ID,
        "page_id": PAGE_ID,
        "page_name": page.get("name"),
        "page_access_token": page_token,
        "user_access_token_long_lived": long_user_token,
        "token_type": dbg.get("type"),
        "scopes": scopes,
        "expires_at": expires_at,
        "data_access_expires_at": data_expires_at,
        "obtained_at": int(time.time()),
        "obtained_by": "fb_auth.py",
    })

    print()
    print("Wrote (mode 600):")
    print("  %s" % CLIENT_PATH)
    print("  %s" % TOKEN_PATH)
    print()
    print("Token type   : %s" % dbg.get("type"))
    print("Scopes       : %s" % ", ".join(scopes) if scopes else "(none reported)")
    print("Page token   : %s" % _fmt_expiry(expires_at))
    print("Data access  : %s" % _fmt_expiry(data_expires_at))
    print()
    if expires_at:
        print("!! The Page token reports a real expiry above rather than 'never'.")
        print("   That means the exchange did not fully take. Put the date on a")
        print("   calendar and re-run this script before then.")
    else:
        print("Page token does not expire. It can still be invalidated by a password")
        print("change, revoking the app, or removing the Page from the portfolio.")
    print()
    print("DATA ACCESS is separate from token expiry: Meta cuts off data access")
    print("~90 days after the last user authorisation regardless. Re-running this")
    print("script resets that clock. Diarise the earlier of the two dates.")
    print()
    print("Next: python3 fb_post.py --show-page")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled. Nothing written.")
