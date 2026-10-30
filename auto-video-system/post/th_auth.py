#!/usr/bin/env python3
"""
th_auth.py - install and refresh the YoungSponge Threads access token.

Stdlib-only Python 3, zero dependencies. Runs as-is on the Mac and on the Pi.
Mirrors yt_auth.py / tt_auth.py / fb_auth.py in shape, but the flow is
DIFFERENT and simpler - read this before assuming it works like the others.

WHY THERE IS NO OAUTH SERVER HERE
---------------------------------
Threads REJECTS loopback redirect URIs. Verified against the live portal
2026-08-08 - all three of these bounce the form with "Form can't be saved":

    http://127.0.0.1:8723/callback/
    https://127.0.0.1:8723/callback/
    https://localhost:8723/callback/

So the tt_auth.py pattern (spin up a local HTTP server, catch the redirect)
CANNOT be ported. Do not spend an afternoon trying; it is a server-side
validation rule, not a formatting problem.

Instead we use the App Dashboard's **User Token Generator**, which mints a
LONG-LIVED (60-day) token directly, with no redirect URI and no app secret:

    developers.facebook.com
      -> YoungSponge Publisher
      -> Use cases -> Access the Threads API -> Settings
      -> User Token Generator -> Generate access token

That requires the Threads account to be a *tester* on the app and to have
accepted the invite (Threads app -> Settings -> Account -> Website
permissions -> Invites), and the account must be PUBLIC.

CONSEQUENCE WORTH KNOWING: because there is no OAuth code exchange, the
Threads APP SECRET is never needed - not here, not in th_post.py, not on the
Pi. There is no th_client.json. Threads is the only channel in this stack
where the secret never leaves the dashboard.

Usage
-----
    python3 th_auth.py --set-token          # paste the generated token
    python3 th_auth.py --show               # who am I, when does it expire
    python3 th_auth.py --refresh            # extend by another 60 days
    python3 th_auth.py --refresh --if-due   # refresh only if <21 days left (cron)

THE REFRESH IS NOT OPTIONAL
---------------------------
Facebook's Page token never expires. This one does, in 60 days, and a LAPSED
token cannot be refreshed - it needs a human back in the dashboard. Meta also
refuses to refresh a token that is less than 24 hours old.

Put this on the Pi's cron next to dispatch.py:

    17 4 * * *  cd $HOME/publishing && /usr/bin/python3 th_auth.py --refresh --if-due >> th_auth.log 2>&1

Daily, idempotent, silent when there is nothing to do.
"""

import argparse
import json
import os
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(HERE, "th_token.json")

GRAPH = "https://graph.threads.net"
API_VERSION = "v1.0"
TIMEOUT = 60

# Meta: long-lived Threads tokens last 60 days and may be refreshed once they
# are at least 24h old. Refresh well before the edge - a lapse is unrecoverable
# without a human.
REFRESH_MIN_AGE = 24 * 3600
REFRESH_DUE_WITHIN = 21 * 24 * 3600

THREADS_APP_ID = "YOUR_THREADS_APP_ID"  # NOT YOUR_META_APP_ID (that is the Facebook one)


def _et_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/New_York")
    except Exception:
        import datetime as _dt
        return _dt.timezone(_dt.timedelta(hours=-4))


def fmt_ts(ts):
    import datetime as _dt
    if not ts:
        return "-"
    return _dt.datetime.fromtimestamp(ts, _et_tz()).strftime("%Y-%m-%d %H:%M %Z")


def _raise_threads(e, where):
    body = e.read().decode("utf-8", "replace")
    try:
        err = json.loads(body).get("error", {})
    except ValueError:
        raise SystemExit("HTTP %s on %s\n%s" % (e.code, where, body[:600]))
    msg = err.get("message", "")
    code, sub = err.get("code"), err.get("error_subcode")
    hint = ""
    if code == 190:
        hint = ("\nHINT: token invalid or expired. If it lapsed past 60 days it cannot\n"
                "      be refreshed - generate a fresh one in the App Dashboard\n"
                "      (Use cases -> Access the Threads API -> Settings ->\n"
                "      User Token Generator) and run: python3 th_auth.py --set-token")
    elif code == 100:
        hint = ("\nHINT: check you used the THREADS app id (%s), not the Facebook\n"
                "      app id YOUR_META_APP_ID." % THREADS_APP_ID)
    raise SystemExit("Threads API error on %s\n  %s\n  code=%s subcode=%s%s"
                     % (where, msg, code, sub, hint))


def api_get(path, params, versioned=True):
    base = "%s/%s/%s" % (GRAPH, API_VERSION, path.lstrip("/")) if versioned \
        else "%s/%s" % (GRAPH, path.lstrip("/"))
    url = "%s?%s" % (base, urllib.parse.urlencode(params))
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _raise_threads(e, "GET " + path)


def save_token(data):
    tmp = TOKEN_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2)
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # 600 before it is visible
    os.replace(tmp, TOKEN_PATH)                 # atomic
    print("Wrote %s (mode 600)" % TOKEN_PATH)


def load_token(required=True):
    if not os.path.exists(TOKEN_PATH):
        if not required:
            return None
        raise SystemExit(
            "No %s.\nGenerate a long-lived token in the App Dashboard, then run:\n"
            "    cd %s && python3 th_auth.py --set-token" % (TOKEN_PATH, HERE))
    with open(TOKEN_PATH) as fh:
        return json.load(fh)


def whoami(access_token):
    return api_get("me", {
        "fields": "id,username,threads_profile_picture_url,threads_biography",
        "access_token": access_token})


def set_token(args):
    tok = args.token
    if not tok:
        sys.stderr.write(
            "Paste the long-lived token from the App Dashboard\n"
            "(Use cases -> Access the Threads API -> Settings -> User Token Generator).\n"
            "It will not be echoed to the terminal history by this script.\n\n")
        try:
            import getpass
            tok = getpass.getpass("Threads access token: ")
        except Exception:
            tok = input("Threads access token: ")
    tok = tok.strip()
    # The TikTok client key arrived with a trailing space once and cost time.
    # Strip aggressively and sanity-check before writing anything.
    if not tok:
        raise SystemExit("Empty token, nothing written.")
    if tok.startswith("Bearer "):
        tok = tok[len("Bearer "):].strip()
    if len(tok) < 40:
        raise SystemExit(
            "That is only %d characters - too short for a Threads token.\n"
            "Did the copy get truncated?" % len(tok))

    me = whoami(tok)
    now = int(time.time())
    data = {
        "access_token": tok,
        "user_id": me.get("id"),
        "username": me.get("username"),
        "threads_app_id": THREADS_APP_ID,
        "obtained_at": now,
        # The generator issues long-lived tokens; assume the documented 60 days.
        # --refresh overwrites this with the value Meta actually returns.
        "expires_at": now + 60 * 24 * 3600,
        "expires_at_is_assumed": True,
        "scopes": ["threads_basic", "threads_content_publish"],
        "source": "app dashboard User Token Generator",
    }
    save_token(data)
    print("Account   : @%s (%s)" % (data["username"], data["user_id"]))
    print("Expires   : %s  (assumed 60d - confirmed on first --refresh)"
          % fmt_ts(data["expires_at"]))


def refresh(args):
    data = load_token()
    now = int(time.time())
    age = now - int(data.get("obtained_at") or 0)
    left = int(data.get("expires_at") or 0) - now

    if left <= 0:
        raise SystemExit(
            "Token expired %s. It CANNOT be refreshed.\n"
            "Generate a new one in the App Dashboard, then: python3 th_auth.py --set-token"
            % fmt_ts(data.get("expires_at")))
    if age < REFRESH_MIN_AGE:
        msg = ("Token is only %d hours old; Meta refuses to refresh below 24h."
               % (age // 3600))
        if args.if_due:
            print(msg + " Nothing to do.")
            return
        raise SystemExit(msg)
    if args.if_due and left > REFRESH_DUE_WITHIN:
        print("Token still has %d days left (threshold %d). Nothing to do."
              % (left // 86400, REFRESH_DUE_WITHIN // 86400))
        return

    r = api_get("refresh_access_token", {
        "grant_type": "th_refresh_token",
        "access_token": data["access_token"]}, versioned=False)

    new_tok = r.get("access_token")
    if not new_tok:
        raise SystemExit("Refresh returned no access_token: %s" % json.dumps(r))
    expires_in = int(r.get("expires_in") or 60 * 24 * 3600)

    data["access_token"] = new_tok
    data["obtained_at"] = now
    data["expires_at"] = now + expires_in
    data["expires_at_is_assumed"] = False
    data["last_refresh"] = now
    save_token(data)
    print("Refreshed. Valid until %s (%d days)."
          % (fmt_ts(data["expires_at"]), expires_in // 86400))


def show(args):
    data = load_token()
    now = int(time.time())
    left = int(data.get("expires_at") or 0) - now
    me = whoami(data["access_token"])
    print("Account   : @%s" % me.get("username"))
    print("User ID   : %s" % me.get("id"))
    print("App ID    : %s (Threads)" % data.get("threads_app_id", THREADS_APP_ID))
    print("Obtained  : %s" % fmt_ts(data.get("obtained_at")))
    print("Expires   : %s%s" % (fmt_ts(data.get("expires_at")),
                                "  [assumed]" if data.get("expires_at_is_assumed") else ""))
    print("Days left : %d" % (left // 86400))
    print("Scopes    : %s" % ", ".join(data.get("scopes") or []))
    if left < REFRESH_DUE_WITHIN:
        print("\n>> Refresh is DUE. Run: python3 th_auth.py --refresh")


def main():
    ap = argparse.ArgumentParser(
        description="Install / inspect / refresh the Threads access token.")
    ap.add_argument("--set-token", action="store_true", dest="set_token")
    ap.add_argument("--token", help="pass the token inline instead of being prompted "
                                    "(avoid: it lands in shell history)")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--if-due", action="store_true", dest="if_due",
                    help="with --refresh: no-op unless expiry is near (for cron)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.set_token:
        return set_token(args)
    if args.refresh:
        return refresh(args)
    if args.show:
        return show(args)
    ap.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
