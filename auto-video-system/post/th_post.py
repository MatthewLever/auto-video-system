#!/usr/bin/env python3
"""
th_post.py - post to the YoungSponge Threads account.

Stdlib-only Python 3, zero dependencies. Runs as-is on the Mac and on the Pi.
Mirrors yt_upload.py / tt_post.py / fb_post.py: it is the PRIMITIVE. No time
gate, no queue logic - scheduling policy lives in dispatch.py, in one place.

Reads ~/publishing/th_token.json (written by th_auth.py).

    python3 th_post.py --show-me
    python3 th_post.py --text "hello" --dry-run
    python3 th_post.py --text "hello" --yes
    python3 th_post.py --image-url https://cdn.shopify.com/... --text "caption"
    python3 th_post.py --video-url https://cdn.shopify.com/... --text "caption"
    python3 th_post.py --status <container_id>
    python3 th_post.py --quota
    python3 th_post.py --delete <media_id>

============================================================================
THREADS DOES NOT SCHEDULE. VERIFIED 2026-08-08 - DO NOT RE-LITIGATE THIS.
============================================================================
Checked three ways against the live docs:
  - create-container params: no scheduled_publish_time, no published=false,
    no video_state enum. Only creation_id goes to /threads_publish.
  - the publishing reference lists no scheduling parameter at all.
  - the changelog has never carried a scheduling entry.
The Threads *consumer app* got in-app scheduling in Dec 2024. That is a
first-party UI feature and is NOT exposed through the API.

So Threads follows the TIKTOK pattern, not the YouTube/Facebook one:

  YouTube / Facebook : upload early, the platform holds it. Pi may sleep.
  TikTok / THREADS   : the Pi must be awake AT the slot and post then.

--schedule is deliberately REFUSED below rather than silently ignored.

...and one constraint TikTok did not have: a media container EXPIRES IF NOT
PUBLISHED WITHIN 24 HOURS. So you cannot even pre-build containers the night
before and publish them at the slot. Container creation AND publish both have
to happen at post time. dispatch.py must plan for that.

MEDIA MUST BE A PUBLIC URL
--------------------------
There is no file-upload edge on Threads. image_url / video_url must be
publicly reachable by Meta's fetchers. Use the same Shopify pattern already
used for Instagram:
    stagedUploadsCreate -> POST the file to the returned URL -> fileCreate
    -> use the CDN URL here.
Passing a local path is caught early with that reminder rather than a
confusing 400 from Meta.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(HERE, "th_token.json")

GRAPH = "https://graph.threads.net"
API_VERSION = "v1.0"
TIMEOUT = 120

# Meta's documented publishing quotas (GET /{user}/threads_publishing_limit).
QUOTA_POSTS_24H = 250

# Container status polling. Meta: "query once per minute, for no more than
# 5 minutes". Text containers are ready instantly; video needs the wait.
POLL_INTERVAL = 10
POLL_MAX_WAIT = 300

MAX_TEXT = 500


def _et_tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/New_York")
    except Exception:
        import datetime as _dt
        sys.stderr.write("WARNING: zoneinfo unavailable, using fixed -04:00.\n")
        return _dt.timezone(_dt.timedelta(hours=-4))


def fmt_ts(ts):
    import datetime as _dt
    if not ts:
        return "-"
    return _dt.datetime.fromtimestamp(ts, _et_tz()).strftime("%Y-%m-%d %H:%M %Z")


# --- token -----------------------------------------------------------------
def load_token():
    if not os.path.exists(TOKEN_PATH):
        raise SystemExit(
            "No %s.\nRun:  cd %s && python3 th_auth.py --set-token" % (TOKEN_PATH, HERE))
    with open(TOKEN_PATH) as fh:
        data = json.load(fh)
    if not data.get("access_token") or not data.get("user_id"):
        raise SystemExit("%s is missing access_token / user_id. Re-run th_auth.py."
                         % TOKEN_PATH)
    exp = int(data.get("expires_at") or 0)
    now = int(time.time())
    if exp and exp < now:
        raise SystemExit(
            "The Threads token expired %s and CANNOT be refreshed.\n"
            "Generate a new one in the App Dashboard, then: python3 th_auth.py --set-token"
            % fmt_ts(exp))
    if exp and exp - now < 7 * 24 * 3600:
        sys.stderr.write(
            "WARNING: Threads token expires %s (%d days). Run th_auth.py --refresh.\n"
            % (fmt_ts(exp), (exp - now) // 86400))
    return data


# --- http ------------------------------------------------------------------
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
        hint = "\nHINT: token invalid/expired. python3 th_auth.py --show"
    elif code == 200:
        hint = ("\nHINT: missing permission. threads_content_publish must be attached\n"
                "      to the use case AND the account must be an accepted Threads\n"
                "      Tester while the app is unpublished.")
    elif code == 100:
        hint = ("\nHINT: check the media URL is publicly reachable - Meta fetches it\n"
                "      server-side, so anything behind the Shopify storefront password\n"
                "      will fail here.")
    raise SystemExit("Threads API error on %s\n  %s\n  code=%s subcode=%s%s"
                     % (where, msg, code, sub, hint))


def api_get(path, params):
    url = "%s/%s/%s?%s" % (GRAPH, API_VERSION, path.lstrip("/"),
                           urllib.parse.urlencode(params))
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _raise_threads(e, "GET " + path)


def api_post(path, fields):
    url = "%s/%s/%s" % (GRAPH, API_VERSION, path.lstrip("/"))
    req = urllib.request.Request(
        url, data=urllib.parse.urlencode(fields).encode("utf-8"), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _raise_threads(e, "POST " + path)


def api_delete(path, params):
    url = "%s/%s/%s?%s" % (GRAPH, API_VERSION, path.lstrip("/"),
                           urllib.parse.urlencode(params))
    req = urllib.request.Request(url, method="DELETE")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _raise_threads(e, "DELETE " + path)


# --- publish ---------------------------------------------------------------
def wait_ready(container_id, token, quiet=False):
    """Poll a container until FINISHED. Returns on success, exits on failure."""
    deadline = time.time() + POLL_MAX_WAIT
    last = None
    while time.time() < deadline:
        r = api_get(container_id, {"fields": "status,error_message",
                                   "access_token": token})
        st = r.get("status")
        if st != last and not quiet:
            print("  container %s: %s" % (container_id, st))
            last = st
        if st == "FINISHED":
            return
        if st == "PUBLISHED":
            return
        if st in ("ERROR", "EXPIRED"):
            raise SystemExit("Container %s: %s\n  %s"
                             % (container_id, st, r.get("error_message") or ""))
        time.sleep(POLL_INTERVAL)
    raise SystemExit(
        "Container %s was still %s after %d s. Not publishing.\n"
        "Check it later with: python3 th_post.py --status %s"
        % (container_id, last, POLL_MAX_WAIT, container_id))


def publish(tok, args):
    user_id, token = tok["user_id"], tok["access_token"]

    if args.schedule or args.at:
        raise SystemExit(
            "Threads has NO native scheduling - the API has no scheduled_publish_time\n"
            "and no published=false. Verified against the live docs 2026-08-08.\n"
            "\n"
            "Scheduling has to be done by holding the post until the slot, which is\n"
            "dispatch.py's job (same as TikTok, unlike YouTube/Facebook).\n"
            "Note a container also expires 24h after creation, so the container\n"
            "cannot be pre-built either - create AND publish at the slot.")

    for local in (args.image_url, args.video_url):
        if local and not local.lower().startswith(("http://", "https://")):
            raise SystemExit(
                "Threads has no file-upload edge - media must be a PUBLIC URL, not\n"
                "a local path (%s).\n"
                "Host it first, the same way Instagram does:\n"
                "  Shopify stagedUploadsCreate -> POST the file -> fileCreate -> CDN URL"
                % local)

    if args.image_url and args.video_url:
        raise SystemExit("Pick one of --image-url / --video-url.")

    text = args.text or ""
    if not text and not args.image_url and not args.video_url:
        raise SystemExit("Nothing to post. Give --text, --image-url or --video-url.")
    if len(text) > MAX_TEXT:
        raise SystemExit("Text is %d characters; Threads caps at %d."
                         % (len(text), MAX_TEXT))

    if args.image_url:
        media_type = "IMAGE"
    elif args.video_url:
        media_type = "VIDEO"
    else:
        media_type = "TEXT"

    fields = {"media_type": media_type, "access_token": token}
    if text:
        fields["text"] = text
    if args.image_url:
        fields["image_url"] = args.image_url
    if args.video_url:
        fields["video_url"] = args.video_url
    if args.link:
        if media_type != "TEXT":
            raise SystemExit("--link only applies to a text-only post.")
        fields["link_attachment"] = args.link
    if args.reply_control:
        fields["reply_control"] = args.reply_control
    if args.alt_text:
        fields["alt_text"] = args.alt_text

    # --- the draft ---------------------------------------------------------
    print("=" * 68)
    print("DRAFT - nothing has been sent yet")
    print("=" * 68)
    print("Account   : @%s (%s)" % (tok.get("username", "?"), user_id))
    print("Endpoint  : /%s/threads  ->  /%s/threads_publish" % (user_id, user_id))
    print("Type      : %s" % media_type)
    if args.image_url:
        print("Image URL : %s" % args.image_url)
    if args.video_url:
        print("Video URL : %s" % args.video_url)
    if args.link:
        print("Link      : %s" % args.link)
    if text:
        print("Text      : (%d/%d chars)" % (len(text), MAX_TEXT))
        for line in text.splitlines() or [""]:
            print("    %s" % line)
    print("Schedule  : NOT POSSIBLE on Threads - this publishes IMMEDIATELY")
    print("=" * 68)

    if args.dry_run:
        print("--dry-run: stopping here. No container created.")
        return

    if not args.yes:
        if input("Send it? type 'yes' to confirm: ").strip().lower() != "yes":
            raise SystemExit("Cancelled. Nothing sent.")

    # --- step 1: container --------------------------------------------------
    c = api_post("%s/threads" % user_id, fields)
    container_id = c.get("id")
    if not container_id:
        raise SystemExit("No container id returned: %s" % json.dumps(c))
    print("container : %s  (expires in 24h if not published)" % container_id)

    # Text containers are ready immediately; media needs processing.
    if media_type != "TEXT":
        wait_ready(container_id, token)

    # --- step 2: publish ----------------------------------------------------
    p = api_post("%s/threads_publish" % user_id,
                 {"creation_id": container_id, "access_token": token})
    media_id = p.get("id")
    print("OK: %s" % json.dumps(p))
    if media_id:
        info = api_get(media_id, {"fields": "id,permalink,timestamp",
                                  "access_token": token})
        print("Posted    : %s" % info.get("permalink", "(no permalink returned)"))
        print("Media ID  : %s" % media_id)
        print("Delete it : python3 th_post.py --delete %s" % media_id)


# --- actions ---------------------------------------------------------------
def show_me(tok):
    me = api_get("me", {
        "fields": "id,username,threads_profile_picture_url,threads_biography",
        "access_token": tok["access_token"]})
    print("Account   : @%s" % me.get("username"))
    print("User ID   : %s" % me.get("id"))
    print("Bio       : %s" % (me.get("threads_biography") or "-"))
    print("Token exp : %s" % fmt_ts(tok.get("expires_at")))


def quota(tok):
    r = api_get("%s/threads_publishing_limit" % tok["user_id"], {
        "fields": "quota_usage,config,reply_quota_usage,reply_config",
        "access_token": tok["access_token"]})
    rows = r.get("data") or [{}]
    d = rows[0]
    cfg = d.get("config") or {}
    print("Posts used   : %s / %s in the last %s h"
          % (d.get("quota_usage", "?"), cfg.get("quota_total", QUOTA_POSTS_24H),
             int(cfg.get("quota_duration", 86400)) // 3600))
    rcfg = d.get("reply_config") or {}
    if rcfg:
        print("Replies used : %s / %s"
              % (d.get("reply_quota_usage", "?"), rcfg.get("quota_total", "?")))


def status(tok, obj_id):
    """Works for BOTH a container id and a published media id.

    They are different node types with different fields: `status` and
    `error_message` exist ONLY on an unpublished container. Asking for them on
    a published post fails with code=100 "Tried accessing nonexisting field",
    which reads like a permissions problem and is not. Try the container shape
    first, fall back to the media shape.
    """
    url = "%s/%s/%s?%s" % (GRAPH, API_VERSION, obj_id, urllib.parse.urlencode({
        "fields": "id,status,error_message", "access_token": tok["access_token"]}))
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=TIMEOUT) as r:
            print("container:")
            print(json.dumps(json.loads(r.read().decode("utf-8")), indent=2))
            return
    except urllib.error.HTTPError:
        pass  # not a container - treat it as published media
    r = api_get(obj_id, {
        "fields": "id,permalink,timestamp,text,media_type,is_quote_post",
        "access_token": tok["access_token"]})
    print("published post:")
    print(json.dumps(r, indent=2))


def delete(tok, media_id, assume_yes):
    print("About to DELETE Threads post %s" % media_id)
    print("NOTE: this needs the threads_delete permission. As of 2026-08-08 the")
    print("      portal would not attach it to the use case (it silently reverts),")
    print("      so expect code=200 until that sticks. Delete by hand in the")
    print("      Threads app if this fails - it is not worth fighting the portal.")
    if not assume_yes:
        if input("Type 'delete' to confirm: ").strip().lower() != "delete":
            raise SystemExit("Cancelled. Nothing deleted.")
    r = api_delete(media_id, {"access_token": tok["access_token"]})
    print("OK: %s" % json.dumps(r))


def main():
    ap = argparse.ArgumentParser(description="Post to the YoungSponge Threads account.")
    ap.add_argument("--text", "-m")
    ap.add_argument("--image-url", dest="image_url",
                    help="PUBLIC image URL (Threads has no file upload)")
    ap.add_argument("--video-url", dest="video_url",
                    help="PUBLIC video URL (Threads has no file upload)")
    ap.add_argument("--link", help="link_attachment, text-only posts")
    ap.add_argument("--alt-text", dest="alt_text")
    ap.add_argument("--reply-control", dest="reply_control",
                    choices=["everyone", "accounts_you_follow", "mentioned_only"])
    ap.add_argument("--schedule", help="NOT SUPPORTED - Threads has no native scheduling")
    ap.add_argument("--at", type=int, help="NOT SUPPORTED - see --schedule")
    ap.add_argument("--yes", action="store_true", help="skip confirmation (cron)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-me", action="store_true", dest="show_me")
    ap.add_argument("--quota", action="store_true")
    ap.add_argument("--status", metavar="ID")
    ap.add_argument("--delete", metavar="MEDIA_ID")
    args = ap.parse_args()

    tok = load_token()
    if args.show_me:
        return show_me(tok)
    if args.quota:
        return quota(tok)
    if args.status:
        return status(tok, args.status)
    if args.delete:
        return delete(tok, args.delete, args.yes)
    publish(tok, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
