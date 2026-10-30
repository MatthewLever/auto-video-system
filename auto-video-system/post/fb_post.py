#!/usr/bin/env python3
"""
fb_post.py - post or schedule to the YoungSponge Facebook Page.

Stdlib-only Python 3, zero dependencies. Runs as-is on the Mac and on the Pi.
Mirrors yt_upload.py / tt_post.py: it is the PRIMITIVE. It has no time gate and
no queue logic - scheduling policy lives in dispatch.py, in one place.

Reads ~/publishing/fb_token.json (written by fb_auth.py).

Examples
--------
    python3 fb_post.py --show-page
    python3 fb_post.py --message "hello" --dry-run
    python3 fb_post.py --message "hello" --schedule "2026-08-16 09:00"
    python3 fb_post.py --reel clip.mp4 --message "caption" --schedule "2026-08-16 09:00"
    python3 fb_post.py --reel clip.mp4 --message "caption" --ai-generated --yes
    python3 fb_post.py --video clip.mp4 --message "caption" --schedule "2026-08-16 09:00"
    python3 fb_post.py --photo shoe.jpg --message "caption" --yes
    python3 fb_post.py --list-scheduled
    python3 fb_post.py --status YOUR_PAGE_ID_POST_ID

--reel vs --video
-----------------
--reel  -> /video_reels, posts as a REEL. Supports scheduling via
           video_state=SCHEDULED. This is what you want for YoungSponge.
--video -> /videos, posts as a plain feed video. Kept for long-form only.

Safety
------
Nothing is sent until you confirm, unless you pass --yes (for cron).
This script never deletes anything. Cancel scheduled posts in Meta Business
Suite -> Planner.
"""

import argparse
import json
import mimetypes
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(HERE, "fb_token.json")

GRAPH = "https://graph.facebook.com"
API_VERSION = "v26.0"
TIMEOUT = 300

# Meta's documented scheduling windows, verified against the live docs 2026-08-08.
MIN_LEAD = 10 * 60                 # 10 minutes, both edges
MAX_LEAD_FEED = 75 * 24 * 3600     # 75 days  (/feed and /photos)
MAX_LEAD_VIDEO = 180 * 24 * 3600   # ~6 months (/videos)
MAX_LEAD_REEL = 75 * 24 * 3600     # /video_reels cap is NOT documented; 75d is a
                                   # conservative guess. If Meta rejects a far-out
                                   # reel, this is the first knob to loosen.

# REELS DO SUPPORT SCHEDULING. The reels-publishing *guide* only ever shows
# video_state=PUBLISHED, which is what makes people think otherwise. The Graph
# API *reference* for /{page-id}/video_reels lists
#     video_state  enum {DRAFT, PUBLISHED, SCHEDULED}
# plus scheduled_publish_time. Trust the reference, not the guide.

REELS_UPLOAD_HOST = "https://rupload.facebook.com/video-upload"


# --- time ------------------------------------------------------------------
def _et_tz():
    """America/New_York, with DST handled properly.

    dispatch.py hardcodes -04:00 and needs a manual fix after Nov 1.
    Don't repeat that here - zoneinfo is stdlib on 3.9+.
    """
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/New_York")
    except Exception:
        import datetime as _dt
        sys.stderr.write(
            "WARNING: zoneinfo unavailable, falling back to a FIXED -04:00 offset.\n"
            "         This is wrong from Nov 1 until Mar. Check the printed time.\n")
        return _dt.timezone(_dt.timedelta(hours=-4))


def parse_when(text):
    """'YYYY-MM-DD HH:MM' in Eastern time -> unix seconds."""
    import datetime as _dt
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            naive = _dt.datetime.strptime(text, fmt)
            break
        except ValueError:
            continue
    else:
        raise SystemExit("Could not parse --schedule %r. Use 'YYYY-MM-DD HH:MM' (Eastern)."
                         % text)
    return int(naive.replace(tzinfo=_et_tz()).timestamp())


def fmt_ts(ts):
    import datetime as _dt
    if not ts:
        return "-"
    return _dt.datetime.fromtimestamp(ts, _et_tz()).strftime("%Y-%m-%d %H:%M %Z")


# --- token -----------------------------------------------------------------
def load_token():
    if not os.path.exists(TOKEN_PATH):
        raise SystemExit(
            "No %s.\nRun:  cd %s && python3 fb_auth.py" % (TOKEN_PATH, HERE))
    with open(TOKEN_PATH) as fh:
        data = json.load(fh)
    if not data.get("page_access_token") or not data.get("page_id"):
        raise SystemExit("%s is missing page_access_token / page_id. Re-run fb_auth.py."
                         % TOKEN_PATH)
    exp = data.get("expires_at") or 0
    if exp and exp < time.time():
        raise SystemExit(
            "The stored Page token expired on %s. Re-run fb_auth.py." % fmt_ts(exp))
    dexp = data.get("data_access_expires_at") or 0
    if dexp and dexp < time.time():
        sys.stderr.write(
            "WARNING: data access expired %s. Calls will likely fail; re-run fb_auth.py.\n"
            % fmt_ts(dexp))
    return data


# --- http ------------------------------------------------------------------
def _raise_graph(e, where):
    body = e.read().decode("utf-8", "replace")
    try:
        err = json.loads(body).get("error", {})
    except ValueError:
        raise SystemExit("HTTP %s on %s\n%s" % (e.code, where, body[:600]))
    msg = err.get("message", "")
    code, sub = err.get("code"), err.get("error_subcode")
    hint = ""
    if code == 100 and sub == 33:
        hint = ("\nHINT: that object is not visible to this app. If you changed the\n"
                "      Page ID, note that OTHER_OBJECT_ID is NOT the Page - the real\n"
                "      one is YOUR_PAGE_ID.")
    elif code == 190:
        hint = "\nHINT: token invalid or expired. Re-run fb_auth.py."
    elif code == 200:
        hint = ("\nHINT: missing permission. The token needs pages_manage_posts, and\n"
                "      you must have a role on the app while it is in Development Mode.")
    raise SystemExit("Graph API error on %s\n  %s\n  code=%s subcode=%s%s"
                     % (where, msg, code, sub, hint))


def graph_get(path, params):
    url = "%s/%s/%s?%s" % (GRAPH, API_VERSION, path.lstrip("/"),
                           urllib.parse.urlencode(params))
    try:
        with urllib.request.urlopen(urllib.request.Request(url), timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _raise_graph(e, "GET " + path)


def graph_post(path, fields, file_field=None, file_path=None):
    """POST urlencoded, or multipart when a file is attached."""
    url = "%s/%s/%s" % (GRAPH, API_VERSION, path.lstrip("/"))
    if file_path:
        boundary = "----fbpost%s" % uuid.uuid4().hex
        ctype = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        pre = []
        for k, v in fields.items():
            pre.append("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                       % (boundary, k, v))
        head = ("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                "Content-Type: %s\r\n\r\n"
                % (boundary, file_field, os.path.basename(file_path), ctype))
        tail = "\r\n--%s--\r\n" % boundary
        with open(file_path, "rb") as fh:
            payload = ("".join(pre) + head).encode("utf-8") + fh.read() + tail.encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "multipart/form-data; boundary=%s" % boundary)
    else:
        req = urllib.request.Request(
            url, data=urllib.parse.urlencode(fields).encode("utf-8"), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _raise_graph(e, "POST " + path)


def post_reel(page_id, token, path, description=None, title=None,
              when=None, ai_generated=False):
    """Three-phase Reels upload: START -> rupload -> FINISH.

    Returns the FINISH response dict.
    """
    size = os.path.getsize(path)

    # --- phase 1: START ----------------------------------------------------
    start = graph_post("%s/video_reels" % page_id,
                       {"upload_phase": "START", "access_token": token})
    video_id = start.get("video_id")
    upload_url = start.get("upload_url") or "%s/%s" % (REELS_UPLOAD_HOST, video_id)
    if not video_id:
        raise SystemExit("Reels START gave no video_id: %s" % start)
    print("  START ok - video_id %s" % video_id)

    # --- phase 2: binary upload to rupload.facebook.com --------------------
    # NOT a Graph call: raw bytes, OAuth header, offset/file_size headers.
    with open(path, "rb") as fh:
        blob = fh.read()
    req = urllib.request.Request(upload_url, data=blob, method="POST")
    req.add_header("Authorization", "OAuth %s" % token)
    req.add_header("offset", "0")
    req.add_header("file_size", str(size))
    req.add_header("Content-Type", "application/octet-stream")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            up = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _raise_graph(e, "rupload video_reels")
    if not up.get("success", True):
        raise SystemExit("Reels upload did not report success: %s" % up)
    print("  upload ok - %.1f MB" % (size / 1e6))

    # --- phase 3: FINISH ---------------------------------------------------
    fields = {"upload_phase": "FINISH", "video_id": video_id, "access_token": token}
    if when:
        fields["video_state"] = "SCHEDULED"
        fields["scheduled_publish_time"] = str(when)
    else:
        fields["video_state"] = "PUBLISHED"
    if description:
        fields["description"] = description
    if title:
        fields["title"] = title
    if ai_generated:
        fields["is_ai_generated"] = "true"

    fin = graph_post("%s/video_reels" % page_id, fields)
    fin.setdefault("video_id", video_id)
    return fin


# --- actions ---------------------------------------------------------------
def show_page(tok):
    p = graph_get(tok["page_id"], {
        "fields": "id,name,fan_count,followers_count,link",
        "access_token": tok["page_access_token"]})
    print("Page      : %s" % p.get("name"))
    print("ID        : %s" % p.get("id"))
    print("Followers : %s" % p.get("followers_count", p.get("fan_count")))
    print("Link      : %s" % p.get("link", "-"))
    print("Token exp : %s" % (fmt_ts(tok.get("expires_at")) if tok.get("expires_at")
                              else "never"))
    print("Data exp  : %s" % fmt_ts(tok.get("data_access_expires_at")))
    print("Scopes    : %s" % ", ".join(tok.get("scopes") or []))


def list_scheduled(tok):
    r = graph_get("%s/scheduled_posts" % tok["page_id"], {
        "fields": "id,message,scheduled_publish_time",
        "access_token": tok["page_access_token"]})
    rows = r.get("data", [])
    if not rows:
        print("No scheduled posts.")
        return
    print("%d scheduled post(s):" % len(rows))
    for x in rows:
        msg = (x.get("message") or "").replace("\n", " ")
        print("  %s  %s  %s" % (fmt_ts(x.get("scheduled_publish_time")), x["id"],
                                (msg[:57] + "...") if len(msg) > 60 else msg))


def status(tok, post_id):
    r = graph_get(post_id, {
        "fields": "id,message,created_time,scheduled_publish_time,is_published,permalink_url",
        "access_token": tok["page_access_token"]})
    print(json.dumps(r, indent=2))


def publish(tok, args):
    page_id, token = tok["page_id"], tok["page_access_token"]

    media, edge, file_field = None, "feed", None
    if args.reel:
        media, edge, file_field = args.reel, "video_reels", None
    elif args.video:
        media, edge, file_field = args.video, "videos", "source"
    elif args.photo:
        media, edge, file_field = args.photo, "photos", "source"

    if sum(bool(x) for x in (args.reel, args.video, args.photo)) > 1:
        raise SystemExit("Pick one of --reel / --video / --photo.")
    if media and not os.path.isfile(media):
        raise SystemExit("File not found: %s" % media)
    if not media and not args.message and not args.link:
        raise SystemExit(
            "Nothing to post. Give --message, --link, --photo, --video or --reel.")

    fields = {"access_token": token}
    if args.message:
        fields["description" if edge in ("videos", "video_reels") else "message"] = \
            args.message
    if args.link:
        if edge != "feed":
            raise SystemExit("--link only applies to a plain text post.")
        fields["link"] = args.link
    if args.title and edge in ("videos", "video_reels"):
        fields["title"] = args.title
    if args.ai_generated and edge != "video_reels":
        raise SystemExit("--ai-generated only applies to --reel.")

    when = None
    if args.schedule or args.at:
        when = args.at if args.at else parse_when(args.schedule)
        lead = when - int(time.time())
        cap = {"videos": MAX_LEAD_VIDEO,
               "video_reels": MAX_LEAD_REEL}.get(edge, MAX_LEAD_FEED)
        if lead < MIN_LEAD:
            raise SystemExit(
                "Scheduled time must be at least 10 minutes out. That one is %d min away."
                % (lead // 60))
        if lead > cap:
            raise SystemExit(
                "Scheduled time is %d days out; the /%s edge caps at %d days."
                % (lead // 86400, edge, cap // 86400))
        if edge != "video_reels":
            fields["published"] = "false"
            fields["scheduled_publish_time"] = str(when)
    elif edge != "video_reels":
        fields["published"] = "true"

    # --- the draft ---------------------------------------------------------
    print("=" * 68)
    print("DRAFT - nothing has been sent yet")
    print("=" * 68)
    print("Page      : %s (%s)" % (tok.get("page_name", "?"), page_id))
    print("Endpoint  : /%s/%s" % (page_id, edge))
    if media:
        print("File      : %s (%.1f MB)" % (media, os.path.getsize(media) / 1e6))
    if edge == "video_reels":
        print("Posts as  : REEL")
        if args.ai_generated:
            print("AI flag   : is_ai_generated=true")
    if args.title and edge in ("videos", "video_reels"):
        print("Title     : %s" % args.title)
    if args.message:
        print("Message   :")
        for line in args.message.splitlines() or [""]:
            print("    %s" % line)
    if args.link:
        print("Link      : %s" % args.link)
    if when:
        print("Schedule  : %s  (Facebook holds it; this machine need not be awake)"
              % fmt_ts(when))
    else:
        print("Schedule  : NONE - this publishes IMMEDIATELY to all followers")
    if edge == "videos":
        print("Note      : plain FEED VIDEO, not a Reel. Use --reel for a Reel.")
    print("=" * 68)

    if args.dry_run:
        print("--dry-run: stopping here.")
        return

    if not args.yes:
        if input("Send it? type 'yes' to confirm: ").strip().lower() != "yes":
            raise SystemExit("Cancelled. Nothing sent.")

    if edge == "video_reels":
        r = post_reel(page_id, token, media,
                      description=args.message, title=args.title,
                      when=when, ai_generated=args.ai_generated)
    else:
        r = graph_post("%s/%s" % (page_id, edge), fields,
                       file_field=file_field, file_path=media)
    post_id = r.get("id") or r.get("post_id")
    print("OK: %s" % json.dumps(r))
    if when:
        print("Scheduled for %s" % fmt_ts(when))
        print("Cancel it in Meta Business Suite -> Planner if you change your mind.")
    if post_id:
        print("Check: python3 fb_post.py --status %s" % post_id)


def main():
    ap = argparse.ArgumentParser(description="Post or schedule to the YoungSponge FB Page.")
    ap.add_argument("--message", "-m")
    ap.add_argument("--title", help="video title (videos only)")
    ap.add_argument("--link", help="attach a URL (text posts only)")
    ap.add_argument("--photo")
    ap.add_argument("--video", help="plain feed video (long-form). Use --reel for Reels.")
    ap.add_argument("--reel", help="post as a REEL via /video_reels (schedulable)")
    ap.add_argument("--ai-generated", action="store_true", dest="ai_generated",
                    help="set is_ai_generated on a Reel")
    ap.add_argument("--schedule", help="'YYYY-MM-DD HH:MM' Eastern time")
    ap.add_argument("--at", type=int, help="unix timestamp instead of --schedule")
    ap.add_argument("--yes", action="store_true", help="skip confirmation (cron)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--show-page", action="store_true")
    ap.add_argument("--list-scheduled", action="store_true")
    ap.add_argument("--status", metavar="POST_ID")
    args = ap.parse_args()

    tok = load_token()
    if args.show_page:
        return show_page(tok)
    if args.list_scheduled:
        return list_scheduled(tok)
    if args.status:
        return status(tok, args.status)
    publish(tok, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
