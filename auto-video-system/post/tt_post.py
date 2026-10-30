#!/usr/bin/env python3
"""
tt_post.py — post a video to TikTok via the Content Posting API (Direct Post).

Standard library only. No pip install. Requires Python 3.8+.
Needs tt_token.json in this directory — run tt_auth.py once to create it.

Usage:
    python3 tt_post.py --file ~/Downloads/Doritos-edit.mp4 \\
        --title "Dorito shoes #fyp" \\
        --privacy SELF_ONLY

Defaults to privacy SELF_ONLY and prints the full draft for approval before
anything leaves the machine. Pass --yes to skip the prompt (for cron).

TWO THINGS THAT DIFFER FROM yt_upload.py — do not assume parity:

1. THERE IS NO publishAt. TikTok has no scheduling parameter. A Direct Post
   goes live the moment TikTok finishes processing it. "Post at 9am" therefore
   means something has to be awake at 9am to make the call. On YouTube the
   Pi can upload any time before the slot and YouTube publishes it; here the
   Pi is the scheduler, so its uptime is load-bearing. The time gate belongs
   in the dispatcher, not in this script — this script posts now.

2. UNTIL THE APP PASSES TIKTOK'S AUDIT, every post is force-restricted to
   private/self-only. Asking for PUBLIC_TO_EVERYONE before then fails with
   unaudited_client_can_only_post_to_private_accounts. That is expected, not
   a bug in this script.
"""

import argparse
import http.client
import json
import math
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(HERE, "tt_token.json")

API_BASE = "https://open.tiktokapis.com/v2"
CREATOR_INFO = f"{API_BASE}/post/publish/creator_info/query/"
VIDEO_INIT = f"{API_BASE}/post/publish/video/init/"
STATUS_FETCH = f"{API_BASE}/post/publish/status/fetch/"
TOKEN_ENDPOINT = f"{API_BASE}/oauth/token/"

MIN_CHUNK = 5 * 1024 * 1024        # TikTok floor: 5 MB
MAX_CHUNK = 64 * 1024 * 1024       # TikTok ceiling: 64 MB (final chunk may reach 128)
TARGET_CHUNK = 32 * 1024 * 1024    # what we actually aim for; see plan_chunks
MAX_CHUNKS = 1000
TITLE_LIMIT = 2200                 # UTF-16 code units
MAX_RETRIES = 6

PRIVACY_CHOICES = [
    "PUBLIC_TO_EVERYONE",
    "MUTUAL_FOLLOW_FRIENDS",
    "FOLLOWER_OF_CREATOR",
    "SELF_ONLY",
]

FAIL_HINTS = {
    "file_format_check_failed": "TikTok rejected the container/codec. Use H.264 MP4.",
    "duration_check_failed": "Video is longer than this account may post via API.",
    "frame_rate_check_failed": "Frame rate outside TikTok's accepted range.",
    "picture_size_check_failed": "Resolution outside TikTok's accepted range.",
    "spam_risk_too_many_posts": "Daily post cap hit for this account. Wait it out.",
    "spam_risk_user_banned_from_posting": "The account itself is restricted.",
    "spam_risk_text": "Something in the title tripped spam detection.",
    "auth_removed": "The creator revoked the app's access. Re-run tt_auth.py.",
    "publish_cancelled": "Cancelled on TikTok's side.",
    "internal": "TikTok-side error. Retry later; nothing to fix locally.",
}


def die(msg):
    sys.exit(f"error: {msg}")


def utf16_len(text):
    """TikTok counts titles in UTF-16 code units, so emoji cost 2."""
    return len(text.encode("utf-16-le")) // 2


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------

def load_creds():
    if not os.path.exists(TOKEN_PATH):
        die(f"{TOKEN_PATH} not found. Run: python3 tt_auth.py")
    with open(TOKEN_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_creds(creds):
    """Atomic, mode 600. Called on every refresh because TikTok rotates the
    refresh token — losing the new one costs you a re-auth."""
    tmp = TOKEN_PATH + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(creds, fh, indent=2)
    os.replace(tmp, TOKEN_PATH)


def access_token(creds):
    """Return a valid access token, refreshing and persisting if needed."""
    now = int(time.time())
    if creds.get("access_token") and creds.get("access_expires_at", 0) > now + 120:
        return creds["access_token"]

    if creds.get("refresh_expires_at", 0) and creds["refresh_expires_at"] < now:
        die(
            "The refresh token expired (TikTok's last ~365 days).\n"
            "Re-run: python3 tt_auth.py"
        )

    body = urllib.parse.urlencode(
        {
            "client_key": creds["client_key"],
            "client_secret": creds["client_secret"],
            "grant_type": "refresh_token",
            "refresh_token": creds["refresh_token"],
        }
    ).encode()
    req = urllib.request.Request(
        creds.get("token_uri", TOKEN_ENDPOINT),
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
        die(f"Token refresh failed (HTTP {exc.code}):\n{detail}")

    if payload.get("error"):
        die(
            f"Token refresh rejected: {payload.get('error')} — "
            f"{payload.get('error_description', '')}\n"
            "If this says the token is invalid, re-run: python3 tt_auth.py"
        )

    now = int(time.time())
    creds["access_token"] = payload["access_token"]
    creds["access_expires_at"] = now + int(payload.get("expires_in", 0))
    # TikTok issues a NEW refresh token here. Persist it or the next run breaks.
    if payload.get("refresh_token"):
        creds["refresh_token"] = payload["refresh_token"]
        creds["refresh_expires_at"] = now + int(payload.get("refresh_expires_in", 0))
    save_creds(creds)
    return creds["access_token"]


# --------------------------------------------------------------------------
# API plumbing
# --------------------------------------------------------------------------

def api_post(url, token, body=None):
    """POST JSON, return the `data` object. Dies with TikTok's own error text."""
    data = json.dumps(body).encode("utf-8") if body is not None else b""
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except ValueError:
            die(f"{url} failed (HTTP {exc.code}):\n{raw}")

    err = payload.get("error") or {}
    code = err.get("code", "ok")
    if code not in ("ok", "", None):
        explain(code, err)
    return payload.get("data", {})


def explain(code, err):
    msg = err.get("message", "")
    log_id = err.get("log_id", "n/a")
    if code == "unaudited_client_can_only_post_to_private_accounts":
        die(
            "TikTok refused a non-private post because the app has not passed the\n"
            "Content Posting API audit yet.\n\n"
            "This is the expected state right now. Two options:\n"
            "  - re-run with --privacy SELF_ONLY to prove the pipeline works, or\n"
            "  - submit the audit at\n"
            "    https://developers.tiktok.com/application/content-posting-api\n"
            f"\nlog_id: {log_id}"
        )
    if code == "privacy_level_option_mismatch":
        die(
            "The privacy level you asked for is not offered to this creator.\n"
            "Run with --show-creator to see the allowed values.\n"
            f"\nlog_id: {log_id}"
        )
    if code in ("access_token_invalid", "scope_not_authorized"):
        die(
            f"{code}: {msg}\n"
            "Re-run: python3 tt_auth.py  (and confirm video.publish is approved\n"
            "on the app in the developer portal).\n"
            f"\nlog_id: {log_id}"
        )
    if code == "rate_limit_exceeded":
        die(f"Rate limited (init is capped at 6 requests/minute).\nlog_id: {log_id}")
    die(f"TikTok error {code}: {msg}\nlog_id: {log_id}")


# --------------------------------------------------------------------------
# chunk plan
# --------------------------------------------------------------------------

def plan_chunks(size):
    """
    TikTok's rule, which is not the obvious one: total_chunk_count is
    floor(video_size / chunk_size), and the FINAL chunk absorbs the remainder
    rather than getting its own short chunk. A file under the 5 MB floor goes
    up whole, as a single chunk.

    Naively taking chunk_size = 64 MB satisfies the rule but produces an ugly
    tail: a 100 MB file becomes ONE 100 MB PUT, and a 700 MB file ends with a
    124 MB one. Both are legal (the final chunk may run to 128 MB) and both
    are exactly what you don't want on a domestic uplink. So aim for evenly
    sized chunks near TARGET_CHUNK instead, which keeps the remainder small.

    Returns (chunk_size, total_chunk_count).
    """
    if size < MIN_CHUNK:
        return size, 1
    count = max(1, round(size / TARGET_CHUNK))
    count = min(count, MAX_CHUNKS)
    chunk = size // count
    chunk = max(MIN_CHUNK, min(MAX_CHUNK, chunk))
    count = max(1, size // chunk)
    if count > MAX_CHUNKS:
        chunk = math.ceil(size / MAX_CHUNKS)
        if chunk > MAX_CHUNK:
            die(
                f"File is {size / 1_073_741_824:.1f} GB — too large to fit in "
                f"{MAX_CHUNKS} chunks of {MAX_CHUNK // 1_048_576} MB."
            )
        count = size // chunk
    return chunk, max(1, count)


def chunk_ranges(size, chunk, count):
    """Byte ranges to PUT. The last one runs to the end of the file."""
    for i in range(count):
        start = i * chunk
        end = size - 1 if i == count - 1 else start + chunk - 1
        yield start, end


def put_chunk(upload_url, data, start, end, size, mime):
    parsed = urllib.parse.urlparse(upload_url)
    path = parsed.path + ("?" + parsed.query if parsed.query else "")
    conn = http.client.HTTPSConnection(parsed.netloc, timeout=600)
    headers = {
        "Content-Type": mime,
        "Content-Length": str(len(data)),
        "Content-Range": f"bytes {start}-{end}/{size}",
    }
    try:
        conn.request("PUT", path, body=data, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read().decode("utf-8", "replace")
    finally:
        conn.close()


def upload(upload_url, path, size, chunk, count, mime):
    with open(path, "rb") as fh:
        for idx, (start, end) in enumerate(chunk_ranges(size, chunk, count), 1):
            fh.seek(start)
            data = fh.read(end - start + 1)
            attempt = 0
            while True:
                try:
                    status, body = put_chunk(upload_url, data, start, end, size, mime)
                except (http.client.HTTPException, OSError) as exc:
                    status, body = 0, str(exc)

                if status in (200, 201, 206, 308):
                    pct = 100.0 * (end + 1) / size
                    print(
                        f"  chunk {idx}/{count}  {end + 1:,}/{size:,} bytes ({pct:.1f}%)",
                        flush=True,
                    )
                    break
                if status in (0, 429, 500, 502, 503, 504):
                    attempt += 1
                    if attempt > MAX_RETRIES:
                        die(f"Gave up on chunk {idx} after {MAX_RETRIES} retries:\n{body}")
                    wait = min(2 ** attempt, 64)
                    print(
                        f"  transient failure on chunk {idx} (HTTP {status}); "
                        f"retrying in {wait}s",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                if status == 403:
                    die(
                        "Upload URL rejected (403). It is only valid for 1 hour after\n"
                        "init — re-run the command to get a fresh one."
                    )
                die(f"Chunk {idx} failed (HTTP {status}):\n{body}")


# --------------------------------------------------------------------------
# status polling
# --------------------------------------------------------------------------

def poll(publish_id, token, timeout, interval):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        data = api_post(STATUS_FETCH, token, {"publish_id": publish_id})
        status = data.get("status")
        if status != last:
            print(f"  status: {status}")
            last = status
        if status == "PUBLISH_COMPLETE":
            return data
        if status == "SEND_TO_USER_INBOX":
            print("  (landed in the creator's TikTok inbox — finish it in the app)")
            return data
        if status == "FAILED":
            reason = data.get("fail_reason", "unknown")
            hint = FAIL_HINTS.get(reason, "")
            die(f"TikTok reported FAILED: {reason}\n{hint}".rstrip())
        time.sleep(interval)
    print(
        f"\n  Still processing after {timeout}s. Not a failure — TikTok is slow\n"
        f"  sometimes. Check again with:\n"
        f"    python3 tt_post.py --status {publish_id}"
    )
    return None


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Post a video to TikTok (Direct Post).")
    ap.add_argument("--file", help="path to the video file (H.264 MP4)")
    ap.add_argument("--title", default="", help=f"caption, max {TITLE_LIMIT} UTF-16 units")
    ap.add_argument(
        "--privacy",
        default="SELF_ONLY",
        choices=PRIVACY_CHOICES,
        help="default: SELF_ONLY (the only level an unaudited app may use)",
    )
    ap.add_argument("--disable-comment", action="store_true")
    ap.add_argument("--disable-duet", action="store_true")
    ap.add_argument("--disable-stitch", action="store_true")
    ap.add_argument(
        "--cover-ms", type=int, default=0, help="frame to use as the cover, in ms"
    )
    ap.add_argument(
        "--aigc", action="store_true", help="declare AI-generated/altered content"
    )
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--dry-run", action="store_true", help="validate and print; send nothing")
    ap.add_argument("--show-creator", action="store_true", help="print creator info and exit")
    ap.add_argument("--status", metavar="PUBLISH_ID", help="check a previous post and exit")
    ap.add_argument("--timeout", type=int, default=300, help="seconds to poll (default 300)")
    ap.add_argument("--interval", type=int, default=5, help="poll interval (default 5s)")
    a = ap.parse_args()

    creds = load_creds()  # fail before showing a draft if auth was never set up

    if a.status:
        token = access_token(creds)
        data = api_post(STATUS_FETCH, token, {"publish_id": a.status})
        print(json.dumps(data, indent=2))
        return

    token = access_token(creds)
    info = api_post(CREATOR_INFO, token)

    if a.show_creator:
        print(json.dumps(info, indent=2))
        return

    if not a.file:
        die("--file is required (or use --show-creator / --status).")

    path = os.path.expanduser(a.file)
    if not os.path.isfile(path):
        die(f"No such file: {path}")
    size = os.path.getsize(path)
    if size == 0:
        die("File is empty.")
    if utf16_len(a.title) > TITLE_LIMIT:
        die(f"Title is {utf16_len(a.title)} UTF-16 units; TikTok's limit is {TITLE_LIMIT}.")

    allowed = info.get("privacy_level_options") or []
    if allowed and a.privacy not in allowed:
        die(
            f"--privacy {a.privacy} is not offered to this creator.\n"
            f"Allowed: {', '.join(allowed)}"
        )

    # TikTok requires these to be disabled if the creator has them off account-wide.
    disable_comment = a.disable_comment or bool(info.get("comment_disabled"))
    disable_duet = a.disable_duet or bool(info.get("duet_disabled"))
    disable_stitch = a.disable_stitch or bool(info.get("stitch_disabled"))

    chunk, count = plan_chunks(size)
    mime = "video/mp4" if path.lower().endswith(".mp4") else "video/*"

    print("\n--- draft ---")
    print(f"account   @{info.get('creator_username', '?')} ({info.get('creator_nickname', '')})")
    print(f"file      {path}  ({size / 1_048_576:.1f} MB, {mime})")
    print(f"title     {a.title or '(none)'}")
    print(f"privacy   {a.privacy}")
    print(
        f"toggles   comment {'off' if disable_comment else 'on'} · "
        f"duet {'off' if disable_duet else 'on'} · "
        f"stitch {'off' if disable_stitch else 'on'}"
        + ("  · declared AI-generated" if a.aigc else "")
    )
    print(f"upload    {count} chunk(s) of {chunk / 1_048_576:.1f} MB")
    maxdur = info.get("max_video_post_duration_sec")
    if maxdur:
        print(f"limit     this account may post up to {maxdur}s via API")
    if a.privacy != "SELF_ONLY":
        print("\n  NOTE: this will be visible on the account the moment TikTok")
        print("        finishes processing. There is no scheduled-publish option.")
    print("-------------\n")

    if a.dry_run:
        print("DRY RUN: nothing was sent.")
        return

    if not a.yes:
        if input("Post this? type yes to confirm: ").strip().lower() != "yes":
            sys.exit("Aborted. Nothing was posted.")

    body = {
        "post_info": {
            "title": a.title,
            "privacy_level": a.privacy,
            "disable_comment": disable_comment,
            "disable_duet": disable_duet,
            "disable_stitch": disable_stitch,
            "video_cover_timestamp_ms": a.cover_ms,
        },
        "source_info": {
            "source": "FILE_UPLOAD",
            "video_size": size,
            "chunk_size": chunk,
            "total_chunk_count": count,
        },
    }
    if a.aigc:
        body["post_info"]["is_aigc"] = True

    data = api_post(VIDEO_INIT, token, body)
    publish_id = data.get("publish_id")
    upload_url = data.get("upload_url")
    if not publish_id or not upload_url:
        die(f"init returned no upload target:\n{json.dumps(data, indent=2)}")

    print(f"publish_id {publish_id}")
    print("Uploading...")
    upload(upload_url, path, size, chunk, count, mime)

    print("Upload complete. Waiting for TikTok to process...")
    final = poll(publish_id, token, a.timeout, a.interval)

    print("\nDone.")
    print(f"  publish_id  {publish_id}")
    if final:
        ids = final.get("publicaly_available_post_id")  # TikTok's own spelling
        if ids:
            uname = info.get("creator_username")
            for pid in ids:
                print(f"  post        https://www.tiktok.com/@{uname}/video/{pid}")
        elif a.privacy == "SELF_ONLY":
            print("  Posted as SELF_ONLY — visible only to you, in the TikTok app.")


if __name__ == "__main__":
    main()
