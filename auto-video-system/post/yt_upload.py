#!/usr/bin/env python3
"""
yt_upload.py — upload a video to YouTube via the Data API v3 (resumable).

Standard library only. No pip install. Requires Python 3.8+.
Needs yt_token.json in this directory — run yt_auth.py once to create it.

Usage:
    python3 yt_upload.py --file ~/Downloads/Doritos-edit.mp4 \\
        --title "Dorito shoes" \\
        --description "..." \\
        --tags "3dprinting,shoes,doritos" \\
        --privacy private

Defaults to privacy=private and prints the full metadata for approval before
anything leaves the machine. Pass --yes to skip the prompt (for cron/n8n).

Quota: videos.insert has its own bucket of 100 uploads/day, resetting at
midnight US Pacific.
"""

import argparse
import http.client
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(HERE, "yt_token.json")
UPLOAD_HOST = "www.googleapis.com"
UPLOAD_PATH = "/upload/youtube/v3/videos"
CHUNK = 8 * 1024 * 1024  # 8 MiB; must be a multiple of 256 KiB
MAX_RETRIES = 6

CATEGORIES = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
    "15": "Pets & Animals", "17": "Sports", "19": "Travel & Events",
    "20": "Gaming", "22": "People & Blogs", "23": "Comedy",
    "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style",
    "27": "Education", "28": "Science & Technology",
}


def die(msg):
    sys.exit(f"error: {msg}")


def load_creds():
    if not os.path.exists(TOKEN_PATH):
        die(f"{TOKEN_PATH} not found. Run: python3 yt_auth.py")
    with open(TOKEN_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def access_token(creds):
    """Trade the refresh token for a short-lived access token."""
    body = urllib.parse.urlencode(
        {
            "client_id": creds["client_id"],
            "client_secret": creds["client_secret"],
            "refresh_token": creds["refresh_token"],
            "grant_type": "refresh_token",
        }
    ).encode()
    req = urllib.request.Request(
        creds.get("token_uri", "https://oauth2.googleapis.com/token"),
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)["access_token"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        if "invalid_grant" in detail:
            die(
                "Refresh token rejected (invalid_grant). Usual causes:\n"
                "  - the OAuth consent screen is still in 'Testing' (7-day expiry).\n"
                "    Publish it: https://console.cloud.google.com/auth/audience\n"
                "  - access was revoked at https://myaccount.google.com/permissions\n"
                "  - the OAuth client was deleted or its secret was rotated\n"
                "Fix the cause, then re-run: python3 yt_auth.py"
            )
        die(f"Token refresh failed ({exc.code}):\n{detail}")


def start_session(token, metadata, size, mime):
    """Open a resumable upload session; returns the session URL."""
    payload = json.dumps(metadata).encode("utf-8")
    query = urllib.parse.urlencode(
        {"uploadType": "resumable", "part": ",".join(metadata.keys())}
    )
    conn = http.client.HTTPSConnection(UPLOAD_HOST, timeout=120)
    conn.request(
        "POST",
        f"{UPLOAD_PATH}?{query}",
        body=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "Content-Length": str(len(payload)),
            "X-Upload-Content-Length": str(size),
            "X-Upload-Content-Type": mime,
        },
    )
    resp = conn.getresponse()
    body = resp.read().decode("utf-8", "replace")
    location = resp.getheader("Location")
    status = resp.status
    conn.close()
    if status != 200 or not location:
        die(f"Could not start the upload session (HTTP {status}):\n{body}")
    return location


def _put(session_url, data, content_range, token):
    parsed = urllib.parse.urlparse(session_url)
    path = parsed.path + ("?" + parsed.query if parsed.query else "")
    conn = http.client.HTTPSConnection(parsed.netloc, timeout=600)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Length": str(len(data)),
        "Content-Range": content_range,
    }
    try:
        conn.request("PUT", path, body=data, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.getheader("Range"), resp.read().decode("utf-8", "replace")
    finally:
        conn.close()


def committed_bytes(session_url, total, token):
    """Ask the server how much it already has. Returns next offset."""
    status, rng, _ = _put(session_url, b"", f"bytes */{total}", token)
    if status in (200, 201):
        return total
    if status == 308 and rng:
        return int(rng.split("-")[-1]) + 1
    return 0


def upload(session_url, path, size, token):
    offset = 0
    attempt = 0
    with open(path, "rb") as fh:
        while offset < size:
            fh.seek(offset)
            data = fh.read(CHUNK)
            end = offset + len(data) - 1
            try:
                status, rng, body = _put(
                    session_url, data, f"bytes {offset}-{end}/{size}", token
                )
            except (http.client.HTTPException, OSError) as exc:
                status, rng, body = 0, None, str(exc)

            if status in (200, 201):
                return json.loads(body)
            if status == 308:
                attempt = 0
                offset = int(rng.split("-")[-1]) + 1 if rng else offset + len(data)
                pct = 100.0 * offset / size
                print(f"  {offset:,}/{size:,} bytes ({pct:.1f}%)", flush=True)
                continue
            if status in (0, 500, 502, 503, 504) or status == 429:
                attempt += 1
                if attempt > MAX_RETRIES:
                    die(f"Gave up after {MAX_RETRIES} retries. Last response:\n{body}")
                wait = min(2 ** attempt, 64)
                print(f"  transient failure (HTTP {status}); retrying in {wait}s", flush=True)
                time.sleep(wait)
                offset = committed_bytes(session_url, size, token)
                continue
            if status == 404:
                die("Upload session expired (404). Re-run the command to start over.")
            die(f"Upload failed (HTTP {status}):\n{body}")
    die("Reached end of file without a completion response.")


def build_metadata(a):
    snippet = {"title": a.title, "categoryId": a.category}
    if a.description:
        snippet["description"] = a.description
    if a.tags:
        snippet["tags"] = [t.strip() for t in a.tags.split(",") if t.strip()]
    status = {
        "privacyStatus": a.privacy,
        "selfDeclaredMadeForKids": a.made_for_kids,
    }
    if a.synthetic_media:
        status["containsSyntheticMedia"] = True
    if a.publish_at:
        if a.privacy != "private":
            die("--publish-at requires --privacy private (YouTube schedules private videos).")
        status["publishAt"] = a.publish_at
    return {"snippet": snippet, "status": status}


def main():
    ap = argparse.ArgumentParser(description="Upload a video to YouTube.")
    ap.add_argument("--file", required=True, help="path to the video file")
    ap.add_argument("--title", required=True, help="video title (max 100 chars)")
    ap.add_argument("--description", default="", help="video description (max 5000 chars)")
    ap.add_argument("--tags", default="", help="comma-separated tags")
    ap.add_argument("--category", default="22", help=f"category ID; one of {sorted(CATEGORIES)}")
    ap.add_argument(
        "--privacy",
        default="private",
        choices=["private", "unlisted", "public"],
        help="default: private",
    )
    ap.add_argument("--publish-at", help="RFC3339 UTC, e.g. 2026-08-10T17:00:00Z (needs --privacy private)")
    ap.add_argument("--made-for-kids", action="store_true", help="declare as made for kids")
    ap.add_argument("--synthetic-media", action="store_true", help="declare AI-generated/altered content")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    a = ap.parse_args()

    path = os.path.expanduser(a.file)
    if not os.path.isfile(path):
        die(f"No such file: {path}")
    size = os.path.getsize(path)
    if size == 0:
        die("File is empty.")
    if len(a.title) > 100:
        die(f"Title is {len(a.title)} chars; YouTube's limit is 100.")
    if len(a.description) > 5000:
        die(f"Description is {len(a.description)} chars; YouTube's limit is 5000.")
    if a.category not in CATEGORIES:
        die(f"Unknown category ID {a.category}. Valid: " +
            ", ".join(f"{k}={v}" for k, v in sorted(CATEGORIES.items(), key=lambda x: int(x[0]))))
    mime = mimetypes.guess_type(path)[0] or "video/*"

    metadata = build_metadata(a)
    creds = load_creds()  # fail before showing a draft if auth was never set up

    print("\n--- draft ---")
    print(f"file      {path}  ({size / 1_048_576:.1f} MB, {mime})")
    print(f"title     {a.title}")
    print(f"category  {a.category} ({CATEGORIES[a.category]})")
    print(f"privacy   {a.privacy}" + (f"  -> public at {a.publish_at}" if a.publish_at else ""))
    if metadata["snippet"].get("tags"):
        print(f"tags      {', '.join(metadata['snippet']['tags'])}")
    if a.description:
        preview = a.description if len(a.description) < 300 else a.description[:300] + "..."
        print(f"description\n{preview}")
    print("-------------\n")

    if not a.yes:
        if input("Upload this? type yes to confirm: ").strip().lower() != "yes":
            sys.exit("Aborted. Nothing was uploaded.")

    token = access_token(creds)
    session = start_session(token, metadata, size, mime)
    print("Uploading...")
    result = upload(session, path, size, token)

    vid = result.get("id")
    print("\nDone.")
    print(f"  video id  {vid}")
    print(f"  watch     https://www.youtube.com/watch?v={vid}")
    print(f"  studio    https://studio.youtube.com/video/{vid}/edit")
    if a.privacy == "private" and not a.publish_at:
        print("\n  Still private. Flip it to public in Studio, or re-run with --privacy public.")


if __name__ == "__main__":
    main()
