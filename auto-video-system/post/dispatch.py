#!/usr/bin/env python3
"""
dispatch.py — watched-queue YouTube publisher.

Runs on a cron every 15 minutes. Reads queue.json, and for every entry whose
video file has appeared and hasn't been posted yet, uploads it to YouTube with
publishAt set to its scheduled slot. YouTube does the actual publishing, so this
machine does NOT need to be awake at the scheduled moment — it only needs to
have run at least once between you dropping the file and the deadline.

Idempotent: a posted entry is recorded in posted.json and never re-uploaded,
even if the cron fires a hundred times.

Usage:
    python3 dispatch.py              # normal run (what cron calls)
    python3 dispatch.py --dry-run    # show what it WOULD do, upload nothing
    python3 dispatch.py --status     # print the whole queue and exit

Layout it expects:
    ~/publishing/
        dispatch.py
        yt_upload.py
        yt_token.json
        queue.json
        posted.json      (created automatically)
        queue/           (drop video files here)
        dispatch.log     (created automatically)
"""

import argparse
import datetime as dt
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

QUEUE_JSON = os.path.join(HERE, "queue.json")
POSTED_JSON = os.path.join(HERE, "posted.json")
QUEUE_DIR = os.path.join(HERE, "queue")
LOG_PATH = os.path.join(HERE, "dispatch.log")

# Upload this far ahead of the slot at most. Stops a manifest full of
# far-future dates from burning the daily quota weeks early.
LOOKAHEAD_DAYS = 7


def log(msg):
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    line = f"{stamp}  {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log(f"FATAL: cannot read {os.path.basename(path)}: {exc}")
        sys.exit(1)


def save_posted(data):
    tmp = POSTED_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, POSTED_JSON)  # atomic; a crash mid-write can't corrupt it


def parse_slot(value):
    """Accept '2026-08-09T09:00:00-04:00' or '2026-08-09 09:00' (assumed ET)."""
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        pass
    try:
        naive = dt.datetime.strptime(value, "%Y-%m-%d %H:%M")
        return naive.replace(tzinfo=dt.timezone(dt.timedelta(hours=-4)))
    except ValueError:
        return None


def validate(queue):
    """Return (entries, problems). Never raises on bad user input."""
    problems = []
    if not isinstance(queue, list):
        return [], ["queue.json must be a JSON list of entries"]
    seen = set()
    good = []
    for i, e in enumerate(queue):
        where = f"entry {i}"
        if not isinstance(e, dict):
            problems.append(f"{where}: not an object")
            continue
        fname, title = e.get("file"), e.get("title")
        if not fname:
            problems.append(f"{where}: missing 'file'")
            continue
        where = f"{fname}"
        if not title:
            problems.append(f"{where}: missing 'title'")
            continue
        if len(title) > 100:
            problems.append(f"{where}: title is {len(title)} chars (limit 100)")
            continue
        if len(e.get("description", "")) > 5000:
            problems.append(f"{where}: description over 5000 chars")
            continue
        if fname in seen:
            problems.append(f"{where}: listed twice in queue.json")
            continue
        seen.add(fname)
        if e.get("publish_at") and parse_slot(e["publish_at"]) is None:
            problems.append(f"{where}: unparseable publish_at {e['publish_at']!r}")
            continue
        good.append(e)
    return good, problems


def describe(entry, posted, now):
    """One-line status for --status / --dry-run."""
    fname = entry["file"]
    path = os.path.join(QUEUE_DIR, fname)
    if fname in posted:
        rec = posted[fname]
        return f"POSTED    {fname}  -> youtu.be/{rec.get('video_id')}  ({rec.get('scheduled_for') or 'immediate'})"
    if not os.path.exists(path):
        return f"WAITING   {fname}  (file not in queue/ yet)"
    if os.path.getsize(path) == 0:
        return f"WAITING   {fname}  (0 bytes — still copying?)"
    if now.timestamp() - os.path.getmtime(path) < 60:
        return f"WAITING   {fname}  (modified <60s ago, letting the copy settle)"
    slot = parse_slot(entry.get("publish_at"))
    if slot and slot > now + dt.timedelta(days=LOOKAHEAD_DAYS):
        return f"TOO EARLY {fname}  (slot {slot.isoformat()}, holding until {LOOKAHEAD_DAYS}d out)"
    if slot and slot <= now:
        return f"READY     {fname}  (slot has passed -> will publish immediately)"
    return f"READY     {fname}  (-> schedule for {slot.isoformat() if slot else 'immediate public'})"


def main():
    ap = argparse.ArgumentParser(description="Watched-queue YouTube publisher.")
    ap.add_argument("--dry-run", action="store_true", help="report only, upload nothing")
    ap.add_argument("--status", action="store_true", help="print the queue and exit")
    args = ap.parse_args()

    os.makedirs(QUEUE_DIR, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)

    raw = load_json(QUEUE_JSON, None)
    if raw is None:
        log(f"No queue.json at {QUEUE_JSON}. Nothing to do.")
        return
    entries, problems = validate(raw)
    for p in problems:
        log(f"SKIP (bad entry): {p}")

    posted = load_json(POSTED_JSON, {})

    if args.status or args.dry_run:
        print(f"\nqueue dir: {QUEUE_DIR}")
        print(f"now:       {now.isoformat()}\n")
        for e in entries:
            print("  " + describe(e, posted, now))
        print()
        if args.status:
            return

    # yt_upload is imported lazily so --status works even if the token is absent
    try:
        import yt_upload
    except ImportError:
        log("FATAL: yt_upload.py must sit next to dispatch.py")
        sys.exit(1)

    todo = []
    for e in entries:
        fname = e["file"]
        if fname in posted:
            continue
        path = os.path.join(QUEUE_DIR, fname)
        if not os.path.exists(path):
            continue
        size = os.path.getsize(path)
        if size == 0:
            log(f"SKIP {fname}: file is 0 bytes (still copying?)")
            continue
        # Guard against picking up a file mid-transfer: require it to have been
        # untouched for 60s before we trust its length.
        if now.timestamp() - os.path.getmtime(path) < 60:
            log(f"SKIP {fname}: modified <60s ago, waiting for the copy to settle")
            continue
        slot = parse_slot(e.get("publish_at"))
        if slot and slot > now + dt.timedelta(days=LOOKAHEAD_DAYS):
            continue
        todo.append((e, path, slot))

    if not todo:
        log("Nothing ready to upload.")
        return

    if args.dry_run:
        log(f"DRY RUN: would upload {len(todo)} file(s); nothing was sent.")
        return

    try:
        creds = yt_upload.load_creds()
        token = yt_upload.access_token(creds)
    except SystemExit as exc:
        log(f"FATAL: auth failed: {exc}")
        raise

    for entry, path, slot in todo:
        fname = entry["file"]
        try:
            snippet = {"title": entry["title"], "categoryId": str(entry.get("category", "24"))}
            if entry.get("description"):
                snippet["description"] = entry["description"]
            if entry.get("tags"):
                tags = entry["tags"]
                snippet["tags"] = tags if isinstance(tags, list) else [
                    t.strip() for t in str(tags).split(",") if t.strip()
                ]

            status = {"selfDeclaredMadeForKids": bool(entry.get("made_for_kids", False))}
            if entry.get("synthetic_media"):
                status["containsSyntheticMedia"] = True

            if slot and slot > now:
                status["privacyStatus"] = "private"
                status["publishAt"] = slot.astimezone(dt.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                plan = f"scheduled for {status['publishAt']}"
            else:
                status["privacyStatus"] = entry.get("privacy", "public")
                plan = f"immediate ({status['privacyStatus']})"

            metadata = {"snippet": snippet, "status": status}
            size = os.path.getsize(path)
            mime = "video/mp4" if path.lower().endswith(".mp4") else "video/*"

            log(f"UPLOAD {fname} ({size / 1_048_576:.1f} MB) -> {plan}")
            session = yt_upload.start_session(token, metadata, size, mime)
            result = yt_upload.upload(session, path, size, token)
            vid = result.get("id")

            posted[fname] = {
                "video_id": vid,
                "uploaded_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "scheduled_for": status.get("publishAt"),
                "title": entry["title"],
            }
            save_posted(posted)  # written after EVERY upload, not at the end
            log(f"OK     {fname} -> https://www.youtube.com/watch?v={vid}")

        except SystemExit as exc:
            log(f"FAIL   {fname}: {exc}")
        except Exception:
            log(f"FAIL   {fname}: unexpected error\n{traceback.format_exc()}")

    log("Run complete.")


if __name__ == "__main__":
    main()
