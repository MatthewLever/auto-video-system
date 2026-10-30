#!/usr/bin/env python3
"""
vid_gen.py — YoungSponge video generation PRIMITIVE.

Generates one short vertical clip from a REAL product photo (image-to-video).
Two backends behind one interface, chosen with --provider:

  fal     Kling 3.0 Pro / Seedance via fal.ai   [WORKING]
  google  Gemini Omni Flash                     [BLOCKED: see note below]

This is a PRIMITIVE, like yt_upload.py / th_post.py:
  - no queue logic, no time gate, no scheduling policy
  - scheduling policy lives in dispatch.py, in one place
  - draft-and-confirm by default; --yes for automation
  - never deletes anything

Stdlib-only Python 3, zero pip dependencies. Runs as-is on a Raspberry Pi.
ffmpeg is used ONLY for --prep (building a 9:16 start frame) and is optional.

------------------------------------------------------------------------------
VERIFIED THE HARD WAY, 2026-08-08 — do not re-litigate:

1. Omni Flash has NO duration parameter. `duration_seconds` returns
   400 "Unknown parameter". Clip length is controlled by PROMPT TEXT ONLY,
   using timecode syntax like "[0-3s] ... [3-8s] ...".
2. The Gemini API bills from PREPAY CREDITS bought inside AI Studio. The
   $300 Google Cloud free-trial credit is EXPLICITLY EXCLUDED from the
   Gemini API as of March 2026. Cloud billing being set up is not enough.
3. Google Cloud API keys made via Agent Platform are restricted to
   "Agent Platform API" and return 403 API_KEY_SERVICE_BLOCKED against
   generativelanguage.googleapis.com. Agent Platform and Gemini API are
   MUTUALLY EXCLUSIVE on one key -- pick Gemini API.
4. Kling image-to-video takes its ASPECT RATIO FROM THE START IMAGE.
   There is no aspect parameter. Feed it a 9:16 frame or you get 16:9.
   Use --prep to build one.
5. fal accepts a base64 data URI anywhere it wants a URL, so no file
   hosting is needed.
------------------------------------------------------------------------------

Usage:
  python3 vid_gen.py --set-key fal
  python3 vid_gen.py --prep shoe.png --prep-out shoe_9x16.jpg
  python3 vid_gen.py --image shoe_9x16.jpg --prompt "..." --duration 5
  python3 vid_gen.py --estimate --duration 5
  python3 vid_gen.py --dry-run --image ... --prompt "..."
"""

import argparse
import base64
import json
import mimetypes
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------- config

PROVIDERS = {
    "fal": {
        "key_file": os.path.join(HERE, "fal_key.json"),
        "queue": "https://queue.fal.run",
        "models": {
            # label: (model id, $/sec with audio, notes)
            "kling":    ("fal-ai/kling-video/v3/pro/image-to-video", 0.168),
            "kling-std": ("fal-ai/kling-video/v3/standard/image-to-video", 0.126),
            "seedance": ("bytedance/seedance-2.5/image-to-video", 0.47),
        },
        "default_model": "kling",
    },
    "google": {
        "key_file": os.path.join(HERE, "gemini_key.json"),
        "url": "https://generativelanguage.googleapis.com/v1beta/interactions",
        "model": "gemini-omni-flash-preview",
        "usd_per_sec": 0.10,
    },
}

VALID_ASPECTS = ("9:16", "16:9")


# ------------------------------------------------------------------- utilities

def die(msg, code=1):
    sys.stderr.write("ERROR: %s\n" % msg)
    sys.exit(code)


def atomic_write(path, data, mode=0o600, binary=False):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d)
    try:
        with os.fdopen(fd, "wb" if binary else "w") as fh:
            fh.write(data)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_key(provider):
    path = PROVIDERS[provider]["key_file"]
    if not os.path.exists(path):
        die("No %s key. Run:  python3 vid_gen.py --set-key %s" % (provider, provider))
    try:
        with open(path) as fh:
            key = json.load(fh).get("api_key", "").strip()
    except (OSError, ValueError) as exc:
        die("Could not read %s: %s" % (path, exc))
    if not key:
        die("%s has no 'api_key'." % path)
    return key


def set_key(provider):
    where = {"fal": "fal.ai/dashboard/keys",
             "google": "aistudio.google.com/apikey"}[provider]
    print("Paste your %s API key (from %s)." % (provider, where))
    print("It is not written to disk until you press Enter.")
    key = input("key> ").strip()
    if not key:
        die("Empty key, nothing written.")
    path = PROVIDERS[provider]["key_file"]
    atomic_write(path, json.dumps({"api_key": key}, indent=2) + "\n")
    print("Wrote %s (mode 600)." % path)


def data_uri(path):
    """fal accepts a base64 data URI anywhere it expects a URL."""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        die("Image not found: %s" % path)
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as fh:
        raw = fh.read()
    if len(raw) > 8 * 1024 * 1024:
        print("  ! %.1f MB start frame. Run --prep first to shrink it."
              % (len(raw) / 1048576.0))
    return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii")), len(raw)


# ------------------------------------------------------------------------ prep

def sample_corner(src):
    """Average the top-left 200x200 of the source down to one pixel -> 0xRRGGBB."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", os.path.expanduser(src),
             "-vf", "crop=200:200:0:0,scale=1:1",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-frames:v", "1", "-"],
            capture_output=True, check=True)
        if len(r.stdout) >= 3:
            return "0x%02X%02X%02X" % tuple(r.stdout[:3])
    except Exception:
        pass
    return None


def prep_frame(src, out, width=1080, height=1920, bg="0x2A2A2A"):
    """Build a 9:16 start frame with ffmpeg. Kling derives aspect from it."""
    src, out = os.path.expanduser(src), os.path.expanduser(out)
    if not os.path.exists(src):
        die("Image not found: %s" % src)
    # Pad with the source image's OWN corner colour, sampled from the file.
    # Tried and rejected: a blurred pillarbox (turns a dark product shot into a
    # coloured smear that the video model then animates), and a hardcoded grey
    # (leaves a visible seam against the photo's background).
    bg = sample_corner(src) or bg
    vf = ("scale=%d:%d:force_original_aspect_ratio=decrease,"
          "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:%s" % (width, height, width, height, bg))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
           "-vf", vf, "-q:v", "2", out]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        die("ffmpeg not found. Install it, or supply an already-9:16 image.")
    except subprocess.CalledProcessError as exc:
        die("ffmpeg failed (%s)." % exc)
    print("Wrote %s (%dx%d)" % (out, width, height))
    return out


# ------------------------------------------------------------------- backends

def http(url, key_header, body=None, timeout=120, method=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in key_header.items():
        req.add_header(k, v)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1500]
        die("HTTP %s from %s\n%s" % (exc.code, url.split("/")[2], detail))
    except urllib.error.URLError as exc:
        die("Network error: %s\n"
            "If this is the Mac shell mounted into Claude, it has NO network. "
            "Run in your own Terminal instead." % exc)


def run_fal(args, body_extra=None):
    cfg = PROVIDERS["fal"]
    model_id, rate = cfg["models"][args.model]
    key = load_key("fal")
    hdr = {"Authorization": "Key %s" % key}

    uri, nbytes = data_uri(args.image)
    body = {
        "start_image_url": uri,
        "prompt": args.prompt,
        "duration": str(args.duration),
        "generate_audio": not args.no_audio,
    }
    if args.negative:
        body["negative_prompt"] = args.negative
    if args.end_image:
        body["end_image_url"] = data_uri(args.end_image)[0]
    if body_extra:
        body.update(body_extra)

    cost = args.duration * (rate if not args.no_audio else rate * 0.667)
    show_draft(args, model_id, cost, body, nbytes)
    if args.dry_run:
        print("Dry run. Nothing sent, nothing spent.")
        return
    if not confirm(args, cost):
        return

    started = time.time()
    sub = http("%s/%s" % (cfg["queue"], model_id), hdr, body, timeout=120)
    rid = sub.get("request_id")
    status_url = sub.get("status_url")
    result_url = sub.get("response_url")
    print("Queued. request_id=%s" % rid)

    last = None
    while True:
        if time.time() - started > args.timeout:
            die("Timed out after %ds. Job may still finish; check request_id %s"
                % (args.timeout, rid))
        st = http(status_url, hdr, timeout=60)
        s = st.get("status")
        if s != last:
            print("  %s  (%ds)" % (s, int(time.time() - started)))
            last = s
        if s == "COMPLETED":
            break
        if s in ("FAILED", "ERROR"):
            die("Generation failed: %s" % json.dumps(st)[:1000])
        time.sleep(5)

    out = http(result_url, hdr, timeout=120)
    video = (out.get("video") or {})
    url = video.get("url")
    if not url:
        die("No video in result:\n%s" % json.dumps(out, indent=2)[:1500])
    download(url, args.out)
    print("Done in %.0fs." % (time.time() - started))
    probe(args.out)


def run_google(args):
    cfg = PROVIDERS["google"]
    key = load_key("google")
    path = os.path.expanduser(args.image)
    mime = mimetypes.guess_type(path)[0]
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    body = {
        "model": cfg["model"],
        "input": [
            {"type": "image", "data": b64, "mime_type": mime},
            {"type": "text", "text": args.prompt},
        ],
        "response_format": {"type": "video", "aspect_ratio": args.aspect,
                            "delivery": "uri"},
        "generation_config": {"video_config": {"task": "image_to_video"}},
    }
    cost = args.duration * cfg["usd_per_sec"]
    show_draft(args, cfg["model"], cost, {"input": "<image+text>"}, 0)
    if args.dry_run:
        print("Dry run. Nothing sent, nothing spent.")
        return
    if not confirm(args, cost):
        return
    payload = http("%s?key=%s" % (cfg["url"], key), {}, body, timeout=args.timeout)
    for step in payload.get("steps", []):
        for item in step.get("content", []):
            if item.get("type") == "video":
                if item.get("data"):
                    atomic_write(os.path.expanduser(args.out),
                                 base64.b64decode(item["data"]),
                                 mode=0o644, binary=True)
                    print("Wrote %s" % args.out)
                    return probe(args.out)
                for f in ("uri", "url", "file_uri"):
                    if item.get(f):
                        return download(item[f], args.out)
    die("No video in response:\n%s" % json.dumps(payload, indent=2)[:2000])


# --------------------------------------------------------------------- output

def download(url, out_path):
    with urllib.request.urlopen(url, timeout=300) as resp:
        data = resp.read()
    atomic_write(os.path.expanduser(out_path), data, mode=0o644, binary=True)
    print("Wrote %s (%.2f MB)" % (out_path, len(data) / 1048576.0))


def probe(path):
    """Report what we actually got. Duration is prompt-driven, so measure it."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=width,height,codec_type",
             "-of", "json", os.path.expanduser(path)],
            capture_output=True, text=True, check=True)
        d = json.loads(r.stdout)
        dur = float(d.get("format", {}).get("duration", 0))
        dims = [(s.get("width"), s.get("height")) for s in d.get("streams", [])
                if s.get("codec_type") == "video"]
        has_audio = any(s.get("codec_type") == "audio" for s in d.get("streams", []))
        w, h = dims[0] if dims else ("?", "?")
        print("  -> %.1fs, %sx%s, audio=%s" % (dur, w, h, "YES" if has_audio else "NO"))
    except Exception:
        pass


def show_draft(args, model_id, cost, body, nbytes):
    print("=" * 70)
    print("PROVIDER  %s        MODEL  %s" % (args.provider, model_id))
    print("START     %s%s" % (args.image,
                              " (%.2f MB)" % (nbytes / 1048576.0) if nbytes else ""))
    print("DURATION  %ss requested     AUDIO  %s" % (
        args.duration, "off" if args.no_audio else "ON"))
    print("OUT       %s" % args.out)
    print("COST      ~$%.2f" % cost)
    print("-" * 70)
    print("PROMPT:")
    print(args.prompt)
    print("=" * 70)


def confirm(args, cost):
    if args.yes:
        return True
    ans = input("Send this and spend ~$%.2f? [y/N] " % cost).strip().lower()
    if ans in ("y", "yes"):
        return True
    print("Aborted. Nothing sent, nothing spent.")
    return False


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(
        description="Generate one short vertical clip from a real product photo.")
    ap.add_argument("--provider", default="fal", choices=sorted(PROVIDERS))
    ap.add_argument("--model", default="kling",
                    help="fal only: kling | kling-std | seedance")
    ap.add_argument("--image", help="Start frame. A REAL photo of the product.")
    ap.add_argument("--end-image", help="Optional final frame, for chaining clips.")
    ap.add_argument("--prompt")
    ap.add_argument("--duration", type=int, default=5,
                    help="Seconds. Kling 3-15. Google: ESTIMATE ONLY (no such param).")
    ap.add_argument("--aspect", default="9:16", choices=VALID_ASPECTS,
                    help="Google only. Kling takes aspect from the start image.")
    ap.add_argument("--negative", help="Negative prompt (fal).")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--out", default="clip.mp4")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--set-key", metavar="PROVIDER", choices=sorted(PROVIDERS))
    ap.add_argument("--prep", metavar="SRC", help="Build a 9:16 start frame.")
    ap.add_argument("--prep-out", default="start_9x16.jpg")
    ap.add_argument("--schedule", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.schedule:
        die("--schedule is not supported here and is refused rather than "
            "silently ignored. vid_gen.py is a primitive: it generates a clip "
            "and stops. Scheduling policy belongs in dispatch.py.")
    if args.set_key:
        return set_key(args.set_key)
    if args.prep:
        return prep_frame(args.prep, args.prep_out)
    if args.estimate:
        if args.provider == "fal":
            rate = PROVIDERS["fal"]["models"][args.model][1]
        else:
            rate = PROVIDERS["google"]["usd_per_sec"]
        print("%ds x $%.3f/s = $%.2f" % (args.duration, rate, args.duration * rate))
        return

    if not args.image:
        die("--image is required. Text-to-video invents its own product and "
            "will not match your actual shoe.")
    if not args.prompt:
        die("--prompt is required.")

    if args.provider == "fal":
        if args.model not in PROVIDERS["fal"]["models"]:
            die("--model must be one of: %s"
                % ", ".join(sorted(PROVIDERS["fal"]["models"])))
        if not 3 <= args.duration <= 15:
            die("Kling accepts 3-15 seconds.")
        run_fal(args)
    else:
        run_google(args)


if __name__ == "__main__":
    main()
