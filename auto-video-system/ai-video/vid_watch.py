#!/usr/bin/env python3
"""
vid_watch.py — let Claude WATCH a video.

Claude cannot take video as input. Verified against Anthropic's own docs
2026-08-09: images and PDFs only, and for GIFs "only the first frame is used."
A feature request for native video was closed as "not planned."

So this script turns a video into things Claude CAN read:
  1. a timestamped CONTACT SHEET  (Claude sees this natively as an image)
  2. FULL-RES frames at suspicious moments
  3. a JSON REPORT of measurable facts ffmpeg can prove

Everything here is ffmpeg + Python stdlib. No API, no account, no cost.
It runs offline on a Raspberry Pi.

WHAT IT MEASURES (deterministic, not vibes):
  scdet           scene cuts you did not author
  freezedetect    frozen / duplicated frames (a classic AI-video failure)
  signalstats     per-frame luma delta (YDIF) = a motion-energy curve
                  -> broken AI motion = YDIF spikes with high local variance
                  -> best 3 seconds  = smooth, high-energy, no cut inside
  ebur128         integrated loudness (LUFS) + true peak
  silencedetect   dead audio
Loudness matters: Omni Flash clips vary up to 26 dB between generations, so
always loudnorm before concatenating.

WHAT IT CANNOT DO: hear. There is no transcription here. ffmpeg can tell you
the audio is -21 LUFS and not silent; it cannot tell you what was said. For
that you need Gemini (video+audio natively, ~$0.003/clip) or AssemblyAI
(185 free hours). See --upgrade.

Usage:
  python3 vid_watch.py clip.mp4
  python3 vid_watch.py clip.mp4 --grid 4x3 --stills 3
  python3 vid_watch.py clip.mp4 --json-only
"""

import argparse
import json
import os
import subprocess
import sys

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def die(msg):
    sys.stderr.write("ERROR: %s\n" % msg)
    sys.exit(1)


def run(cmd, capture=True):
    return subprocess.run(cmd, capture_output=capture, text=True)


def find_font():
    for f in FONT_CANDIDATES:
        if os.path.exists(f):
            return f
    return None


def probe(path):
    r = run(["ffprobe", "-v", "error", "-show_entries",
             "format=duration,size:stream=codec_type,width,height,r_frame_rate,codec_name",
             "-of", "json", path])
    if r.returncode:
        die("ffprobe failed: %s" % r.stderr.strip())
    d = json.loads(r.stdout)
    fmt = d.get("format", {})
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), None)
    fps = 0.0
    try:
        n, den = v.get("r_frame_rate", "0/1").split("/")
        fps = round(float(n) / float(den), 2)
    except Exception:
        pass
    return {
        "duration_s": round(float(fmt.get("duration", 0)), 2),
        "size_mb": round(int(fmt.get("size", 0)) / 1048576.0, 2),
        "width": v.get("width"), "height": v.get("height"),
        "fps": fps, "video_codec": v.get("codec_name"),
        "has_audio": bool(a), "audio_codec": (a or {}).get("codec_name"),
        "aspect": ("9:16 vertical" if v.get("height", 0) > v.get("width", 1)
                   else "16:9 landscape"),
    }


def scene_cuts(path, threshold=10):
    r = run(["ffprobe", "-v", "error", "-f", "lavfi",
             "movie=%s,scdet=s=1:t=%d" % (path.replace("\\", "/").replace(":", "\\:"), threshold),
             "-show_entries", "frame=pts_time:frame_tags=lavfi.scd.score",
             "-of", "csv=p=0"])
    cuts = []
    for line in (r.stdout or "").strip().splitlines():
        parts = [p for p in line.split(",") if p]
        if len(parts) >= 2:
            try:
                cuts.append({"t": round(float(parts[0]), 2),
                             "score": round(float(parts[1]), 1)})
            except ValueError:
                pass
    return cuts


def freezes(path):
    r = run(["ffmpeg", "-hide_banner", "-i", path,
             "-vf", "freezedetect=n=-60dB:d=0.5", "-f", "null", "-"], capture=True)
    out = []
    cur = {}
    for line in (r.stderr or "").splitlines():
        if "freeze_start" in line:
            try:
                cur = {"start": round(float(line.rsplit(":", 1)[1].strip()), 2)}
            except ValueError:
                pass
        elif "freeze_duration" in line and cur:
            try:
                cur["duration"] = round(float(line.rsplit(":", 1)[1].strip()), 2)
                out.append(cur)
                cur = {}
            except ValueError:
                pass
    return out


def motion_curve(path):
    esc = path.replace("\\", "/").replace(":", "\\:")
    r = run(["ffprobe", "-v", "error", "-f", "lavfi",
             "movie=%s,signalstats" % esc,
             "-show_entries", "frame=pts_time:frame_tags=lavfi.signalstats.YDIF",
             "-of", "csv=p=0"])
    pts = []
    for line in (r.stdout or "").strip().splitlines():
        parts = [p for p in line.split(",") if p]
        if len(parts) >= 2:
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return pts


def best_window(curve, cuts, window=3.0):
    """Highest smooth motion energy with no scene cut inside."""
    if not curve:
        return None
    cut_times = [c["t"] for c in cuts]
    best = None
    for i, (t0, _) in enumerate(curve):
        seg = [y for (t, y) in curve if t0 <= t < t0 + window]
        if len(seg) < 3:
            continue
        if any(t0 - 0.4 <= ct <= t0 + window + 0.4 for ct in cut_times):
            continue
        mean = sum(seg) / len(seg)
        var = sum((y - mean) ** 2 for y in seg) / len(seg)
        # reward energy, punish jitter (high variance = warping/stutter)
        score = mean / (1.0 + (var ** 0.5) / max(mean, 1e-6))
        if best is None or score > best["score"]:
            best = {"start": round(t0, 2), "end": round(t0 + window, 2),
                    "score": round(score, 2), "mean_motion": round(mean, 2)}
    return best


def loudness(path):
    r = run(["ffmpeg", "-hide_banner", "-i", path, "-af", "ebur128=peak=true",
             "-f", "null", "-"], capture=True)
    out = {}
    lines = (r.stderr or "").splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("I:") and "LUFS" in s:
            out["integrated_lufs"] = s.split()[1]
        if s.startswith("Peak:") and "dBFS" in s:
            out["true_peak_dbfs"] = s.split()[1]
    return out


def silences(path):
    r = run(["ffmpeg", "-hide_banner", "-i", path,
             "-af", "silencedetect=n=-35dB:d=0.3", "-f", "null", "-"], capture=True)
    out = []
    for line in (r.stderr or "").splitlines():
        if "silence_start" in line:
            try:
                out.append(round(float(line.rsplit(":", 1)[1].strip()), 2))
            except ValueError:
                pass
    return out


def contact_sheet(path, out, dur, grid="4x3", tile_w=270):
    cols, rows = (int(x) for x in grid.lower().split("x"))
    n = cols * rows
    step = max(dur / float(n), 0.05)
    fps_expr = "1/%.4f" % step
    tile_h = int(tile_w * 16 / 9)
    font = find_font()
    dt = ""
    if font:
        dt = (",drawtext=fontfile=%s:text='%%{pts\\:hms}':x=6:y=6:fontsize=26:"
              "fontcolor=yellow:box=1:boxcolor=black@0.6" % font)
    vf = ("fps=%s,scale=%d:%d:force_original_aspect_ratio=decrease,"
          "pad=%d:%d:(ow-iw)/2:(oh-ih)/2:black%s,tile=%dx%d:padding=4:color=black"
          % (fps_expr, tile_w, tile_h, tile_w, tile_h, dt, cols, rows))
    r = run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
             "-vf", vf, "-frames:v", "1", "-q:v", "3", out])
    return out if r.returncode == 0 and os.path.exists(out) else None


def stills(path, times, prefix, width=900):
    made = []
    for i, t in enumerate(times):
        o = "%s_%02d.jpg" % (prefix, i + 1)
        r = run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "%.2f" % t,
                 "-i", path, "-vf", "scale=%d:-2" % width,
                 "-frames:v", "1", "-q:v", 2 and "2", o])
        if r.returncode == 0 and os.path.exists(o):
            made.append({"t": round(t, 2), "file": o})
    return made


def main():
    ap = argparse.ArgumentParser(description="Turn a video into things Claude can read.")
    ap.add_argument("video")
    ap.add_argument("--grid", default="4x3")
    ap.add_argument("--stills", type=int, default=3,
                    help="Full-res frames at flagged moments (0 to skip).")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--upgrade", action="store_true",
                    help="Explain what this cannot do and what fixes it.")
    args = ap.parse_args()

    if args.upgrade:
        print(__doc__.split("WHAT IT CANNOT DO:")[1].split("Usage:")[0].strip())
        return

    path = os.path.expanduser(args.video)
    if not os.path.exists(path):
        die("No such file: %s" % path)
    prefix = args.out_prefix or os.path.splitext(os.path.basename(path))[0]

    info = probe(path)
    cuts = scene_cuts(path)
    frz = freezes(path)
    curve = motion_curve(path)
    best = best_window(curve, cuts)
    rep = {
        "file": os.path.basename(path),
        "info": info,
        "scene_cuts": cuts,
        "frozen_segments": frz,
        "best_3s": best,
        "loudness": loudness(path) if info["has_audio"] else None,
        "silence_starts": silences(path) if info["has_audio"] else None,
        "motion_samples": len(curve),
    }

    flags = []
    if frz:
        flags.append("FROZEN FRAMES — %d segment(s); classic AI-video failure" % len(frz))
    if cuts:
        flags.append("%d scene cut(s) detected at %s — unexpected in a single "
                     "generated shot" % (len(cuts), ", ".join(str(c["t"]) for c in cuts)))
    if info["has_audio"] and rep["loudness"].get("integrated_lufs"):
        try:
            lufs = float(rep["loudness"]["integrated_lufs"])
            if lufs < -20:
                flags.append("QUIET (%.1f LUFS) — loudnorm to about -14 before posting" % lufs)
            if lufs > -9:
                flags.append("HOT (%.1f LUFS) — will be turned down by the platform" % lufs)
        except ValueError:
            pass
    if not info["has_audio"]:
        flags.append("NO AUDIO TRACK")
    if curve:
        ys = [y for _, y in curve]
        mean = sum(ys) / len(ys)
        spikes = [round(t, 2) for t, y in curve if y > mean * 3 and mean > 0]
        if spikes:
            flags.append("MOTION SPIKES at %s — check these frames for warping"
                         % ", ".join(str(s) for s in spikes[:6]))
    rep["flags"] = flags

    if not args.json_only:
        sheet = contact_sheet(path, prefix + "_sheet.jpg", info["duration_s"], args.grid)
        rep["contact_sheet"] = sheet
        if args.stills:
            picks = [c["t"] for c in cuts][:args.stills]
            picks += [f["start"] for f in frz][:args.stills]
            if best:
                picks.append(best["start"] + 1.0)
            while len(picks) < args.stills and info["duration_s"]:
                picks.append(info["duration_s"] * (len(picks) + 1) / (args.stills + 1.0))
            rep["stills"] = stills(path, sorted(set(round(p, 2) for p in picks))[:args.stills],
                                   prefix + "_still")

    out = prefix + "_watch.json"
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=2)
    print(json.dumps(rep, indent=2))
    print("\nWrote %s" % out, file=sys.stderr)


if __name__ == "__main__":
    main()
