#!/usr/bin/env python3
"""encode.py - platform encode profiles for 1080x1920 reels.

Profiles (2026 IG Reels guidance: H.264 High yuv420p, 1080x1920, 30fps,
5-10 Mbps, AAC 48 kHz stereo, mp4 +faststart; sources in safe_areas.json
research log / final report):
  master  - crf 18, preset slow, archival. Audio loudnorm to -14 LUFS / TP -1.5.
  cdn     - HARD <20 MiB (Shopify-CDN-hosted IG upload). Two-pass x264 at
            calculated bitrate: (19.3 MiB * 8 / dur) - 192k audio, then verify;
            if over budget, re-run at 0.96x until under (deterministic ladder).
  preview - <30 MiB chat delivery. crf 26 single pass; fallback ladder
            crf 28/31 if over.
All profiles: -movflags +faststart, yuv420p, High profile, AAC 192k 48k stereo.
"""
import argparse, json, os, subprocess, sys, tempfile

CDN_BUDGET_B = int(19.3 * 1024 * 1024)   # target bytes (under 20 MiB hard cap)
CDN_HARD_B = 20 * 1024 * 1024
PREVIEW_HARD_B = 30 * 1024 * 1024
ABR = 192_000                            # audio bits/s

def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {cmd}\n{r.stderr[-800:]}")
    return r

def duration_of(path):
    r = sh(f"ffprobe -v error -show_entries format=duration -of csv=p=0 '{path}'")
    return float(r.stdout.strip())

def measure_lufs(path):
    r = subprocess.run(
        f"ffmpeg -hide_banner -nostats -i '{path}' -map a -af ebur128=peak=true -f null -",
        shell=True, capture_output=True, text=True)
    txt = r.stderr
    tail = txt[txt.rfind("Summary:"):]
    lufs = tp = None
    lines = iter(tail.splitlines())
    for L in lines:
        if L.strip().startswith("I:"):
            lufs = float(L.split()[1])
        if "True peak" in L:
            for L2 in lines:
                if L2.strip().startswith("Peak:"):
                    tp = float(L2.split()[1]); break
    return lufs, tp

def audio_filter_for_target(src, target=-14.0):
    """Measured gain to hit integrated target, then true-peak limit at -1.5."""
    lufs, tp = measure_lufs(src)
    gain = target - lufs
    return (f"volume={gain:.2f}dB,alimiter=limit=-2.0dB:attack=5:release=50:level=false",
            lufs, tp)

def cdn_bitrate(dur_s, budget_b=CDN_BUDGET_B):
    """bits/s for video: (budget*8/dur) - audio - mux overhead margin (2%)."""
    total_bps = budget_b * 8 / dur_s
    v = int((total_bps - ABR) * 0.98)
    return max(v, 300_000)

COMMON_V = ("-c:v libx264 -profile:v high -pix_fmt yuv420p -r 30 "
            "-x264-params keyint=90:min-keyint=30 -movflags +faststart")
COMMON_A = "-c:a aac -b:a 192k -ar 48000 -ac 2"

def encode(profile, vin, ain, out, vf=None, norm_audio=True, dur=None):
    vf_arg = f"-vf \"{vf}\"" if vf else ""
    a_in = f"-i '{ain}'" if ain else ""
    amap = "-map 1:a" if ain else "-map 0:a"
    af = ""
    if norm_audio:
        flt, lufs, tp = audio_filter_for_target(ain or vin)
        af = f"-af \"{flt}\""
    tcut = f"-t {dur}" if dur else "-shortest"
    if profile == "master":
        sh(f"ffmpeg -y -i '{vin}' {a_in} -map 0:v {amap} {vf_arg} {af} "
           f"{COMMON_V} -preset slow -crf 18 {COMMON_A} {tcut} '{out}'")
    elif profile == "cdn":
        d = dur or duration_of(vin)
        vb = cdn_bitrate(d)
        predicted = (vb / 0.98 + ABR) * d / 8
        attempts = []
        for it in range(4):
            with tempfile.TemporaryDirectory() as td:
                log = os.path.join(td, "x264")
                sh(f"ffmpeg -y -i '{vin}' -map 0:v {vf_arg} {COMMON_V} -preset medium "
                   f"-b:v {vb} -maxrate {int(vb*1.5)} -bufsize {int(vb*2)} "
                   f"-pass 1 -passlogfile {log} -an -f mp4 /dev/null")
                sh(f"ffmpeg -y -i '{vin}' {a_in} -map 0:v {amap} {vf_arg} {af} "
                   f"{COMMON_V} -preset medium -b:v {vb} -maxrate {int(vb*1.5)} "
                   f"-bufsize {int(vb*2)} -pass 2 -passlogfile {log} "
                   f"{COMMON_A} {tcut} '{out}'")
            size = os.path.getsize(out)
            attempts.append({"pass_bitrate": vb, "size": size})
            if size < CDN_HARD_B:
                break
            vb = int(vb * 0.96 * CDN_HARD_B / size)
        return {"profile": "cdn", "duration": d, "video_bitrate": vb,
                "predicted_bytes": int(predicted), "actual_bytes": size,
                "pred_err_pct": round(100 * (size - predicted) / predicted, 2),
                "under_cap": size < CDN_HARD_B, "attempts": attempts}
    elif profile == "preview":
        for crf in (26, 28, 31):
            sh(f"ffmpeg -y -i '{vin}' {a_in} -map 0:v {amap} {vf_arg} {af} "
               f"{COMMON_V} -preset medium -crf {crf} {COMMON_A} {tcut} '{out}'")
            if os.path.getsize(out) < PREVIEW_HARD_B:
                return {"profile": "preview", "crf": crf,
                        "actual_bytes": os.path.getsize(out), "under_cap": True}
        return {"profile": "preview", "crf": crf,
                "actual_bytes": os.path.getsize(out), "under_cap": False}
    else:
        raise ValueError(profile)
    return {"profile": profile, "actual_bytes": os.path.getsize(out)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=["master", "cdn", "preview"])
    ap.add_argument("--vin", required=True)
    ap.add_argument("--ain")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vf", help="e.g. subtitles=... burn filter")
    ap.add_argument("--dur", type=float)
    ap.add_argument("--no-norm", action="store_true")
    a = ap.parse_args()
    r = encode(a.profile, a.vin, a.ain, a.out, a.vf, not a.no_norm, a.dur)
    print(json.dumps(r, indent=1))

if __name__ == "__main__":
    main()
