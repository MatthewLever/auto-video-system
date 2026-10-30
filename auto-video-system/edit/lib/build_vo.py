#!/usr/bin/env python3
"""build_vo.py — assemble the voiceover track from vo_selects.json.

Cuts each chosen take out of its source file (with edge padding), applies the
per-segment gain correction (gain-match BEFORE any normalisation — never
loudnorm raw takes), lays lines end to end with planned gaps, and writes:

  vo.wav          48 kHz mono pcm_s16le, gain-matched (renderer normalises later)
  vo_layout.json  [{line, text, t0, t1}] timeline positions per line
  vo_words.json   {"words":[{word,start,end,prob}]} faster-whisper on vo.wav
                  (timeline-time; feeds both the planner and captions.py)

Gaps: --gap between all lines, overridable per line with --gap-after IDX=SEC
(e.g. --gap-after 4=1.8 for the reveal hold). --head/--tail pad the ends.
"""
import argparse, json, os, subprocess, sys

PAD_IN, PAD_OUT = 0.12, 0.18   # proven silencedetect edge padding
SR = 48000

def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stderr[-2000:] + "\n")
        raise SystemExit("cmd failed: " + " ".join(cmd[:8]))
    return r

def dur_of(path):
    r = sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", path])
    return float(r.stdout.strip())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selects", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--src-root", default="", help="prefix for relative vo source paths")
    ap.add_argument("--gap", type=float, default=0.35)
    ap.add_argument("--gap-after", action="append", default=[],
                    help="IDX=SEC extra hold after line IDX (0-based)")
    ap.add_argument("--head", type=float, default=0.15)
    ap.add_argument("--tail", type=float, default=0.7)
    ap.add_argument("--no-words", action="store_true")
    a = ap.parse_args()

    a.workdir = os.path.abspath(a.workdir)
    os.makedirs(a.workdir, exist_ok=True)
    seg_dir = os.path.join(a.workdir, "vo_segs"); os.makedirs(seg_dir, exist_ok=True)
    sel = json.load(open(a.selects))
    gaps = {}
    for spec in a.gap_after:
        k, v = spec.split("="); gaps[int(k)] = float(v)

    parts, layout, t = [], [], a.head
    concat_lines = []
    sil = os.path.join(seg_dir, "sil_1s.wav")
    sh(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi",
        "-i", "anullsrc=r=%d:cl=mono" % SR, "-t", "1.0", "-c:a", "pcm_s16le", sil])

    def add_silence(sec):
        nonlocal t
        if sec <= 0.005: return
        concat_lines.append("file '%s'\ninpoint 0\noutpoint %.3f" % (sil, min(sec, 1.0)))
        rem = sec - 1.0
        while rem > 0.005:
            concat_lines.append("file '%s'\ninpoint 0\noutpoint %.3f" % (sil, min(rem, 1.0)))
            rem -= 1.0
        t += sec

    add_silence(a.head) if a.head > 0 else None
    t = a.head
    srcmap_p = os.path.join(os.path.dirname(a.selects), "vo_sources.json")
    srcmap = json.load(open(srcmap_p)) if os.path.exists(srcmap_p) else {}
    for i, line in enumerate(sel["lines"]):
        p = line["pick"]
        src = (p["f"] if os.path.isabs(p["f"])
               else srcmap.get(os.path.basename(p["f"]))
               or os.path.join(a.src_root, p["f"]))
        t0 = max(0.0, p["t"][0] - PAD_IN); t1 = p["t"][1] + PAD_OUT
        seg = os.path.join(seg_dir, "line_%02d.wav" % i)
        sh(["ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-ss", "%.3f" % t0, "-to", "%.3f" % t1, "-i", src,
            "-af", "volume=%.2fdB,aresample=%d" % (p.get("gain", 0.0), SR),
            "-ac", "1", "-c:a", "pcm_s16le", seg])
        d = dur_of(seg)
        layout.append({"line": i, "text": line.get("text", ""),
                       "t0": round(t, 3), "t1": round(t + d, 3)})
        concat_lines.append("file '%s'" % seg)
        t += d
        add_silence(gaps.get(i, a.gap) if i < len(sel["lines"]) - 1 else a.tail)

    lst = os.path.join(seg_dir, "concat.txt")
    open(lst, "w").write("\n".join(concat_lines) + "\n")
    vo = os.path.join(a.workdir, "vo.wav")
    sh(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
        "-i", lst, "-c:a", "pcm_s16le", vo])
    json.dump(layout, open(os.path.join(a.workdir, "vo_layout.json"), "w"), indent=1)

    if not a.no_words:
        from faster_whisper import WhisperModel
        m = WhisperModel("base.en", device="cpu", compute_type="int8")
        segs, _ = m.transcribe(vo, word_timestamps=True, vad_filter=True,
                               condition_on_previous_text=False)
        words = []
        for s in segs:
            for w in (s.words or []):
                words.append({"word": w.word, "start": round(w.start, 3),
                              "end": round(w.end, 3), "prob": round(w.probability, 3)})
        json.dump({"words": words}, open(os.path.join(a.workdir, "vo_words.json"), "w"), indent=1)

    print(json.dumps({"vo": vo, "duration": round(dur_of(vo), 3),
                      "lines": len(layout)}))

if __name__ == "__main__":
    main()
