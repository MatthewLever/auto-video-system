#!/usr/bin/env python3
"""caption_check.py - programmatic caption overflow / safe-area checker.

Method: every ASS event is rendered by the REAL renderer (ffmpeg/libass, same
fontsdir as production burns) at its midpoint onto a solid magenta canvas in ONE
ffmpeg pass (concat-demuxer PTS trick: one magenta frame per probe time,
-fps_mode passthrough, subtitles filter keyed on frame PTS). Non-magenta pixels
= caption ink. numpy gives bbox + violations.

Checks:
  A ink outside platform safe area (per-edge px counts)
  B bbox width > MAX_W_FRAC of frame width (default 0.92)
  C overlapping events -> stacked-collision probe at overlap midpoint
    (combined ink height > 1.25x the taller single event)
  D per-style rendered cap height ('H' probe) > CAP_H_MAX px (default 200)
  E event renders no ink at all (invisible captions, e.g. misparsed style)
Exit 0 = pass, 1 = fail. JSON verdict on stdout or --json file.
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))
MAGENTA = np.array([255, 0, 255])
TOL = 40            # per-channel distance from magenta that counts as ink
MAX_W_FRAC = 0.92
CAP_H_MAX = 200     # px on 1920-high canvas
STACK_FACTOR = 1.25
MIN_OVERLAP = 0.03  # s of time overlap that triggers a collision probe

def parse_time(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)

def parse_ass(path):
    play = [1080, 1920]; styles = {}; events = []
    fmt = None
    for L in open(path, encoding="utf-8-sig"):
        L = L.strip()
        if L.startswith("PlayResX:"): play[0] = int(L.split(":")[1])
        elif L.startswith("PlayResY:"): play[1] = int(L.split(":")[1])
        elif L.startswith("Format:") and "Fontname" in L:
            fmt = [f.strip() for f in L.split(":", 1)[1].split(",")]
        elif L.startswith("Style:"):
            vals = [v.strip() for v in L.split(":", 1)[1].split(",")]
            d = dict(zip(fmt, vals)) if fmt else {}
            styles[vals[0]] = d
        elif L.startswith("Dialogue:"):
            p = L.split(",", 9)
            events.append({"start": parse_time(p[1]), "end": parse_time(p[2]),
                           "style": p[3], "text": p[9]})
    return play, styles, events

def render_at_times(ass, times, w, h, fontsdir):
    """One ffmpeg pass; yields (index, rgb ndarray) for each probe time."""
    td = tempfile.mkdtemp()
    png = os.path.join(td, "m.png")
    import cv2
    canvas = np.zeros((h, w, 3), np.uint8); canvas[:] = (255, 0, 255)  # BGR magenta
    cv2.imwrite(png, canvas)
    ts = [0.0] + [max(0.001, t) for t in times]
    lst = os.path.join(td, "l.txt")
    with open(lst, "w") as f:
        f.write("ffconcat version 1.0\n")
        for i in range(len(ts)):
            f.write(f"file '{png}'\n")
            if i < len(ts) - 1:
                f.write(f"duration {max(0.001, ts[i+1]-ts[i]):.6f}\n")
    cmd = ["ffmpeg", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst,
           "-vf", f"subtitles=f={ass}:fontsdir={fontsdir}",
           "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n = w * h * 3
    frames = []
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
    err = proc.stderr.read().decode()
    proc.wait()
    if len(frames) != len(ts):
        raise RuntimeError(f"expected {len(ts)} frames got {len(frames)}: {err[:400]}")
    return frames[1:]  # drop the t=0 dummy

def ink_stats(frame):
    mask = (np.abs(frame.astype(np.int16) - MAGENTA).max(axis=2) > TOL)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return {"x0": int(xs.min()), "x1": int(xs.max()),
            "y0": int(ys.min()), "y1": int(ys.max()), "px": int(mask.sum()),
            "mask": mask}

def cap_height_probe(style_line_dict, fmtkeys, w, h, fontsdir):
    """Render a bare 'H' in the given style, return ink height in px."""
    vals = [style_line_dict.get(k, "0") for k in fmtkeys]
    hdr = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\n"
           "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\nFormat: " + ", ".join(fmtkeys) +
           "\nStyle: " + ",".join(vals) +
           "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
           f"Dialogue: 0,0:00:00.00,0:00:02.00,{vals[0]},,0,0,0,,H\n")
    with tempfile.NamedTemporaryFile("w", suffix=".ass", delete=False) as f:
        f.write(hdr); p = f.name
    fr = render_at_times(p, [1.0], w, h, fontsdir)[0]
    os.unlink(p)
    s = ink_stats(fr)
    return (s["y1"] - s["y0"] + 1) if s else 0

def check(ass, platform="engine_default", fontsdir=os.path.join(D, "fonts"),
          safe_json=os.path.join(D, "safe_areas.json"),
          max_w_frac=MAX_W_FRAC, cap_h_max=CAP_H_MAX, scale=0.5):
    """scale<1 renders probes on a smaller canvas (PlayRes scaling keeps layout
    geometry identical); all reported px are converted back to full-res."""
    t_start = time.time()
    (pw, ph), styles, events = parse_ass(ass)
    FW, FH = 1080, 1920                      # full-res canvas
    W, H = int(FW * scale), int(FH * scale)  # probe canvas
    inv = 1.0 / scale
    sa = json.load(open(safe_json))["platforms"][platform]
    rect = (int(sa["left"] * scale), int(sa["top"] * scale),
            W - int(sa["right"] * scale), H - int(sa["bottom"] * scale))
    failures, per_event = [], []

    # probe list: midpoints + overlap midpoints
    probes = [(0.5 * (e["start"] + e["end"]), ("mid", i)) for i, e in enumerate(events)]
    overlaps = []
    for i in range(len(events)):
        for j in range(i + 1, len(events)):
            a, b = events[i], events[j]
            o0, o1 = max(a["start"], b["start"]), min(a["end"], b["end"])
            if o1 - o0 > MIN_OVERLAP:
                overlaps.append((i, j))
                probes.append((0.5 * (o0 + o1), ("ovl", (i, j))))
    order = sorted(range(len(probes)), key=lambda k: probes[k][0])
    # dedupe/force strictly increasing for concat
    times, keys, last = [], [], -1.0
    for k in order:
        t, key = probes[k]
        t = max(t, last + 0.002)
        times.append(t); keys.append(key); last = t
    frames = render_at_times(ass, times, W, H, fontsdir)

    mids, ovl_frames = {}, {}
    for (kind, key), fr in zip(keys, frames):
        if kind == "mid": mids[key] = ink_stats(fr)
        else: ovl_frames[key] = ink_stats(fr)

    for i, e in enumerate(events):
        s = mids.get(i)
        rec = {"event": i, "start": e["start"], "end": e["end"],
               "text": e["text"][:60], "issues": []}
        if s is None:
            rec["issues"].append("E:no_ink_rendered")
        else:
            m = s.pop("mask")
            out_l = int(m[:, :rect[0]].sum()); out_r = int(m[:, rect[2]:].sum())
            out_t = int(m[:rect[1], :].sum()); out_b = int(m[rect[3]:, :].sum())
            if out_l + out_r + out_t + out_b > 0:
                rec["issues"].append(
                    f"A:outside_safe_area(l={out_l},r={out_r},t={out_t},b={out_b})")
            wfrac = (s["x1"] - s["x0"] + 1) / W
            if wfrac > max_w_frac:
                rec["issues"].append(f"B:too_wide({wfrac:.2f}>{max_w_frac})")
            rec["bbox"] = {k: int(round(s[k] * inv)) if k != "px" else s[k]
                           for k in ("x0", "x1", "y0", "y1", "px")}
        per_event.append(rec)
        for iss in rec["issues"]:
            failures.append(f"event {i} [{e['text'][:40]}]: {iss}")

    # C: stacked collisions - baseline = each event rendered ISOLATED (a probe
    # inside the shared window would contain both events and mask the stack)
    if overlaps:
        iso_ids = sorted({k for pair in overlaps for k in pair})
        src = open(ass, encoding="utf-8-sig").read().splitlines()
        head = [L for L in src if not L.strip().startswith("Dialogue:")]
        dlg = [L for L in src if L.strip().startswith("Dialogue:")]
        iso_h = {}
        for k in iso_ids:
            with tempfile.NamedTemporaryFile("w", suffix=".ass", delete=False) as f:
                f.write("\n".join(head) + "\n" + dlg[k] + "\n"); p = f.name
            e = events[k]
            fr = render_at_times(p, [0.5 * (e["start"] + e["end"])], W, H, fontsdir)[0]
            os.unlink(p)
            s = ink_stats(fr)
            iso_h[k] = (s["y1"] - s["y0"] + 1) if s else 0
        for (i, j) in overlaps:
            s = ovl_frames.get((i, j))
            hi = max(iso_h.get(i, 0), iso_h.get(j, 0))
            if s and hi and (s["y1"] - s["y0"] + 1) > STACK_FACTOR * hi:
                failures.append(
                    f"events {i}+{j}: C:stacked_collision (combined "
                    f"h={int((s['y1']-s['y0']+1)*inv)}px vs single {int(hi*inv)}px)")

    fmtkeys = ["Name", "Fontname", "Fontsize", "PrimaryColour", "SecondaryColour",
               "OutlineColour", "BackColour", "Bold", "Italic", "Underline",
               "StrikeOut", "ScaleX", "ScaleY", "Spacing", "Angle", "BorderStyle",
               "Outline", "Shadow", "Alignment", "MarginL", "MarginR", "MarginV",
               "Encoding"]
    cap_heights = {}
    for name, sd in styles.items():
        if not any(e["style"] == name for e in events):
            continue
        ch = cap_height_probe(sd, fmtkeys, FW, FH, fontsdir)
        cap_heights[name] = ch
        if ch > cap_h_max:
            failures.append(f"style {name}: D:cap_height {ch}px > {cap_h_max}px")
        if ch == 0:
            failures.append(f"style {name}: E:style_renders_no_ink (misparse?)")

    verdict = {"file": ass, "platform": platform, "events": len(events),
               "overlap_pairs_probed": len(overlaps),
               "style_cap_heights_px": cap_heights,
               "failures": failures, "pass": not failures,
               "runtime_s": round(time.time() - t_start, 2),
               "per_event": per_event}
    return verdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ass", required=True)
    ap.add_argument("--platform", default="engine_default")
    ap.add_argument("--fontsdir", default=os.path.join(D, "fonts"))
    ap.add_argument("--json", help="write verdict JSON here")
    ap.add_argument("--cap-h-max", type=int, default=CAP_H_MAX)
    ap.add_argument("--max-w-frac", type=float, default=MAX_W_FRAC)
    a = ap.parse_args()
    v = check(a.ass, a.platform, a.fontsdir, cap_h_max=a.cap_h_max,
              max_w_frac=a.max_w_frac)
    out = json.dumps({k: v[k] for k in v if k != "per_event"}, indent=1)
    print(out)
    if a.json:
        json.dump(v, open(a.json, "w"), indent=1)
    sys.exit(0 if v["pass"] else 1)

if __name__ == "__main__":
    main()
