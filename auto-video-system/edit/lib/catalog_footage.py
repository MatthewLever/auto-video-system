#!/usr/bin/env python3
"""catalog_footage.py — zero-model-call footage cataloger for the edit engine.
Input: dir of 640px/15fps proxy clips. Output: compact catalog.json (+ --debug full metrics).
Single decode pass per clip: sharpness, exposure, motion, jitter (phase-correlation),
frozen holds / dead air, scene cuts, faces, ranked best-windows for any target duration.
"""
import cv2, numpy as np, json, subprocess, sys, os, time, argparse

# ---------- calibration constants (set empirically on synthetic proxy corpus, see report) ----------
WIN = 0.5              # scoring window seconds
FROZEN_DIFF = None     # set per-clip: max(1.0, 2.2 * p10 of frame diffs) — adapts to noise floor
FROZEN_ABS = 1.0       # absolute floor
CUT_RESP = 0.55        # phase-correlation response below this at a spike = content cut
CUT_ABS = 12.0
CUT_RATIO = 1.5        # diff must exceed ratio * surrounding median
LAP_GOOD = 250.0       # Laplacian variance mapping to sharp_n=1.0 (log scale)
CLIP_TOL = 0.03        # tolerated exposure-clip fraction
J0 = 1.6               # jitter (proxy px) at which steadiness = 0.5
HOLD_MAX = 3.5         # leading frozen span up to this = head hold
DEAD_MIN = 2.5         # trailing frozen span at least this = dead air
CONV_HEAD = 2.0        # shoot-convention head trim (s) for clips > 6s
JUNK_DUR = 1.0
JUNK_VIS = 0.15        # junk if max(sharp_n * luma_ok) below this (invisible content)

def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                       capture_output=True, text=True)
    j = json.loads(r.stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    a = any(s["codec_type"] == "audio" for s in j["streams"])
    rot = 0
    for sd in v.get("side_data_list", []):
        if "rotation" in sd: rot = int(sd["rotation"]) % 360
    if not rot and "rotate" in v.get("tags", {}): rot = int(v["tags"]["rotate"]) % 360
    num, den = v["r_frame_rate"].split("/"); fps = float(num) / float(den)
    w, h = int(v["width"]), int(v["height"])
    if rot in (90, 270): w, h = h, w          # display dims after auto-rotate
    dur = float(v.get("duration") or j["format"]["duration"])
    return dict(w=w, h=h, fps=fps, dur=dur, rot=rot, au=int(a))

def analyze(path, meta, face_det=None, every_face=8):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or meta["fps"]
    sw = 160
    prev = None; hann = None; i = 0
    F = dict(t=[], lap=[], diff=[], dmag=[], dxy=[], resp=[], clip=[], luma=[], face=[], fps=fps)
    while True:
        ok, frame = cap.read()
        if not ok: break
        t = i / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g320 = cv2.resize(gray, (320, int(320 * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA)
        small = cv2.resize(gray, (sw, int(sw * gray.shape[0] / gray.shape[1])), interpolation=cv2.INTER_AREA).astype(np.float32)
        F["t"].append(t)
        F["lap"].append(float(cv2.Laplacian(g320, cv2.CV_32F).var()))
        hist = np.bincount(g320.reshape(-1) >> 3, minlength=32)
        F["clip"].append(float((hist[0] + hist[31]) / g320.size))
        F["luma"].append(float(g320.mean()))
        if prev is None:
            F["diff"].append(0.0); F["dmag"].append(0.0); F["dxy"].append((0.0, 0.0)); F["resp"].append(1.0)
        else:
            F["diff"].append(float(np.abs(small - prev).mean()))
            if hann is None: hann = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(prev, small, hann)
            sc = meta["w"] / sw
            F["dmag"].append(float(np.hypot(dx, dy) * sc)); F["dxy"].append((dx * sc, dy * sc))
            F["resp"].append(float(resp))
        if face_det is not None and i % every_face == 0:
            face_det.setInputSize((g320.shape[1], g320.shape[0]))
            _, faces = face_det.detect(cv2.resize(frame, (g320.shape[1], g320.shape[0])))
            F["face"].append((t, 0 if faces is None else len(faces)))
        prev = small; i += 1
    cap.release()
    return F

def frozen_threshold(F):
    """Noise-adaptive but hard-capped: p5 of frame diffs estimates the static noise floor
    when the clip HAS static frames; cap keeps all-motion clips from freezing themselves."""
    d = np.array(F["diff"][1:])
    if len(d) == 0: return FROZEN_ABS
    return min(3.0, max(FROZEN_ABS, 2.5 * float(np.percentile(d, 5))))

def find_cuts(F):
    d = np.array(F["diff"]); r = np.array(F["resp"]); t = np.array(F["t"]); cuts = []
    for i in range(2, len(d) - 1):
        sur = np.concatenate([d[max(0, i - 8):i - 1], d[i + 2:i + 10]])
        med = float(np.median(sur)) if len(sur) else 0.0
        if d[i] > CUT_ABS and d[i] > CUT_RATIO * max(med, 1.0) and r[i] < CUT_RESP:
            if not cuts or t[i] - cuts[-1] > 1.0: cuts.append(round(float(t[i]), 2))
    return cuts

def frozen_spans_frames(F, thr):
    d = np.array(F["diff"]); t = np.array(F["t"]); fz = d < thr
    fz[0] = fz[1] if len(fz) > 1 else fz[0]
    for i in range(1, len(fz) - 1):                       # close 1-frame gaps
        if not fz[i] and fz[i - 1] and fz[i + 1]: fz[i] = True
    spans, s = [], None
    for i, f in enumerate(fz):
        if f and s is None: s = t[i]
        if not f and s is not None:
            if t[i] - s >= 0.4: spans.append([round(s, 2), round(float(t[i]), 2)])
            s = None
    if s is not None and t[-1] + 1 / F["fps"] - s >= 0.4:
        spans.append([round(s, 2), round(float(t[-1] + 1 / F["fps"]), 2)])
    return spans

def windows(F, dur):
    nw = max(1, int(np.ceil(dur / WIN - 1e-6)))
    t = np.array(F["t"]); out = []
    lap, diff, dmag, clipf, luma = (np.array(F[k]) for k in ("lap", "diff", "dmag", "clip", "luma"))
    dxy = np.array(F["dxy"])
    for w in range(nw):
        m = (t >= w * WIN) & (t < (w + 1) * WIN)
        if not m.any(): continue
        idx = np.where(m)[0]
        acc = np.diff(dxy[idx], axis=0)
        jit = float(np.hypot(*acc.T).mean()) if len(acc) else 0.0
        out.append(dict(t0=round(w * WIN, 2), sharp=float(np.median(lap[idx])), diff=float(diff[idx].mean()),
                        motion=float(dmag[idx].mean()), jit=jit,
                        clip=float(clipf[idx].mean()), luma=float(luma[idx].mean())))
    return out

def qparts(w):
    sharp_n = float(np.clip(np.log10(w["sharp"] + 1) / np.log10(LAP_GOOD), 0, 1))
    expo_ok = max(0.05, float(np.clip(1 - 6 * max(0.0, w["clip"] - CLIP_TOL), 0, 1)))
    luma_ok = 1.0 if 35 <= w["luma"] <= 215 else max(0.15, 1 - abs(w["luma"] - 125) / 200)
    steady = max(0.04, 1.0 / (1.0 + (w["jit"] / J0) ** 2))
    return sharp_n, expo_ok, luma_ok, steady

def qscore(w):
    s, e, l, st = qparts(w); return s * e * l * st

def best_spans(ws, dur, cuts, hold, dead, d_target, topk=3):
    n = max(1, int(round(d_target / WIN))); q = [qscore(w) for w in ws]
    lo = hold[1] if hold else (CONV_HEAD if dur > 6 else 0.0)
    hi = dead[0] if dead else dur
    if hi - lo < d_target: lo, hi = (0.0, dur) if dur >= d_target else (0.0, dur)  # relax if clip too short
    cands = []
    for s in range(0, len(ws) - n + 1):
        t0, t1 = ws[s]["t0"], round(ws[s]["t0"] + d_target, 2)
        if t0 < lo - 1e-6 or t1 > hi + 1e-6: continue
        if any(t0 < c < t1 for c in cuts): continue
        seg = q[s:s + n]
        cands.append((round(float(np.mean(seg) - 0.25 * np.std(seg)), 3), t0, t1))
    cands.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for sc, t0, t1 in cands:
        if all(min(t1, o[2]) - max(t0, o[1]) < 0.5 * d_target for o in out): out.append((sc, t0, t1))
        if len(out) >= topk: break
    return out

def catalog_clip(path, face_det):
    t0 = time.time()
    meta = probe(path)
    F = analyze(path, meta, face_det)
    thr = frozen_threshold(F)
    ws = windows(F, meta["dur"])
    cuts = find_cuts(F)
    fro = frozen_spans_frames(F, thr)
    frozen_total = sum(e - s for s, e in fro)
    global_static = frozen_total >= 0.9 * meta["dur"]
    hold = dead = None
    if not global_static:
        for s in fro:
            if s[0] <= 0.35 and s[1] - s[0] <= HOLD_MAX: hold = s
            if (s[1] >= meta["dur"] - 0.35 and s[1] - s[0] >= DEAD_MIN
                    and not any(abs(s[0] - c) < 0.75 for c in cuts)):   # static shot after a cut != dead air
                dead = s
    vis = [qparts(w) for w in ws]
    junk = meta["dur"] < JUNK_DUR or max(s * l for s, _, l, _ in vis) < JUNK_VIS
    q = [qscore(w) for w in ws]
    usable = [i for i, w in enumerate(ws)
              if (not hold or w["t0"] >= hold[1] - 1e-6) and (not dead or w["t0"] + WIN <= dead[0] + 1e-6)]
    med = lambda k: (round(float(np.median([ws[i][k] for i in usable])), 3) if usable else 0)
    hero2 = best_spans(ws, meta["dur"], cuts, hold, dead, min(2.0, meta["dur"]), 1) if not junk else []
    entry = {"id": os.path.basename(path).split("_")[0], "f": os.path.basename(path),
             "d": round(meta["dur"], 1), "wh": [meta["w"], meta["h"]], "fps": round(meta["fps"], 1),
             "au": meta["au"], "cuts": cuts,
             "hold": [round(float(x), 1) for x in hold] if hold else None,
             "dead": [round(float(x), 1) for x in dead] if dead else None,
             "static": int(global_static), "junk": int(junk),
             "q": {"sharp": int(round(med("sharp") or 0)), "jit": round(med("jit") or 0, 2),
                   "clip": med("clip"), "luma": int(round(med("luma") or 0)), "mot": round(med("motion") or 0, 1)},
             "hero": [hero2[0][1], hero2[0][2], hero2[0][0]] if hero2 else None,
             "face": int(any(c > 0 for _, c in F["face"]))}
    if meta["rot"]: entry["rot"] = meta["rot"]
    dbg = {"windows": ws, "qs": [round(x, 3) for x in q], "cuts": cuts, "frozen_spans": fro,
           "frozen_thr": round(thr, 2),
           "diff_p": [round(float(np.percentile(F["diff"], p)), 2) for p in (10, 50, 95, 99)],
           "sec": round(time.time() - t0, 2)}
    return entry, dbg, F, ws, (cuts, hold, dead)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"); ap.add_argument("--out", default="catalog.json")
    ap.add_argument("--debug", default=None); ap.add_argument("--faces", default="yunet")
    a = ap.parse_args()
    face_det = None
    if a.faces == "yunet":
        mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets/yunet.onnx")
        face_det = cv2.FaceDetectorYN.create(mp, "", (320, 320), 0.6)
    clips = sorted(f for f in os.listdir(a.dir) if f.lower().endswith((".mp4", ".mov")))
    cat, dbg = [], {}
    for c in clips:
        e, d, _, _, _ = catalog_clip(os.path.join(a.dir, c), face_det)
        cat.append(e); dbg[c] = d
        print(f"{c}: {d['sec']}s thr={d['frozen_thr']} junk={e['junk']} static={e['static']} "
              f"hold={e['hold']} dead={e['dead']} cuts={e['cuts']} hero={e['hero']} face={e['face']}")
    with open(a.out, "w") as f: json.dump(cat, f, separators=(",", ":"))
    if a.debug:
        with open(a.debug, "w") as f: json.dump(dbg, f, indent=1)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
