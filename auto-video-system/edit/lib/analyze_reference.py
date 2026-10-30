#!/usr/bin/env python3
"""
analyze_reference.py — Reference Blueprint Extractor v1 (edit engine, research A).

Turns a reference vertical reel (incl. iPhone screen recordings of Instagram reels)
into a machine-readable edit blueprint: cuts, per-shot features, audio structure,
caption events, loop point — plus 3 model-readable artifacts (contact sheet,
per-shot sheet, caption strip).

Empirically validated on ref13.mp4 (18.38s IG screen rec, HEVC 1320x2868, ~60fps):
  - cuts: ffmpeg scdet (score capture via metadata=print) on chrome-cropped frames,
    threshold 10, collapse <0.25s keeping argmax score  -> 18/19 vs hand ground truth, 0 FA.
    NOTE: scdet logs at loglevel INFO on stderr only; with -v error it prints NOTHING —
    that is why a previous production session concluded "scdet found nothing".
  - whisper base.en word timestamps: ~97% word accuracy; cuts align to VO phrase starts
    (0-55ms), NOT to music beats (cut->beat distance ~= chance).
  - loop: pixel window match + audio envelope xcorr must agree within 0.35s.

Usage:
  python3 analyze_reference.py ref.mp4 [--workdir DIR] [--no-ocr]
      [--chrome-top 0.12 --chrome-bottom 0.20 --chrome-right 0.10 | --no-chrome-crop]
      [--band-x0 0.05 --band-x1 0.85 --band-y0 0.36 --band-y1 0.56]
      [--scdet-thr 10] [--gap 0.25] [--whisper-model base.en]
Paths must be space-free (lavfi/metadata file escaping).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import cv2

T_START = time.time()
FONT = cv2.FONT_HERSHEY_SIMPLEX


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd[:6])}...\n{r.stderr[-800:]}")
    return r


def probe(path):
    r = sh(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,nb_frames,duration,avg_frame_rate,side_data_list:stream_side_data=rotation",
            "-show_entries", "format=duration", "-of", "json", path])
    j = json.loads(r.stdout)
    st = j["streams"][0]
    num, den = st["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    dur = float(j["format"]["duration"])
    rot = 0
    for sd in st.get("side_data_list", []):
        rot = int(sd.get("rotation", 0)) or rot
    has_audio = bool(json.loads(sh(["ffprobe", "-v", "error", "-select_streams", "a",
                     "-show_entries", "stream=index", "-of", "json", path]).stdout).get("streams"))
    return int(st["width"]), int(st["height"]), fps, dur, rot, has_audio, st["avg_frame_rate"]


def collapse(times, scores, gap):
    """Merge detections closer than `gap` s; keep argmax-score representative."""
    if not len(times):
        return [], []
    z = sorted(zip(times, scores))
    out, grp = [], [z[0]]
    for t, s in z[1:]:
        if t - grp[-1][0] < gap:
            grp.append((t, s))
        else:
            out.append(max(grp, key=lambda x: x[1])); grp = [(t, s)]
    out.append(max(grp, key=lambda x: x[1]))
    return [round(t, 3) for t, _ in out], [round(s, 2) for _, s in out]


def label_bar(w, h, text, scale=0.55):
    bar = np.full((h, w, 3), 20, np.uint8)
    cv2.putText(bar, text, (6, h - 8), FONT, scale, (0, 255, 255), 1, cv2.LINE_AA)
    return bar


def tile_grid(tiles, cols):
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--no-chrome-crop", action="store_true",
                    help="input is clean footage, not an IG screen recording")
    ap.add_argument("--chrome-top", type=float, default=0.12)
    ap.add_argument("--chrome-bottom", type=float, default=0.20)
    ap.add_argument("--chrome-right", type=float, default=0.10)
    ap.add_argument("--band-x0", type=float, default=0.05)
    ap.add_argument("--band-x1", type=float, default=0.85)
    ap.add_argument("--band-y0", type=float, default=0.36)
    ap.add_argument("--band-y1", type=float, default=0.56)
    ap.add_argument("--scdet-thr", type=float, default=10.0)
    ap.add_argument("--gap", type=float, default=0.25)
    ap.add_argument("--whisper-model", default="base.en")
    ap.add_argument("--no-transitions", action="store_true",
                    help="skip transition classification (blueprint gets no 'transitions' block)")
    ap.add_argument("--merge-dissolves", action="store_true",
                    help="add DISCOVERED dissolve midpoints to cuts.times (default: report only)")
    ap.add_argument("--nat-width", type=int, default=96)
    a = ap.parse_args()

    vid = os.path.abspath(a.input)
    wd = os.path.abspath(a.workdir or os.path.dirname(vid) or ".")
    os.makedirs(wd, exist_ok=True)
    assert " " not in vid + wd, "space-free paths required"

    W, H, fps, dur, rot, has_audio, vrate = probe(vid)
    print(f"[probe] {W}x{H} fps={fps:.2f} dur={dur:.2f}s rotation={rot} audio={has_audio}", flush=True)
    if rot:
        print(f"[warn] rotation metadata {rot} deg present — ffmpeg auto-rotates; W/H above are pre-rotation")

    # ---------------- single video decode pass: 4 branches ----------------
    if a.no_chrome_crop:
        cx0, cy0, cw_, ch_ = 0, 0, W, H
    else:
        cx0, cy0 = 0, int(a.chrome_top * H)
        cw_, ch_ = int((1 - a.chrome_right) * W), int((1 - a.chrome_top - a.chrome_bottom) * H)
    bx0, by0 = int(a.band_x0 * W), int(a.band_y0 * H)
    bw_, bh_ = int((a.band_x1 - a.band_x0) * W), int((a.band_y1 - a.band_y0) * H)
    BW = 476; BH = max(int(bh_ / bw_ * BW) // 2 * 2, 2)
    FW = 240; FH = max(int(ch_ / cw_ * FW) // 2 * 2, 2)
    LW = 48; LH = max(int(H / W * LW) // 2 * 2, 2)
    NW = int(a.nat_width); NH = max(int(ch_ / cw_ * NW) // 2 * 2, 2)
    scdet_txt = f"{wd}/scdet_scores.txt"
    band_raw, feat_raw, loop_raw, bandhi_raw = (f"{wd}/_band.raw", f"{wd}/_feat.raw",
                                                f"{wd}/_loop.raw", f"{wd}/_bandhi.raw")
    nat_raw = f"{wd}/_nat.raw"
    t0 = time.time()
    sh(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", vid, "-filter_complex",
        f"[0:v]split={5 if a.no_transitions else 6}[a][b][c][d][e]{'' if a.no_transitions else '[f]'};"
        f"[a]crop={cw_}:{ch_}:{cx0}:{cy0},scdet=t=1,metadata=mode=print:key=lavfi.scd.score:file={scdet_txt}[sc];"
        f"[b]fps=12,crop={cw_}:{ch_}:{cx0}:{cy0},scale={FW}:{FH}[feat];"
        f"[c]fps=5,crop={bw_}:{bh_}:{bx0}:{by0},scale={BW}:{BH},format=gray[band];"
        f"[d]fps=20,scale={LW}:{LH},format=gray[loop];"
        f"[e]fps=2,crop={bw_}:{bh_}:{bx0}:{by0}[bandhi]"
        + ("" if a.no_transitions else
           # native-fps small BGR for transition features. The fps= is REQUIRED:
           # without it ffmpeg emits rawvideo at r_frame_rate (120 for a 60fps
           # iPhone recording) and silently duplicates every frame.
           f";[f]fps={vrate},crop={cw_}:{ch_}:{cx0}:{cy0},scale={NW}:{NH}[nat]"),
        "-map", "[sc]", "-f", "null", "-",
        "-map", "[feat]", "-f", "rawvideo", "-pix_fmt", "bgr24", feat_raw,
        "-map", "[band]", "-f", "rawvideo", "-pix_fmt", "gray", band_raw,
        "-map", "[loop]", "-f", "rawvideo", "-pix_fmt", "gray", loop_raw,
        "-map", "[bandhi]", "-f", "rawvideo", "-pix_fmt", "bgr24", bandhi_raw]
       + ([] if a.no_transitions else
          ["-map", "[nat]", "-f", "rawvideo", "-pix_fmt", "bgr24", nat_raw]))
    feat = np.fromfile(feat_raw, np.uint8)
    feat = feat[:len(feat) // (FW * FH * 3) * (FW * FH * 3)].reshape(-1, FH, FW, 3)
    band = np.fromfile(band_raw, np.uint8)
    band = band[:len(band) // (BW * BH) * (BW * BH)].reshape(-1, BH, BW).astype(np.float32)
    loopf = np.fromfile(loop_raw, np.uint8)
    loopf = loopf[:len(loopf) // (LW * LH) * (LW * LH)].reshape(-1, LH, LW).astype(np.float32)
    bandhi = np.fromfile(bandhi_raw, np.uint8)
    bandhi = bandhi[:len(bandhi) // (bw_ * bh_ * 3) * (bw_ * bh_ * 3)].reshape(-1, bh_, bw_, 3)
    print(f"[decode] 1 pass, {time.time()-t0:.1f}s  feat={feat.shape} band={band.shape} "
          f"loop={loopf.shape} bandhi={bandhi.shape}", flush=True)

    # ---------------- cuts from scdet scores ----------------
    st_, ss_ = [], []
    cur_t = None
    for line in open(scdet_txt):
        if line.startswith("frame:"):
            cur_t = float(line.split("pts_time:")[1])
        elif "lavfi.scd.score=" in line and cur_t is not None:
            st_.append(cur_t); ss_.append(float(line.strip().split("=")[1]))
    st_, ss_ = np.array(st_), np.array(ss_)
    idx = np.where(ss_ > a.scdet_thr)[0]
    cuts, cut_scores = collapse(st_[idx], ss_[idx], a.gap)
    # secondary signal: 12fps mean-abs-diff on decimated content frames (cross-check only)
    gfeat = feat.mean(axis=3).astype(np.float32)
    mdiff = np.abs(np.diff(gfeat, axis=0)).mean(axis=(1, 2))
    print(f"[cuts] scdet>{a.scdet_thr} -> {len(cuts)} cuts: {cuts}", flush=True)

    bounds = [0.0] + list(cuts) + [round(dur, 3)]
    bounds = [b for i, b in enumerate(bounds) if i == 0 or b - bounds[i - 1] > 0.05]

    # ---------------- transitions (research/f) ----------------
    trans = None
    moves = []
    if not a.no_transitions:
        t0 = time.time()
        import transitions as TR
        nat = np.fromfile(nat_raw, np.uint8)
        nfsz = NW * NH * 3
        nat = nat[:len(nat) // nfsz * nfsz].reshape(-1, NH, NW, 3)
        ft = TR.frame_feats(nat)
        boundaries = TR.classify_boundaries(cuts, ft, fps, sc=(st_, ss_))
        discovered = TR.discover_dissolves(ft, fps, cuts)
        moves = TR.shot_moves(gfeat, bounds, 12.0, ft_native=ft, fps_native=fps)
        if a.merge_dissolves and discovered:
            for d_ in discovered:
                cuts.append(d_["t"]); cut_scores.append(0.0)
                boundaries.append(d_)
            order = sorted(range(len(cuts)), key=lambda i: cuts[i])
            cuts = [cuts[i] for i in order]; cut_scores = [cut_scores[i] for i in order]
            boundaries = sorted(boundaries, key=lambda b: b["t"])
            bounds = [0.0] + list(cuts) + [round(dur, 3)]
            bounds = [b for i, b in enumerate(bounds) if i == 0 or b - bounds[i - 1] > 0.05]
            moves = TR.shot_moves(gfeat, bounds, 12.0, ft_native=ft, fps_native=fps)
        from collections import Counter
        trans = dict(
            version=TR.VERSION, params={k: v for k, v in TR.CFG.items()},
            conf_min=TR.CFG["conf_min"], nat_size=[NW, NH], native_fps=round(fps, 3),
            counts=dict(Counter(b["type"] for b in boundaries)),
            boundaries=boundaries,
            discovered_dissolves=([] if a.merge_dissolves else discovered),
            merged_dissolves=bool(a.merge_dissolves and discovered),
            wall_s=round(time.time() - t0, 2),
            notes=("scdet does NOT plateau across a dissolve (measured): dissolves are "
                   "found on the native-fps frame-diff signal, and may never appear in "
                   "cuts.times at all -- see discovered_dissolves. whip is reported for "
                   "style only; the renderer has no whip primitive."))
        os.path.exists(nat_raw) and os.remove(nat_raw)
        print(f"[transitions] {time.time()-t0:.1f}s {trans['counts']}", flush=True)

    # ---------------- audio ----------------
    audio = dict(has_vo=False, transcript=None, segments=[], words=[],
                 music=dict(bpm=None, beat_times=[]), lufs=None, layers=[])
    words, segs, beats = [], [], np.array([])
    if has_audio:
        wav = f"{wd}/_audio16k.wav"
        sh(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", vid, "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le", wav])
        r = subprocess.run(["ffmpeg", "-nostdin", "-i", vid, "-af", "ebur128", "-f", "null", "-"],
                           capture_output=True, text=True)
        for i, line in enumerate(ls := r.stderr.splitlines()):
            if "Integrated loudness" in line and i + 1 < len(ls):
                audio["lufs"] = float(ls[i + 1].split("I:")[1].split("LUFS")[0])
        t0 = time.time()
        from faster_whisper import WhisperModel
        model = WhisperModel(a.whisper_model, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(wav, word_timestamps=True, vad_filter=True,
                                       vad_parameters=dict(min_silence_duration_ms=300))
        for s in segments:
            segs.append(dict(t0=round(s.start, 3), t1=round(s.end, 3), text=s.text.strip()))
            for w in (s.words or []):
                words.append(dict(t0=round(w.start, 3), t1=round(w.end, 3),
                                  w=w.word.strip(), p=round(w.probability, 2)))
        audio.update(has_vo=bool(words), transcript=" ".join(s["text"] for s in segs),
                     segments=segs, words=words)
        print(f"[whisper] {time.time()-t0:.1f}s  {len(words)} words", flush=True)
        t0 = time.time()
        import librosa
        y, sr = librosa.load(wav, sr=16000, mono=True)
        _, y_p = librosa.effects.hpss(y)
        tempo, beats = librosa.beat.beat_track(y=y_p, sr=sr, units="time")
        beats = np.asarray(beats)
        audio["music"] = dict(bpm=round(float(np.atleast_1d(tempo)[0]), 1),
                              beat_times=[round(float(b), 3) for b in beats])
        rms = librosa.feature.rms(y=y, frame_length=1600, hop_length=1600)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        speech = np.zeros(len(rms_db), bool)
        for s in segs:
            speech[int(s["t0"] / 0.1):min(len(rms_db), int(np.ceil(s["t1"] / 0.1)))] = True
        lab = np.where(rms_db < -45, "silence", np.where(speech, "both", "music"))
        ranges, cur, start = [], lab[0], 0.0
        for i in range(1, len(lab)):
            if lab[i] != cur:
                ranges.append(dict(t0=round(start, 1), t1=round(i * 0.1, 1), layer=str(cur)))
                cur, start = lab[i], i * 0.1
        ranges.append(dict(t0=round(start, 1), t1=round(dur, 2), layer=str(cur)))
        audio["layers"] = [r_ for r_ in ranges if r_["t1"] - r_["t0"] >= 0.3]
        print(f"[librosa] {time.time()-t0:.1f}s bpm={audio['music']['bpm']}", flush=True)
        # alignment diagnostics
        if len(beats) > 1 and cuts:
            d_beat = [float(np.min(np.abs(beats - c))) for c in cuts]
            ibi = float(np.median(np.diff(beats)))
            audio["cut_beat_align"] = dict(mean_s=round(float(np.mean(d_beat)), 3),
                                           random_baseline_s=round(ibi / 4, 3),
                                           frac_within_80ms=round(float(np.mean(np.array(d_beat) < .08)), 2))
        if words and cuts:
            ws = np.array([w["t0"] for w in words])
            d_word = [float(np.min(np.abs(ws - c))) for c in cuts]
            audio["cut_word_align"] = dict(mean_s=round(float(np.mean(d_word)), 3),
                                           frac_within_80ms=round(float(np.mean(np.array(d_word) < .08)), 2))

    # ---------------- caption events + states ----------------
    mask = band >= 205
    cnt = mask.sum(axis=(1, 2))
    changes = []
    for i in range(1, len(mask)):
        xor = (mask[i] ^ mask[i - 1]).sum()
        union = max((mask[i] | mask[i - 1]).sum(), 1)
        if xor / union > 0.45 and (cnt[i] > 60 or cnt[i - 1] > 60):
            changes.append((i * 0.2, xor / union))
    ev, _ = collapse([c[0] for c in changes], [c[1] for c in changes], 0.3)
    sbounds = [0.0] + list(ev) + [len(mask) * 0.2]
    states = []
    for s0, s1 in zip(sbounds[:-1], sbounds[1:]):
        if s1 - s0 < 0.25:
            continue
        i0, i1 = int(s0 / 0.2) + 1, max(int(s1 / 0.2), int(s0 / 0.2) + 2)
        seg = cnt[i0:i1]
        states.append(dict(t0=round(s0, 2), t1=round(s1, 2),
                           has_text=bool(len(seg) and (seg > 60).mean() >= 0.5),
                           text_px=int(np.median(seg)) if len(seg) else 0, ocr=None))
    # OCR via temporal-stability mask (burned text is static while bg moves)
    if not a.no_ocr:
        t0 = time.time()
        for s in states:
            if not s["has_text"]:
                continue
            i0, i1 = int(s["t0"] / 0.2) + 1, max(int(s["t1"] / 0.2), int(s["t0"] / 0.2) + 2)
            fr = band[i0:i1]
            med = np.median(fr, axis=0)
            std = fr.std(axis=0) if len(fr) > 1 else np.zeros_like(med)
            m = ((med >= 195) & (std <= 14)).astype(np.uint8) if len(fr) > 1 else (med >= 205).astype(np.uint8)
            n, labm, stats, _ = cv2.connectedComponentsWithStats(m, 8)
            keep = np.zeros_like(m)
            for i in range(1, n):
                _, _, w_, h_, area = stats[i]
                if 15 <= area <= 20000 and 5 <= h_ <= 100:
                    keep[labm == i] = 1
            if keep.sum() < 30:
                continue
            canvas = 255 - keep * 255
            big = cv2.medianBlur(cv2.resize(canvas, (BW * 3, BH * 3), interpolation=cv2.INTER_CUBIC), 3)
            tmp = f"{wd}/_ocr.png"; cv2.imwrite(tmp, big)
            r = subprocess.run(["tesseract", tmp, "stdout", "--psm", "6", "-l", "eng"],
                               capture_output=True, text=True)
            txt = " ".join(t for t in r.stdout.split() if any(ch.isalnum() for ch in t))
            s["ocr"] = txt or None
        print(f"[ocr] {time.time()-t0:.1f}s", flush=True)

    # caption style summary from masks
    text_states = [s for s in states if s["has_text"]]
    ys, hs = [], []
    for s in text_states:
        i = min(int((s["t0"] + s["t1"]) / 2 / 0.2), len(mask) - 1)
        pts = np.argwhere(mask[i])
        if len(pts) > 20:
            ys.append(pts[:, 0].mean() / BH); hs.append((pts[:, 0].max() - pts[:, 0].min()) / BH)
    cap_style = (f"white bold text, ~{len(text_states)} states over {dur:.0f}s "
                 f"(median state {np.median([s['t1']-s['t0'] for s in text_states]):.2f}s), "
                 f"band y {a.band_y0:.2f}-{a.band_y1:.2f} of frame, "
                 f"y-center ~{a.band_y0 + (np.median(ys) if ys else .5)*(a.band_y1-a.band_y0):.2f}H"
                 ) if text_states else "no burned captions detected"

    # ---------------- per-shot features ----------------
    yn = None
    ypath = f"{os.path.dirname(os.path.abspath(__file__))}/yunet.onnx"
    if os.path.exists(ypath):
        yn = cv2.FaceDetectorYN.create(ypath, "", (0, 0), score_threshold=0.6)
    shots = []
    for si, (b0, b1) in enumerate(zip(bounds[:-1], bounds[1:])):
        i0, i1 = int(b0 * 12) + 1, max(int(b1 * 12) - 1, int(b0 * 12) + 3)
        g = gfeat[i0:min(i1, len(gfeat))]
        if len(g) < 2:
            g = gfeat[max(i0 - 1, 0):i0 + 2]
        d = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2)) if len(g) > 1 else np.array([0.])
        shifts = [float(np.hypot(*cv2.phaseCorrelate(g[j], g[j + 1])[0]))
                  for j in range(0, len(g) - 1, 3)] or [0.]
        motion, med_shift = float(d.mean()), float(np.median(shifts))
        static = motion < 6.0 and med_shift < 0.6
        face_n, face_h = 0, 0.0
        if yn is not None:
            for frac in (0.45, 0.75):
                fi = min(int((b0 + (b1 - b0) * frac) * 12), len(feat) - 1)
                img = cv2.resize(feat[fi], (FW * 3, FH * 3))
                yn.setInputSize((img.shape[1], img.shape[0]))
                _, fcs = yn.detect(img)
                if fcs is not None:
                    face_n = max(face_n, len(fcs))
                    face_h = max(face_h, float(max(f[3] for f in fcs)) / img.shape[0])
        scale = ("close" if face_h > 0.16 else "medium" if face_h > 0.07 else
                 "wide" if face_n else "no-face")
        layer = "none"
        if audio["layers"]:
            ov = {}
            for r_ in audio["layers"]:
                o = max(0, min(b1, r_["t1"]) - max(b0, r_["t0"]))
                ov[r_["layer"]] = ov.get(r_["layer"], 0) + o
            layer = max(ov, key=ov.get) if ov else "none"
        wtxt = " ".join(w["w"] for w in words if b0 <= w["t0"] < b1) or None
        mv = moves[si] if si < len(moves) else None
        punch, speed, mv_ev = 1.0, 1.0, None
        if mv:
            gate = (mv["punch"] >= 1.045 and mv["punch_r2"] >= 0.75
                    and mv["punch_ncc"] >= 0.55 and mv.get("punch_monotone"))
            punch = round(min(1.20, mv["punch"]), 3) if gate else 1.0
            speed = float(mv["slow"]) if mv.get("slow") else 1.0
            mv_ev = dict(punch_raw=mv["punch"], r2=mv["punch_r2"], ncc=mv["punch_ncc"],
                         dup_frac=mv["dup_frac"], global_dup=mv.get("global_dup"))
        shots.append(dict(i=si, t0=round(b0, 3), t1=round(b1, 3), dur=round(b1 - b0, 3),
                          punch=punch, speed=speed, move_evidence=mv_ev,
                          motion=round(motion, 1), luma=round(float(g.mean()), 1),
                          static=bool(static), face=dict(n=int(face_n), h_frac=round(face_h, 3)),
                          scale_guess=scale, audio_layer=layer, text=wtxt,
                          cut_score_in=(cut_scores[si - 1] if 0 < si <= len(cut_scores) else None)))
    print(f"[shots] {len(shots)}", flush=True)

    # ---------------- loop point ----------------
    loop_s = None; loop_info = {}
    lf = loopf[:, int(0.06 * LH):int(0.80 * LH), :]
    early = lf[1:9]  # t 0.05-0.45s
    if len(lf) > 20 * 5:
        lags = range(int(len(lf) * 0.55), len(lf) - len(early))
        dists = [(l / 20.0, float(np.abs(lf[l:l + len(early)] - early).mean())) for l in lags]
        basel = [float(np.abs(lf[l:l + len(early)] - early).mean())
                 for l in range(20, len(lf) - len(early), 7)]
        pix_lag, pix_d = min(dists, key=lambda x: x[1])
        pix_ok = pix_d < 0.6 * float(np.median(basel))
        aud_lag, aud_c, aud_ok = None, None, False
        if has_audio:
            import soundfile as sf
            yy, _ = sf.read(f"{wd}/_audio16k.wav"); yy = yy.astype(np.float32)
            n = len(yy) // 160 * 160
            env = np.abs(yy[:n]).reshape(-1, 160).mean(axis=1)
            e0 = env[5:100]
            cors = [(l / 100.0, float(np.corrcoef(e0, env[l:l + len(e0)])[0, 1]))
                    for l in range(int(len(env) * 0.55), len(env) - len(e0))]
            aud_lag, aud_c = max(cors, key=lambda x: x[1])
            aud_ok = aud_c > 0.7
        agree = pix_ok and aud_ok and aud_lag is not None and abs(pix_lag - aud_lag) <= 0.35
        if agree:
            loop_s = round(pix_lag - 0.05, 3)  # early window started at t=0.05
        loop_info = dict(pixel=dict(lag=round(pix_lag, 2), dist=round(pix_d, 1),
                                    baseline_median=round(float(np.median(basel)), 1), ok=bool(pix_ok)),
                         audio=dict(lag=aud_lag and round(aud_lag, 2), corr=aud_c and round(aud_c, 2),
                                    ok=bool(aud_ok)), agree=bool(agree))
    print(f"[loop] {loop_info}", flush=True)

    # ---------------- artifacts ----------------
    # contact sheet (4fps from feat frames)
    tiles = []
    for i in range(0, len(feat), 3):
        t = cv2.resize(feat[i], (180, int(FH / FW * 180)))
        cv2.putText(t, f"{i/12:.2f}", (4, 22), FONT, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        tiles.append(t)
    cv2.imwrite(f"{wd}/contact_sheet.png", tile_grid(tiles, 6))
    # per-shot sheet (frame at 45% in)
    tiles = []
    for s in shots:
        fi = min(int((s["t0"] + s["dur"] * 0.45) * 12), len(feat) - 1)
        t = feat[fi].copy()
        lb = label_bar(FW, 26, f"S{s['i']:02d} {s['t0']:.2f}-{s['t1']:.2f} ({s['dur']:.2f}s)", 0.42)
        tiles.append(np.vstack([t, lb]))
    cv2.imwrite(f"{wd}/shot_sheet.png", tile_grid(tiles, 5))
    # caption strip: unique states, full-res band crop scaled to 530w, 2 columns
    tiles = []
    for s in states:
        if not s["has_text"]:
            continue
        hi = bandhi[min(int((s["t0"] + s["t1"]) / 2 * 2), len(bandhi) - 1)]
        t = cv2.resize(hi, (530, int(bh_ / bw_ * 530)))
        lb = label_bar(530, 26, f"{s['t0']:.1f}-{s['t1']:.1f}s", 0.5)
        tiles.append(np.vstack([t, lb]))
    if tiles:
        cv2.imwrite(f"{wd}/caption_strip.png", tile_grid(tiles, 2))

    # ---------------- blueprint ----------------
    bp = dict(
        version="blueprint.v1",
        source=dict(path=vid, analyzed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        global_=dict(duration_s=round(dur, 3), size=[W, H], effective_fps=round(fps, 2),
                     rotation=rot, loop_s=loop_s, loop_detail=loop_info,
                     chrome_crop=(None if a.no_chrome_crop else
                                  dict(top=a.chrome_top, bottom=a.chrome_bottom, right=a.chrome_right))),
        audio=audio,
        cuts=dict(method="ffmpeg-scdet-on-chrome-cropped, per-frame scores via metadata=print, "
                         "threshold then collapse argmax",
                  params=dict(threshold=a.scdet_thr, collapse_gap_s=a.gap),
                  times=cuts, scores=cut_scores),
        shots=shots,
        transitions=trans,
        captions=dict(band=dict(x0=a.band_x0, x1=a.band_x1, y0=a.band_y0, y1=a.band_y1),
                      style_summary=cap_style,
                      events=[round(e, 2) for e in ev], states=states),
        artifacts=dict(contact_sheet=f"{wd}/contact_sheet.png", shot_sheet=f"{wd}/shot_sheet.png",
                       caption_strip=f"{wd}/caption_strip.png" if tiles else None),
        wall_time_s=round(time.time() - T_START, 1))
    bp["global"] = bp.pop("global_")
    out = f"{wd}/reference_blueprint.json"
    json.dump(bp, open(out, "w"), indent=1)
    for f in (band_raw, feat_raw, loop_raw, bandhi_raw):
        os.path.exists(f) and os.remove(f)
    print(f"[done] {out}  wall={bp['wall_time_s']}s", flush=True)


if __name__ == "__main__":
    main()
