#!/usr/bin/env python3
"""transitions.py — transition classification for the edit engine's analyzer.

numpy/opencv, ANALYZER SIDE ONLY (render_edit.py stays python-stdlib + ffmpeg).

Signals (all derived from streams analyze_reference.py already decodes):
  nat  : native-fps, chrome-cropped, ~96px-wide BGR  -> luma / sat / sharpness /
         frame-diff / phase-correlation shift, per frame
  feat : the existing 12fps 240px BGR stream         -> per-shot punch (NCC scale search)
  scdet: per-frame scene scores                      -> evidence only (see note)

NOTE (measured, research/f): ffmpeg scdet does NOT plateau during a dissolve —
its score stays at the noise floor (peak 0.05-3.2 vs 18-46 for a hard cut), so a
dissolve is normally never even reported as a cut. The run-vs-spike distinction
therefore lives on the mean-abs frame-diff signal, not on scdet.

Classes: cut | dissolve | flash | whip   (+ per-shot punch / slow-mo hints)
"""
import numpy as np
import cv2

VERSION = "transitions.v1"

CFG = dict(
    # --- windows
    peak_win_s=0.20,        # search for the splice/peak around the reported cut
    base_win_s=1.20,        # local baseline window half-width
    base_excl_s=0.50,       # excluded core inside it
    # --- run / plateau on the frame-diff signal
    run_ratio=2.5,          # diff > max(run_ratio*base, base+run_floor) is "elevated"
    run_floor=0.30,
    min_run=3,              # >=3 elevated frames before a dissolve is considered
    max_run_s=1.40,
    # --- dissolve (linear A->B mixing test)
    mix_res_max=0.35,       # residual for a CONFIDENT dissolve (conf scale ref)
    mix_res_soft=0.55,      # residual still reported, but with conf < conf_min
    mix_span_min=0.45,      # alphas must span this much of [0,1]
    mix_mono_min=0.70,      # fraction of non-decreasing alpha steps
    sharp_dip_max=0.92,     # sharpness inside the run vs its brackets
    # --- flash
    flash_luma_d=25.0,      # luma above local baseline (0-255) to count as "white"
    flash_sat_max=0.40,     # saturation at the peak vs baseline
    flash_luma_abs=200.0,   # ABSOLUTE peak luma (0-255): a real white flash is
                            # near-white. Without this, a cut into a bright
                            # blurred floor read as a flash on ref13 (t=10.83).
    flash_max_s=0.50,       # a longer bright stretch is a bright SHOT, not a flash
    # --- whip
    whip_sharp_drop=0.60,   # sharpness vs that side's own shot baseline
    whip_min_side=2,        # blurred frames required on each side
    # --- punch
    punch_min=1.045,
    punch_r2=0.75,
    # --- slow motion (duplicate-frame cadence)
    dup_eps=0.10,
    dup_lo=0.35, dup_hi=0.80, dup_margin=0.25,
    conf_min=0.60,          # planner acts only at/above this
)


# --------------------------------------------------------------- inputs
def load_scdet(path):
    t, s, cur = [], [], None
    for line in open(path):
        if line.startswith("frame:"):
            cur = float(line.split("pts_time:")[1])
        elif "lavfi.scd.score=" in line and cur is not None:
            t.append(cur); s.append(float(line.strip().split("=")[1]))
    return np.asarray(t, np.float64), np.asarray(s, np.float32)


def frame_feats(nat, want_shift=True):
    """nat: (N,H,W,3) uint8 BGR at native fps (CFR). Per-frame features.

    Kept on uint8 + per-frame cv2 ops: the naive float32 version cost 5.0s on
    ref13 (1101 frames), this costs ~0.6s. gray stays uint8; only the few frames
    inside a candidate run are promoted to float for the mixing fit.
    """
    n = len(nat)
    gray = np.empty((n,) + nat.shape[1:3], np.uint8)
    luma = np.empty(n, np.float32); sat = np.empty(n, np.float32)
    sharp = np.empty(n, np.float32); d = np.zeros(n, np.float32)
    prev = None
    for i in range(n):
        fr = nat[i]
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        gray[i] = g
        b, gg, r = fr[:, :, 0], fr[:, :, 1], fr[:, :, 2]
        mx = cv2.max(cv2.max(b, gg), r); mn = cv2.min(cv2.min(b, gg), r)
        luma[i] = cv2.mean(g)[0]
        sat[i] = cv2.mean(mx)[0] - cv2.mean(mn)[0]
        sharp[i] = float(cv2.Laplacian(g, cv2.CV_32F).var())
        if prev is not None:
            d[i] = cv2.mean(cv2.absdiff(g, prev))[0]
        prev = g
    shift = np.zeros(n, np.float32)
    if want_shift and n > 1:
        win = cv2.createHanningWindow((gray.shape[2], gray.shape[1]), cv2.CV_32F)
        pf = gray[0].astype(np.float32)
        for i in range(1, n):
            cf = gray[i].astype(np.float32)
            (dx, dy), _ = cv2.phaseCorrelate(pf, cf, win)
            shift[i] = float(np.hypot(dx, dy)) / gray.shape[2]
            pf = cf
    return dict(gray=gray, luma=luma, sat=sat, sharp=sharp, diff=d, shift=shift)


# --------------------------------------------------------------- helpers
def _med(a, dflt=0.0):
    return float(np.median(a)) if len(a) else dflt


def _baseline_idx(p, n, fps, c):
    bw = int(round(c["base_win_s"] * fps)); ew = int(round(c["base_excl_s"] * fps))
    left = np.arange(max(1, p - bw), max(1, p - ew))
    right = np.arange(min(n, p + ew), min(n, p + bw))
    return np.r_[left, right].astype(int)


def _mix_fit(gray, i0, i1):
    a_idx, b_idx = max(i0 - 1, 0), min(i1 + 1, len(gray) - 1)
    A = gray[a_idx].astype(np.float32); B = gray[b_idx].astype(np.float32)
    D = (B - A).ravel()
    den = float(D @ D)
    if den < 1e-3:
        return None
    nrm = np.sqrt(den)
    al, res = [], []
    for i in range(i0, i1 + 1):
        F = (gray[i].astype(np.float32) - A).ravel()
        a = float(F @ D / den)
        al.append(a)
        res.append(float(np.linalg.norm(F - a * D)) / nrm)
    al = np.array(al)
    mono = float(np.mean(np.diff(al) > -0.03)) if len(al) > 1 else 0.0
    return dict(res=float(np.mean(res)), span=float(al.max() - al.min()), mono=mono)


# --------------------------------------------------------------- boundaries
def classify_boundaries(cuts, ft, fps, sc=None, cfg=None):
    """cuts: boundary times (s). sc: (times, scores) from scdet, evidence only."""
    c = dict(CFG); c.update(cfg or {})
    diff, luma, sat, sharp, shiftv, gray = (ft["diff"], ft["luma"], ft["sat"],
                                            ft["sharp"], ft["shift"], ft["gray"])
    n = len(diff)
    out = []
    for tc in cuts:
        ci = int(np.clip(round(tc * fps), 1, n - 2))
        pw = max(1, int(round(c["peak_win_s"] * fps)))
        maxrun = int(round(c["max_run_s"] * fps))
        sel = _baseline_idx(ci, n, fps, c)
        b_diff = _med(diff[sel]); b_luma = _med(luma[sel], float(luma.mean()))
        b_sat = _med(sat[sel], float(sat.mean()))
        elev = max(c["run_ratio"] * b_diff, b_diff + c["run_floor"])
        # elevated runs near the boundary; pick the longest one within pw of ci
        w0, w1 = max(1, ci - pw - maxrun), min(n, ci + pw + maxrun + 1)
        mask = diff[w0:w1] > elev
        runs, j = [], 0
        while j < len(mask):
            if mask[j]:
                j2 = j
                while j2 + 1 < len(mask) and mask[j2 + 1]:
                    j2 += 1
                runs.append((w0 + j, w0 + j2))
                j = j2 + 1
            else:
                j += 1
        near = [r for r in runs if r[0] - pw <= ci <= r[1] + pw and r[1] - r[0] + 1 <= maxrun]
        if near:
            i0, i1 = max(near, key=lambda r: (r[1] - r[0], -abs((r[0] + r[1]) / 2 - ci)))
            p = int(i0 + np.argmax(diff[i0:i1 + 1]))
        else:
            lo, hi = max(1, ci - pw), min(n, ci + pw + 1)
            p = int(lo + np.argmax(diff[lo:hi]))
            i0 = i1 = p
        run = i1 - i0 + 1

        # ---- flash: a short, desaturated white burst straddling the splice.
        # Threshold at HALF the excursion so a merely brighter neighbouring shot
        # cannot run away (measured failure mode: fo=100 frames on clip_06).
        fw = max(2, int(round(0.40 * fps)))
        f_lo, f_hi = max(0, ci - fw), min(n, ci + fw + 1)
        pk_i = int(f_lo + np.argmax(luma[f_lo:f_hi]))
        luma_d = float(luma[pk_i]) - b_luma
        fthr = b_luma + max(0.5 * luma_d, 18.0)
        j0 = j1 = pk_i
        while j0 - 1 >= f_lo and luma[j0 - 1] > fthr:
            j0 -= 1
        while j1 + 1 < f_hi and luma[j1 + 1] > fthr:
            j1 += 1
        fl_span = j1 - j0 + 1
        # straddle test on a lower threshold: BOTH frames adjacent to the splice
        # must be whitened (kills "bright object enters and the cut follows")
        sthr = b_luma + max(0.25 * luma_d, 12.0)
        g0 = g1 = pk_i
        while g0 - 1 >= f_lo and luma[g0 - 1] > sthr:
            g0 -= 1
        while g1 + 1 < f_hi and luma[g1 + 1] > sthr:
            g1 += 1
        # A flash is a SYMMETRIC luma peak (fade to white, then fade in from
        # white) sitting on the splice. Anchoring the test on ci alone is too
        # brittle: scdet reports the frame the picture turns white, up to ~3
        # frames before the actual splice (measured on the demo reference).
        # measured on the HIGH (half-excursion) extent: the low threshold leaks
        # into a merely bright neighbouring shot and fakes an asymmetric peak
        rise, fall = pk_i - j0, j1 - pk_i
        if fl_span >= 3:
            # symmetric white peak, measured on the HIGH (half-excursion) extent:
            # the low threshold leaks into a merely bright neighbouring shot and
            # fakes an asymmetric peak (measured: corpus clip_06)
            straddle = (rise >= 1 and fall >= 1
                        and abs(rise - fall) <= max(2, 0.6 * max(rise, fall))
                        and abs((j0 + j1) / 2.0 - ci) <= 0.15 * fps)
        else:
            # 1-2 frame burst: too short to have a shape. Fall back to "white on
            # both sides of the reported splice" (kills a bright object that
            # slides in and is then cut away: its bright run ends at ci-1)
            straddle = (g0 <= ci - 1) and (g1 >= ci)
        k_out, k_in = max(0, ci - g0), max(0, g1 - ci + 1)
        sat_ratio = float(sat[pk_i]) / max(b_sat, 1e-3)

        # ---- whip: sharpness collapse on BOTH sides vs each side's own shot
        w = max(2, int(round(0.12 * fps)))
        gap = max(1, int(round(0.25 * fps)))
        span = max(3, int(round(0.60 * fps)))
        so_base = _med(sharp[max(0, ci - gap - span):max(1, ci - gap)], 1.0)
        si_base = _med(sharp[min(n - 1, ci + gap):min(n, ci + gap + span)], 1.0)
        so_min = float(np.min(sharp[max(0, ci - w):ci])) if ci > 0 else 1e9
        si_min = float(np.min(sharp[ci:min(n, ci + w)]))
        so_r = so_min / max(so_base, 1e-6); si_r = si_min / max(si_base, 1e-6)
        nb_out = int(np.sum(sharp[max(0, ci - w):ci] < c["whip_sharp_drop"] * so_base))
        nb_in = int(np.sum(sharp[ci:min(n, ci + w)] < c["whip_sharp_drop"] * si_base))

        # ---- dissolve: linear mixing across the elevated run + sharpness dip
        run = i1 - i0 + 1
        mix = _mix_fit(gray, i0, i1) if run >= 2 else None
        br = 0.5 * (sharp[max(i0 - 1, 0)] + sharp[min(i1 + 1, n - 1)])
        sdip = float(np.min(sharp[i0:i1 + 1])) / max(float(br), 1e-6)

        sc_peak = None
        if sc is not None and len(sc[1]):
            j0, j1 = max(0, p - pw), min(len(sc[1]), p + pw + 1)
            sc_peak = round(float(np.max(sc[1][j0:j1])), 2)

        ev = dict(diff_peak=round(float(diff[p]), 2), diff_base=round(b_diff, 3),
                  run=int(run), run_s=round(run / fps, 3),
                  luma_d=round(luma_d, 1), luma_pk=round(float(luma[pk_i]), 1),
                  sat_ratio=round(sat_ratio, 2),
                  flash_out=int(k_out), flash_in=int(k_in), flash_span=int(fl_span),
                  flash_rise=int(rise), flash_fall=int(fall),
                  sharp_dip=round(sdip, 3), sharp_out_r=round(so_r, 3),
                  sharp_in_r=round(si_r, 3), blur_out=nb_out, blur_in=nb_in,
                  shift_pk=round(float(np.max(shiftv[max(1, ci - w):min(n, ci + w)])), 4),
                  mix_res=(round(mix["res"], 3) if mix else None),
                  mix_span=(round(mix["span"], 2) if mix else None),
                  mix_mono=(round(mix["mono"], 2) if mix else None),
                  scdet_peak=sc_peak)

        typ, d, conf = "cut", 0.0, 1.0
        is_dissolve = (c["min_run"] <= run <= maxrun and mix
                       and mix["res"] <= c["mix_res_soft"]
                       and mix["span"] >= c["mix_span_min"]
                       and mix["mono"] >= c["mix_mono_min"]
                       and sdip <= c["sharp_dip_max"])
        if (straddle and luma_d >= c["flash_luma_d"]
                and float(luma[pk_i]) >= c["flash_luma_abs"]
                and sat_ratio <= c["flash_sat_max"]
                and (g1 - g0 + 1) <= c["flash_max_s"] * fps):
            # the half-excursion extent catches the middle half of a linear
            # white ramp, so the visible span is ~2x it (measured: 4/6/10 frames
            # for 2/3/5-frame-per-side flashes)
            typ = "flash"
            d = round(min(2 * max(fl_span, 1), c["flash_max_s"] * fps) / fps, 3)
            conf = min(1.0, 0.55 + 0.30 * min(1.0, luma_d / (2 * c["flash_luma_d"]))
                       + 0.15 * (1 - min(1.0, sat_ratio)))
        elif is_dissolve:
            typ = "dissolve"; d = round(run / fps, 3)
            conf = min(1.0, max(0.20, 0.40 + 0.35 * (1 - mix["res"] / c["mix_res_max"])
                                + 0.25 * min(1.0, mix["span"])))
        elif (nb_out >= c["whip_min_side"] and nb_in >= c["whip_min_side"]
              and so_r <= c["whip_sharp_drop"] and si_r <= c["whip_sharp_drop"]):
            typ = "whip"
            d = round((nb_out + nb_in) / fps, 3)
            conf = min(1.0, 0.45 + 0.3 * (1 - so_r / c["whip_sharp_drop"])
                       + 0.3 * (1 - si_r / c["whip_sharp_drop"]))
        out.append(dict(t=round(float(tc), 3), type=typ, dur=float(d),
                        confidence=round(float(conf), 2), evidence=ev))
    return out


def discover_dissolves(ft, fps, known, cfg=None, min_gap=0.35):
    """Plateau scan for dissolves that scdet never reported as cuts at all."""
    c = dict(CFG); c.update(cfg or {})
    diff = ft["diff"]; n = len(diff)
    if n < 8:
        return []
    # LOCAL baseline (block medians) — a global median is dominated by the
    # high-motion parts of the reel and hides dissolves inside calm stretches
    bs = max(4, int(round(1.0 * fps)))
    nb = int(np.ceil(n / bs))
    bm = np.array([float(np.median(diff[max(1, i * bs):min(n, (i + 1) * bs)]))
                   if min(n, (i + 1) * bs) > max(1, i * bs) else 0.0 for i in range(nb)])
    loc = np.empty(n, np.float32)
    for i in range(nb):
        j = slice(i * bs, min(n, (i + 1) * bs))
        loc[j] = np.min(bm[max(0, i - 1):min(nb, i + 2)])
    elev = np.maximum(c["run_ratio"] * loc, loc + c["run_floor"])
    cand, i = [], 1
    while i < n:
        if diff[i] > elev[i]:
            j = i
            while j + 1 < n and diff[j + 1] > elev[j + 1]:
                j += 1
            if c["min_run"] <= (j - i + 1) <= int(round(c["max_run_s"] * fps)):
                cand.append((i, j))
            i = j + 1
        else:
            i += 1
    out = []
    edge = 0.40
    tmax = n / fps
    for i0, i1 in cand:
        tmid = (i0 + i1) / 2.0 / fps
        if tmid < edge or tmid > tmax - edge:
            continue          # iOS control-centre slide-in etc., not an edit
        if any(abs(tmid - k) < min_gap for k in known):
            continue
        res = classify_boundaries([tmid], ft, fps, cfg=cfg)[0]
        if res["type"] == "dissolve":
            res["discovered"] = True
            out.append(res)
    return out


# --------------------------------------------------------------- shot moves
def _scale_between(a, b, lo=1.0, hi=1.30, steps=13):
    a = a.astype(np.float32); b = b.astype(np.float32)
    h, w = a.shape
    bn = b - b.mean()
    bd = np.sqrt(float(bn.ravel() @ bn.ravel())) + 1e-6
    best, bs = -2.0, 1.0
    for s in np.linspace(lo, hi, steps):
        ch, cw = max(2, int(h / s) // 2 * 2), max(2, int(w / s) // 2 * 2)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        z = cv2.resize(a[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
        zn = z - z.mean()
        r = float(zn.ravel() @ bn.ravel()) / ((np.sqrt(float(zn.ravel() @ zn.ravel())) + 1e-6) * bd)
        if r > best:
            best, bs = r, float(s)
    return bs, best


def shot_moves(gfeat, bounds, fps_feat, ft_native=None, fps_native=None, cfg=None):
    """Per-shot punch (center-zoom) estimate + slow-motion (duplicate cadence) hint."""
    c = dict(CFG); c.update(cfg or {})
    # global duplicate rate: a 30fps reel captured as a 60fps screen recording
    # duplicates EVERY frame (ref13: dup=0.45 in every shot). Slow motion is only
    # meaningful RELATIVE to that global cadence.
    gdup = 0.0
    if ft_native is not None:
        dg = ft_native["diff"][1:]
        if len(dg) > 8:
            gdup = float(np.mean(dg < c["dup_eps"]))
    outs = []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        i0 = int(b0 * fps_feat) + 1
        i1 = min(int(b1 * fps_feat) - 1, len(gfeat) - 1)
        rec = dict(punch=1.0, punch_r2=0.0, punch_ncc=0.0, dup_frac=0.0, slow=None,
                   global_dup=round(gdup, 3))
        if i1 - i0 >= 3:
            idx = np.linspace(i0, i1, min(5, i1 - i0 + 1)).astype(int)
            base = gfeat[idx[0]]
            sc, rr = [1.0], [1.0]
            for j in idx[1:]:
                s, r = _scale_between(base, gfeat[j])
                sc.append(s); rr.append(r)
            x = (idx - idx[0]) / max(1, (idx[-1] - idx[0]))
            y = np.array(sc)
            A = np.vstack([x, np.ones_like(x)]).T
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            ss_res = float(((y - pred) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-9
            rec.update(punch=round(float(y[-1]), 3),
                       punch_r2=round(float(1 - ss_res / ss_tot), 3),
                       punch_ncc=round(float(np.min(rr)), 3),
                       punch_monotone=bool(np.all(np.diff(y) >= -0.005)))
        if ft_native is not None and fps_native:
            j0, j1 = int(b0 * fps_native) + 2, int(b1 * fps_native) - 1
            d = ft_native["diff"][max(1, j0):max(2, j1)]
            if len(d) >= 8:
                m = d < c["dup_eps"]
                dup = float(np.mean(m))
                rec["dup_frac"] = round(dup, 3)
                # a duplicated-frame cadence is PERIODIC and the non-duplicate
                # frames still carry real motion; a merely static shot has
                # near-zero diffs everywhere (measured FP: clip_12 static, 0.58)
                pos = np.flatnonzero(m)
                gaps = np.diff(pos) if len(pos) > 2 else np.array([0])
                per = float(np.median(gaps)) if len(gaps) else 0.0
                regular = bool(len(gaps) >= 3 and per >= 2
                               and np.mean(np.abs(gaps - per) < 0.6) >= 0.7)
                live = float(np.median(d[~m])) if (~m).any() else 0.0
                rec.update(dup_period=per, dup_regular=regular, live_diff=round(live, 2))
                if (c["dup_lo"] <= dup <= c["dup_hi"] and regular and live >= 0.5
                        and dup >= gdup + c["dup_margin"]):
                    rec["slow"] = round(float(1.0 / per), 2)
        outs.append(rec)
    return outs
