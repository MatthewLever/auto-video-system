#!/usr/bin/env python3
"""qc.py - one JSON verdict for a finished master + its plan.

Checks: stream asserts (res/codec/profile/pix_fmt/fps, aac 48k stereo),
duration vs plan +-0.15s, exact frame count, integrated LUFS -14+-1 &
TP <= -1.4 (ebur128), blackdetect/freezedetect vs PLANNED whitelist,
A/V start alignment, +faststart (moov before mdat, real box walk),
per-profile size caps, caption_check integration (plan.ass).

plan JSON: {width,height,fps,duration,frames,profile,lufs_target,
            planned_black:[[t0,t1],..],planned_freeze:[[t0,t1],..],
            ass:path|null,platform,size_cap_mib:num|null}
"""
import argparse, json, os, re, struct, subprocess, sys, time

D = os.path.dirname(os.path.abspath(__file__))
TOL_DUR = 0.15
TOL_LUFS = 1.0
TP_MAX = -1.4
BLACK_WHITELIST_PAD = 0.30   # s slack around planned segments
AV_START_TOL = 0.05

def probe(path):
    r = subprocess.run(
        f"ffprobe -v error -show_streams -show_format -count_packets -of json '{path}'",
        shell=True, capture_output=True, text=True)
    return json.loads(r.stdout)

def analyze_av(path):
    """blackdetect + freezedetect + ebur128 in one pass; parse stderr."""
    r = subprocess.run(
        f"ffmpeg -hide_banner -nostats -i '{path}' "
        f"-vf blackdetect=d=0.3:pix_th=0.10,freezedetect=n=-60dB:d=2 "
        f"-af ebur128=peak=true -f null -",
        shell=True, capture_output=True, text=True)
    e = r.stderr
    blacks = [(float(a), float(b)) for a, b in
              re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", e)]
    fstarts = [float(x) for x in re.findall(r"lavfi\.freezedetect\.freeze_start: ([\d.]+)", e)]
    fends = [float(x) for x in re.findall(r"lavfi\.freezedetect\.freeze_end: ([\d.]+)", e)]
    freezes = list(zip(fstarts, fends + [None] * (len(fstarts) - len(fends))))
    tail = e[e.rfind("Summary:"):]
    lufs = tp = None
    lines = tail.splitlines()
    for i, L in enumerate(lines):
        if L.strip().startswith("I:") and lufs is None:
            lufs = float(L.split()[1])
        if "True peak" in L:
            for L2 in lines[i:]:
                if L2.strip().startswith("Peak:"):
                    tp = float(L2.split()[1]); break
    return blacks, freezes, lufs, tp

def moov_before_mdat(path):
    """Walk top-level mp4 boxes; return (ok, order)."""
    order = []
    with open(path, "rb") as f:
        sz = os.path.getsize(path)
        off = 0
        while off < sz - 8:
            f.seek(off)
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            n, typ = struct.unpack(">I4s", hdr)
            typ = typ.decode("latin1")
            if n == 1:
                n = struct.unpack(">Q", f.read(8))[0]
            elif n == 0:
                n = sz - off
            order.append(typ)
            off += n
    ok = "moov" in order and "mdat" in order and order.index("moov") < order.index("mdat")
    return ok, order

def _coalesce(wl, pad=BLACK_WHITELIST_PAD):
    """Merge touching/overlapping planned intervals.

    Detectors report one continuous run, but plans list intervals per slot —
    a text-on-black cut plans black slot-by-slot and blackdetect merges them.
    Comparing one-to-one flags a correct render as broken, so coalesce first
    and ask only whether the detected run is COVERED by planned black."""
    iv = sorted([float(w0), float(w1)] for w0, w1 in wl)
    out = []
    for w0, w1 in iv:
        if out and w0 <= out[-1][1] + 2 * pad:
            out[-1][1] = max(out[-1][1], w1)
        else:
            out.append([w0, w1])
    return out

def in_whitelist(seg, wl, pad=BLACK_WHITELIST_PAD):
    s, e = seg
    e = e if e is not None else s
    return any(s >= w0 - pad and e <= w1 + pad for w0, w1 in _coalesce(wl, pad))

def qc(path, plan, run_caption_check=True):
    t0 = time.time()
    checks = []
    def add(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    p = probe(path)
    v = next((s for s in p["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in p["streams"] if s["codec_type"] == "audio"), None)
    if not v or not a:
        add("streams_exist", False, f"video={bool(v)} audio={bool(a)}")
        return {"file": path, "checks": checks, "pass": False}

    add("resolution", v["width"] == plan["width"] and v["height"] == plan["height"],
        f"{v['width']}x{v['height']} vs plan {plan['width']}x{plan['height']}")
    num, den = map(int, v["r_frame_rate"].split("/"))
    fps = num / den
    add("fps", abs(fps - plan["fps"]) < 0.01, f"{fps:.3f} vs {plan['fps']}")
    add("vcodec", v["codec_name"] == "h264" and v.get("profile") == "High"
        and v["pix_fmt"] == "yuv420p",
        f"{v['codec_name']}/{v.get('profile')}/{v['pix_fmt']}")
    add("acodec", a["codec_name"] == "aac" and int(a["sample_rate"]) == 48000
        and a["channels"] == 2,
        f"{a['codec_name']}/{a['sample_rate']}Hz/{a['channels']}ch")

    dur = float(p["format"]["duration"])
    add("duration", abs(dur - plan["duration"]) <= TOL_DUR,
        f"{dur:.3f}s vs plan {plan['duration']}s (tol {TOL_DUR})")
    nframes = int(v.get("nb_read_packets", -1))
    add("frame_count", nframes == plan["frames"], f"{nframes} vs plan {plan['frames']}")

    vs, as_ = float(v.get("start_time", 0)), float(a.get("start_time", 0))
    add("av_start_align", abs(vs - as_) <= AV_START_TOL, f"v={vs} a={as_}")

    blacks, freezes, lufs, tp = analyze_av(path)
    tgt = plan.get("lufs_target", -14.0)
    add("loudness_integrated", lufs is not None and abs(lufs - tgt) <= TOL_LUFS,
        f"I={lufs} LUFS vs {tgt}+-{TOL_LUFS}")
    add("true_peak", tp is not None and tp <= TP_MAX, f"TP={tp} dBTP (max {TP_MAX})")

    wl_b = plan.get("planned_black", [])
    bad_b = [s for s in blacks if not in_whitelist(s, wl_b)]
    add("blackdetect_whitelist", not bad_b,
        f"detected={blacks} planned={wl_b} unplanned={bad_b}")
    wl_f = plan.get("planned_freeze", [])
    bad_f = [s for s in freezes if not in_whitelist(s, wl_f)]
    add("freezedetect_whitelist", not bad_f,
        f"detected={freezes} planned={wl_f} unplanned={bad_f}")

    ok_fs, order = moov_before_mdat(path)
    add("faststart_moov_first", ok_fs, f"box order: {order}")

    cap = plan.get("size_cap_mib")
    size = os.path.getsize(path)
    if cap:
        add("size_cap", size < cap * 1024 * 1024, f"{size}B vs {cap}MiB")
    else:
        add("size_cap", True, f"{size}B (no cap for profile)")

    if run_caption_check and plan.get("ass"):
        sys.path.insert(0, D)
        from caption_check import check as ccheck
        cv = ccheck(plan["ass"], plan.get("platform", "engine_default"))
        add("caption_check", cv["pass"],
            {"failures": cv["failures"][:8], "runtime_s": cv["runtime_s"]})

    return {"file": path, "profile": plan.get("profile"),
            "pass": all(c["pass"] for c in checks),
            "failed": [c["check"] for c in checks if not c["pass"]],
            "runtime_s": round(time.time() - t0, 2), "checks": checks}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--json")
    ap.add_argument("--no-captions", action="store_true")
    a = ap.parse_args()
    v = qc(a.file, json.load(open(a.plan)), not a.no_captions)
    print(json.dumps(v, indent=1, default=str))
    if a.json:
        json.dump(v, open(a.json, "w"), indent=1, default=str)
    sys.exit(0 if v["pass"] else 1)

if __name__ == "__main__":
    main()
