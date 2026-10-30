#!/usr/bin/env python3
"""render_edit.py — deterministic EDL -> master renderer for the edit engine.

python3 STDLIB ONLY (runs on the production Mac). Shells out to ffmpeg/ffprobe.
Invariants (from real broken edits — do not relax):
  * every ffmpeg call gets -nostdin
  * segment duration via -frames:v N (N = round(dur*fps)), never -t
  * captions via ASS/libass; drawtext only for simple centered cards w/ width math
  * stills/freezes loop via: -loop 1 -framerate FPS -i img
  * segments are -an; ONE audio track muxed at the end
  * -video_track_timescale 30000 on all video segments (clean concat)
  * no -fps_mode (ffmpeg 4.4 compat): rate via fps= filter
"""
import argparse, hashlib, json, math, os, re, shlex, subprocess, sys, time

REND_VERSION = "1.3.1"
# Calibration measured in cloud container (2 cores, testsrc2 1080x1920):
#   ff6 veryfast 40.4 / medium 14.0 fps; ff4.4 veryfast 33.0 / medium 14.0 fps.
# Defaults are the conservative floor; ALWAYS re-measure on the Mac with --calibrate.
DEFAULT_ENCODE_FPS = {"draft": 33.0, "final": 12.0}   # 1080x1920 x264 fps (overridden by --encode-fps)
CALL_OVERHEAD_S = 1.2      # ffmpeg startup+probe per job
AUDIO_XRT = 25.0           # audio filter speed, x realtime (measured)
SAFETY = 0.5
WALL_CAP_S = 45.0

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",          # macOS
    "/System/Library/Fonts/Helvetica.ttc",                        # macOS
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",       # debian/ubuntu
]

def sh(argv, timeout=600):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError("cmd failed: %s\n%s" % (" ".join(map(shlex.quote, argv)), r.stderr[-3000:]))
    return r

def ffprobe_json(ffprobe, path, count=False):
    a = [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format"]
    if count: a += ["-count_frames"]
    return json.loads(sh(a + [path]).stdout)

def vstream(pj):
    return next(s for s in pj["streams"] if s["codec_type"] == "video")

def even(x): return int(math.ceil(x / 2.0) * 2)

def find_font():
    for f in FONT_CANDIDATES:
        if os.path.exists(f): return f
    return None

# ---------------- planning ----------------

def load_edl(path):
    edl = json.load(open(path))
    assert edl.get("version") == 1
    return edl

def plan(edl):
    fps = edl["fps"]; W = edl["width"]; H = edl["height"]
    prof = edl.get("profile", "final")
    slots = []
    for i, s in enumerate(edl["slots"]):
        n = int(s["dur_frames"]) if "dur_frames" in s else round(float(s["dur"]) * fps)
        assert n > 0, "slot %s zero frames" % s["id"]
        tr = s.get("transition") or {"type": "cut"}
        xf = round(float(tr.get("dur", 0)) * fps) if tr["type"] == "xfade" else 0
        fl = round(float(tr.get("dur", 2.0 / fps)) * fps) if tr["type"] == "flash" else 0
        if i == 0: assert tr["type"] == "cut", "first slot cannot have a transition"
        slots.append(dict(idx=i, id=s["id"], src=s.get("src"), src_in=float(s.get("src_in", 0)),
                          n=n, speed=float(s.get("speed", 1.0)), punch=float(s.get("punch", 1.0)),
                          tr=tr["type"], xf=xf, fl=fl, card=s.get("card"), freeze=bool(s.get("freeze"))))
    # xfade INTO slot k extends slot k-1 by xf frames (handle material), timeline length unchanged
    for k, s in enumerate(slots):
        s["ext"] = 0
    for k, s in enumerate(slots):
        if s["xf"]:
            assert 0 < s["xf"] < min(s["n"], slots[k-1]["n"]), "xfade longer than a neighbor"
            slots[k-1]["ext"] = s["xf"]
    total = sum(s["n"] for s in slots)
    pos = 0
    for s in slots:
        s["t0"] = pos / fps; pos += s["n"]
    return dict(fps=fps, W=W, H=H, prof=prof, slots=slots, total=total, T=total / fps,
                audio=edl.get("audio", {}), subtitles=edl.get("subtitles"))

def src_ident(path):
    st = os.stat(path)
    return "%s|%d|%d" % (os.path.abspath(path), st.st_size, int(st.st_mtime))

def slot_key(p, s, seek_mode):
    d = dict(v=REND_VERSION, fps=p["fps"], W=p["W"], H=p["H"], prof=p["prof"], seek=seek_mode,
             n=s["n"], ext=s["ext"], src_in=s["src_in"], speed=s["speed"], punch=s["punch"],
             tr=s["tr"], xf=s["xf"], fl=s["fl"], card=s["card"], freeze=s["freeze"],
             fl_out=_flash_out(p, s), src=src_ident(s["src"]) if s["src"] else None,
             subs=src_ident(p["subtitles"]) if p.get("subtitles") else None, t0=s["t0"])
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

def _flash_out(p, s):
    """frames of white fade-out this slot needs because the NEXT slot starts with a flash"""
    nxt = p["slots"][s["idx"] + 1] if s["idx"] + 1 < len(p["slots"]) else None
    return nxt["fl"] if nxt and nxt["tr"] == "flash" else 0

# ---------------- command builders ----------------

def enc_args(p, n):
    preset, crf = ("veryfast", 23) if p["prof"] == "draft" else ("medium", 20)
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-frames:v", str(n), "-an", "-video_track_timescale", "30000"]

def fit_chain(p, s):
    pw, ph = even(p["W"] * s["punch"]), even(p["H"] * s["punch"])
    c = ["scale=%d:%d:force_original_aspect_ratio=increase:flags=bicubic" % (pw, ph),
         "crop=%d:%d" % (p["W"], p["H"])]
    return c

def vfilters(p, s):
    f = []
    if s["speed"] != 1.0: f.append("setpts=(PTS-STARTPTS)/%g" % s["speed"])
    else: f.append("setpts=PTS-STARTPTS")
    f.append("fps=%d" % p["fps"])
    f += fit_chain(p, s)
    if s["tr"] == "flash" and s["fl"]:
        f.append("fade=t=in:s=0:n=%d:color=white" % s["fl"])
    fo = _flash_out(p, s)
    if fo:
        f.append("fade=t=out:s=%d:n=%d:color=white" % (s["n"] - fo, fo))
    f += sub_burn(p, s)
    f += ["setsar=1", "format=yuv420p"]
    return ",".join(f)

def sub_burn(p, s):
    """Captions burn PER SEGMENT via libass: shift pts to timeline time, render, unshift.
    Keeps mux at -c:v copy (a whole-timeline ass burn blew the Mac 45s cap: 46.7s measured)
    and localizes caption edits to the segments they touch."""
    if not p.get("subtitles"): return []
    return ["setpts=PTS+%.6f/TB" % s["t0"],
            "ass=filename=%s%s" % (p["subtitles"], ":fontsdir=" + os.environ["ENGINE_FONTSDIR"] if os.environ.get("ENGINE_FONTSDIR") else ""),
            "setpts=PTS-STARTPTS"]

def wrap_card(text, W, max_frac=0.86, base_size=None):
    """centered card width math: estimate DejaVuSans/Arial-Bold advance ~0.52*fontsize per char"""
    base = base_size or int(W * 0.085)
    # The planner (which has PIL and the real font metrics) normally precomputes
    # this and passes lines+size in the EDL — see engine.py:_layout_card. The
    # arithmetic below is only a fallback for hand-written EDLs, and it is
    # DELIBERATELY conservative: a 0.52 advance estimate under-measured real
    # DejaVu Bold and clipped "KEEPS ASKING ABOUT THE" off both frame edges.
    words, lines, cur = text.split(), [], ""
    limit = max(8, int((W * max_frac) / (0.62 * base)))
    for w in words:
        t = (cur + " " + w).strip()
        if len(t) <= limit: cur = t
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    if not lines: lines = [" "]      # blank card (silent slot in a text-on-black cut)
    longest = max(1, max(len(l) for l in lines))
    size = min(base, int((W * max_frac) / (0.62 * longest)))
    return "\n".join(lines), size

def seg_cmds(ff, p, s, build, seek_mode):
    """Return list of argv commands that produce build/seg_<id>.mp4"""
    out = os.path.join(build, "seg_%s.mp4" % s["id"])
    n = s["n"] + s["ext"]
    cmds = []
    if s["card"]:
        font = find_font()
        c = s["card"]
        if c.get("lines") and c.get("size"):
            txt, size = "\n".join(c["lines"]), int(c["size"])   # measured upstream
        else:
            txt, size = wrap_card(c["text"], p["W"])
        tf = os.path.join(build, "card_%s.txt" % s["id"])
        open(tf, "w").write(txt)
        dt = ("drawtext=fontfile=%s:textfile=%s:fontsize=%d:fontcolor=white:"
              "x=(w-text_w)/2:y=(h-text_h)/2:line_spacing=%d" % (font, tf, size, size // 3))
        vf = ",".join([dt] + (["fade=t=in:s=0:n=%d:color=white" % s["fl"]] if s["tr"] == "flash" and s["fl"] else [])
                      + (["fade=t=out:s=%d:n=%d:color=white" % (s["n"] - _flash_out(p, s), _flash_out(p, s))] if _flash_out(p, s) else [])
                      + sub_burn(p, s) + ["setsar=1", "format=yuv420p"])
        cmds.append([ff, "-nostdin", "-hide_banner", "-y", "-f", "lavfi",
                     "-i", "color=c=black:s=%dx%d:r=%d" % (p["W"], p["H"], p["fps"]),
                     "-filter:v", vf] + enc_args(p, n) + [out])
        return cmds, out
    if s["freeze"]:
        png = os.path.join(build, "frz_%s.png" % s["id"])
        cmds.append([ff, "-nostdin", "-hide_banner", "-y", "-ss", "%.6f" % s["src_in"],
                     "-i", s["src"], "-frames:v", "1", png])
        cmds.append([ff, "-nostdin", "-hide_banner", "-y", "-loop", "1", "-framerate", str(p["fps"]),
                     "-i", png, "-filter:v", vfilters(p, s)] + enc_args(p, n) + [out])
        return cmds, out
    seek_pre, seek_post = [], []
    if seek_mode == "fast" or s["src_in"] == 0:
        seek_pre = ["-ss", "%.6f" % s["src_in"]] if s["src_in"] else []
    elif seek_mode == "exact":
        seek_post = ["-ss", "%.6f" % s["src_in"]]
    else:  # hybrid: coarse pre-seek 2s early, exact output seek for remainder
        pre = max(0.0, s["src_in"] - 2.0)
        seek_pre = ["-ss", "%.6f" % pre] if pre else []
        seek_post = ["-ss", "%.6f" % (s["src_in"] - pre)]
    cmds.append([ff, "-nostdin", "-hide_banner", "-y"] + seek_pre + ["-i", s["src"]] + seek_post
                + ["-filter:v", vfilters(p, s)] + enc_args(p, n) + [out])
    return cmds, out

def pair_cmds(ff, p, k, build):
    """xfade the (extended) previous segment into segment k; replaces both in concat"""
    a, b = p["slots"][k - 1], p["slots"][k]
    fa = os.path.join(build, "seg_%s.mp4" % a["id"]); fb = os.path.join(build, "seg_%s.mp4" % b["id"])
    out = os.path.join(build, "pair_%s_%s.mp4" % (a["id"], b["id"]))
    off = a["n"] / p["fps"]                     # crossfade starts where A originally ended
    dur = b["xf"] / p["fps"]
    n = a["n"] + b["n"]                         # timeline length preserved (A was extended by xf)
    # tpad clone guard: ffmpeg 4.4 xfade emits one frame fewer at EOF than 6.x (measured
    # 179 vs 180); clone-pad the tail and let -frames:v cap at exactly n on both.
    fc = ("[0:v][1:v]xfade=transition=fade:duration=%.6f:offset=%.6f,fps=%d,"
          "tpad=stop=2:stop_mode=clone,setsar=1,format=yuv420p[v]" % (dur, off, p["fps"]))
    return [[ff, "-nostdin", "-hide_banner", "-y", "-i", fa, "-i", fb,
             "-filter_complex", fc, "-map", "[v]"] + enc_args(p, n) + [out]], out, n

# ---------------- job graph ----------------

def jobs_for(p, build):
    J = []
    for s in p["slots"]:
        J.append(("seg:%s" % s["id"], s))
    for k, s in enumerate(p["slots"]):
        if s["xf"]: J.append(("pair:%d" % k, k))
    J.append(("concat", None))
    a = p["audio"]
    if a.get("vo"):
        J += [("vo_meas", None), ("vo_norm", None), ("mix", None),
              ("mix_meas", None), ("mix_final", None)]
    J.append(("mux", None))
    return J

def parse_loudnorm(stderr):
    m = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.S)
    assert m, "no loudnorm json in output"
    return json.loads(m[-1])

def loudnorm_str(tgt, tp, meas=None):
    s = "loudnorm=I=%g:TP=%g:LRA=11" % (tgt, tp)
    if meas:
        s += (":measured_I=%s:measured_TP=%s:measured_LRA=%s:measured_thresh=%s:offset=%s:linear=true"
              % (meas["input_i"], meas["input_tp"], meas["input_lra"], meas["input_thresh"], meas["target_offset"]))
    return s + ":print_format=json"

DUCK_DEFAULT = dict(threshold=0.05, ratio=8, attack=20, release=400, makeup=1)
# lighter default than music: a diegetic sfx one-shot only needs to duck for
# its own brief overlap with speech, not hold a bed down for a whole scene.
SFX_DUCK_DEFAULT = dict(threshold=0.06, ratio=6, attack=15, release=300, makeup=1)

def run_job(name, arg, p, build, ff, ffprobe, seek_mode, out_path, force=False):
    fps, T = p["fps"], p["T"]
    okf = lambda tag: os.path.join(build, ".%s.ok" % tag)

    def cached(tag, key):
        f = okf(tag)
        if not force and os.path.exists(f):
            try:
                return json.load(open(f)).get("key") == key
            except Exception: return False
        return False

    def mark(tag, key, extra=None):
        json.dump(dict(key=key, **(extra or {})), open(okf(tag), "w"))

    if name.startswith("seg:"):
        s = arg; key = slot_key(p, s, seek_mode)
        tag = "seg_%s" % s["id"]
        if cached(tag, key): return "cached"
        cmds, out = seg_cmds(ff, p, s, build, seek_mode)
        for c in cmds: sh(c)
        got = int(vstream(ffprobe_json(ffprobe, out, count=True))["nb_read_frames"])
        want = s["n"] + s["ext"]
        assert got == want, "seg %s frames %d != planned %d" % (s["id"], got, want)
        mark(tag, key, dict(frames=got)); return "rendered(%d f)" % got

    if name.startswith("pair:"):
        k = arg; a, b = p["slots"][k - 1], p["slots"][k]
        key = "pv2:" + slot_key(p, a, seek_mode) + slot_key(p, b, seek_mode)
        tag = "pair_%s_%s" % (a["id"], b["id"])
        if cached(tag, key): return "cached"
        cmds, out, n = pair_cmds(ff, p, k, build)
        for c in cmds: sh(c)
        got = int(vstream(ffprobe_json(ffprobe, out, count=True))["nb_read_frames"])
        assert got == n, "pair frames %d != %d" % (got, n)
        mark(tag, key, dict(frames=got)); return "rendered(%d f)" % got

    if name == "concat":
        segs, keys, i = [], [], 0
        slots = p["slots"]
        while i < len(slots):
            if i + 1 < len(slots) and slots[i + 1]["xf"]:
                segs.append(os.path.join(build, "pair_%s_%s.mp4" % (slots[i]["id"], slots[i + 1]["id"]))); i += 2
            else:
                segs.append(os.path.join(build, "seg_%s.mp4" % slots[i]["id"])); i += 1
        key = hashlib.sha1(json.dumps([json.load(open(okf("seg_%s" % s["id"])))["key"] for s in slots]).encode()).hexdigest()[:16]
        if cached("timeline", key): return "cached"
        lst = os.path.join(build, "concat.txt")
        open(lst, "w").write("".join("file '%s'\n" % os.path.abspath(x) for x in segs))
        tl = os.path.join(build, "timeline.mp4")
        sh([ff, "-nostdin", "-hide_banner", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", tl])
        got = int(vstream(ffprobe_json(ffprobe, tl, count=True))["nb_read_frames"])
        assert got == p["total"], "timeline frames %d != %d" % (got, p["total"])
        mark("timeline", key, dict(frames=got)); return "concat(%d f)" % got

    a = p["audio"]
    AUDIO_V = "3"
    akey = hashlib.sha1(json.dumps([a, T, REND_VERSION, AUDIO_V], sort_keys=True).encode()).hexdigest()[:16]

    if name == "vo_meas":
        if cached("vo_meas", akey): return "cached"
        r = sh([ff, "-nostdin", "-hide_banner", "-y", "-i", a["vo"],
                "-af", loudnorm_str(-16, -2), "-f", "null", "-"])
        meas = parse_loudnorm(r.stderr)
        # trap guard (measured): loudnorm on near-silence flips to dynamic mode and
        # amplifies the noise floor to program level. Refuse quiet/empty VO outright.
        assert float(meas["input_i"]) > -35.0, \
            "VO measures %s LUFS — near-silent input, loudnorm would produce garbage" % meas["input_i"]
        json.dump(meas, open(os.path.join(build, "vo_meas.json"), "w"))
        mark("vo_meas", akey); return "measured"

    if name == "vo_norm":
        if cached("vo_norm", akey): return "cached"
        meas = json.load(open(os.path.join(build, "vo_meas.json")))
        sh([ff, "-nostdin", "-hide_banner", "-y", "-i", a["vo"],
            "-af", loudnorm_str(-16, -2, meas) + ",aresample=48000,apad=whole_dur=%.6f,atrim=0:%.6f" % (T, T),
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", os.path.join(build, "vo_norm.wav")])
        mark("vo_norm", akey); return "normalized"

    if name == "mix":
        if cached("mix", akey): return "cached"
        von = os.path.join(build, "vo_norm.wav"); mx = os.path.join(build, "mix.wav")
        m = a.get("music")
        sfx_list = a.get("sfx") or []
        if not m and not sfx_list:
            sh([ff, "-nostdin", "-hide_banner", "-y", "-i", von, "-c:a", "pcm_s16le", mx])
        else:
            # explicit aformat on every sidechaincompress input: ffmpeg 4.4 fails on
            # guessed wav channel layouts ("No channel layout for input 1") without it
            AF = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
            beds = []  # each: dict(file=, pre=<filter chain up to and incl. AF>, duck=dict)
            if m:
                d = dict(DUCK_DEFAULT); d.update(m.get("duck", {}))
                if "target_lufs" in m:   # set bed level by measurement, not blind gain
                    r = sh([ff, "-nostdin", "-hide_banner", "-y", "-i", m["file"],
                            "-af", loudnorm_str(-16, -2), "-f", "null", "-"])
                    mi = float(parse_loudnorm(r.stderr)["input_i"])
                    gain = m["target_lufs"] - mi
                else:
                    gain = m.get("gain_db", -21)
                beds.append(dict(file=m["file"], duck=d,
                                 pre="volume=%gdB,aresample=48000,%s" % (gain, AF)))
            # DIEGETIC sfx: each is a short one-shot placed at its own shot's
            # start via adelay, then ducked under the VO exactly like a music
            # bed is -- "the VO must stay dominant" applies to every audio
            # layer, not just music. Already loudness-normalised/limited at
            # fetch time (fetch_assets.py normalize_sfx), so gain_db defaults
            # to 0 here; it is per-event only as an override valve.
            for sx in sfx_list:
                d = dict(SFX_DUCK_DEFAULT); d.update(sx.get("duck", {}))
                gain = float(sx.get("gain_db", 0))
                delay_ms = max(0, int(round(float(sx.get("start", 0)) * 1000)))
                beds.append(dict(file=sx["file"], duck=d,
                                 pre="volume=%gdB,aresample=48000,%s,adelay=%d|%d"
                                     % (gain, AF, delay_ms, delay_ms)))
            n = len(beds)
            split_labels = ["vk%d" % i for i in range(n)] + ["vmx"]
            fc = ["[0:a]%s,asplit=%d%s;" % (AF, n + 1, "".join("[%s]" % l for l in split_labels))]
            sd_labels = []
            for i, b in enumerate(beds):
                fc.append("[%d:a]%s[bed%d];" % (i + 1, b["pre"], i))
                dd = b["duck"]
                fc.append("[bed%d][vk%d]sidechaincompress=threshold=%g:ratio=%g:attack=%g:"
                          "release=%g:makeup=%g[sd%d];"
                          % (i, i, dd["threshold"], dd["ratio"], dd["attack"], dd["release"],
                             dd["makeup"], i))
                sd_labels.append("sd%d" % i)
            mix_in = "[vmx]" + "".join("[%s]" % l for l in sd_labels)
            fc.append("%samix=inputs=%d:duration=first:normalize=0,apad=whole_dur=%.6f,atrim=0:%.6f[out]"
                      % (mix_in, n + 1, T, T))
            args = [ff, "-nostdin", "-hide_banner", "-y", "-i", von]
            for b in beds:
                args += ["-i", b["file"]]
            args += ["-filter_complex", "".join(fc), "-map", "[out]",
                     "-ar", "48000", "-c:a", "pcm_s16le", mx]
            sh(args)
        mark("mix", akey); return "mixed"

    if name == "mix_meas":
        if cached("mix_meas", akey): return "cached"
        tgt = a.get("target_lufs", -14)
        # PCM target TP -2.0: AAC encode overshoots ~0.1-0.6 dB; keeps final TP <= -1.5 (measured)
        r = sh([ff, "-nostdin", "-hide_banner", "-y", "-i", os.path.join(build, "mix.wav"),
                "-af", loudnorm_str(tgt, -3.0), "-f", "null", "-"])
        json.dump(parse_loudnorm(r.stderr), open(os.path.join(build, "mix_meas.json"), "w"))
        mark("mix_meas", akey); return "measured"

    if name == "mix_final":
        if cached("mix_final", akey): return "cached"
        meas = json.load(open(os.path.join(build, "mix_meas.json")))
        tgt = a.get("target_lufs", -14)
        sh([ff, "-nostdin", "-hide_banner", "-y", "-i", os.path.join(build, "mix.wav"),
            "-af", loudnorm_str(tgt, -3.0, meas) + ",aresample=48000,apad=whole_dur=%.6f,atrim=0:%.6f" % (T, T),
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", os.path.join(build, "mix_final.wav")])
        mark("mix_final", akey); return "final audio"

    if name == "mux":
        tl = os.path.join(build, "timeline.mp4")
        args = [ff, "-nostdin", "-hide_banner", "-y", "-i", tl]
        au = os.path.join(build, "mix_final.wav")
        if p["audio"].get("vo"):
            args += ["-i", au, "-map", "0:v:0", "-map", "1:a:0",
                     "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
        else:
            args += ["-map", "0:v:0", "-an"]
        sh(args + ["-c:v", "copy", "-movflags", "+faststart", out_path])
        return "muxed -> %s" % out_path

    raise ValueError(name)

# ---------------- estimation / --calls ----------------

def est_seconds(name, arg, p, enc_fps):
    if name.startswith("seg:"):
        s = arg; n = s["n"] + s["ext"]
        base = CALL_OVERHEAD_S + n / enc_fps
        return base + (CALL_OVERHEAD_S if s["freeze"] else 0)
    if name.startswith("pair:"):
        k = arg; n = p["slots"][k - 1]["n"] + p["slots"][k]["n"]
        return CALL_OVERHEAD_S + n / enc_fps
    if name == "concat": return CALL_OVERHEAD_S + p["total"] / 2000.0
    if name == "mux": return CALL_OVERHEAD_S + p["total"] / 1500.0 + p["T"] / AUDIO_XRT
    return CALL_OVERHEAD_S + p["T"] / AUDIO_XRT   # audio jobs

def print_calls(p, edl_path, build, out_path, ff, ffprobe, enc_fps, seek_mode):
    cap = WALL_CAP_S * SAFETY
    J = jobs_for(p, build)
    groups, cur, cur_t = [], [], 0.0
    for name, arg in J:
        e = est_seconds(name, arg, p, enc_fps)
        if cur and cur_t + e > cap:
            groups.append((cur, cur_t)); cur, cur_t = [], 0.0
        cur.append(name); cur_t += e
        if e > cap:
            sys.stderr.write("WARN job %s est %.1fs alone exceeds cap %.1fs\n" % (name, e, cap))
    if cur: groups.append((cur, cur_t))
    print("# %d jobs -> %d calls (cap %.1fs = %.0fs wall * %.1f safety; enc_fps=%.1f)"
          % (len(J), len(groups), cap, WALL_CAP_S, SAFETY, enc_fps))
    base = "python3 %s %s --build %s --out %s --ffmpeg %s --ffprobe %s --seek-mode %s" % (
        os.path.abspath(sys.argv[0]), os.path.abspath(edl_path), build, out_path, ff, ffprobe, seek_mode)
    for g, t in groups:
        print("%s --jobs %s  # est %.1fs" % (base, ",".join(g), t))

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edl"); ap.add_argument("--build", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ffmpeg", default="ffmpeg"); ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--calls", action="store_true"); ap.add_argument("--jobs")
    ap.add_argument("--encode-fps", type=float); ap.add_argument("--seek-mode", default="fast",
                    choices=["fast", "exact", "hybrid"])
    ap.add_argument("--force", action="store_true"); ap.add_argument("--verify", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    if a.calibrate:
        t0 = time.time()
        sh([a.ffmpeg, "-nostdin", "-hide_banner", "-y", "-f", "lavfi", "-i", "testsrc2=s=1080x1920:r=30:d=3",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-frames:v", "90", "-f", "null", "-"])
        print("measured encode fps (medium 1080x1920): %.1f" % (90 / (time.time() - t0))); return
    os.makedirs(a.build, exist_ok=True)
    p = plan(load_edl(a.edl))
    enc_fps = a.encode_fps or DEFAULT_ENCODE_FPS["draft" if p["prof"] == "draft" else "final"]
    if a.calls:
        print_calls(p, a.edl, a.build, a.out, a.ffmpeg, a.ffprobe, enc_fps, a.seek_mode); return
    if a.verify:
        rep = dict(slots=[], total_planned=p["total"], T=p["T"])
        for s in p["slots"]:
            f = os.path.join(a.build, "seg_%s.mp4" % s["id"])
            got = int(vstream(ffprobe_json(a.ffprobe, f, count=True))["nb_read_frames"])
            rep["slots"].append(dict(id=s["id"], planned=s["n"] + s["ext"], got=got, ok=got == s["n"] + s["ext"]))
        tl = os.path.join(a.build, "timeline.mp4")
        if os.path.exists(tl):
            rep["timeline_frames"] = int(vstream(ffprobe_json(a.ffprobe, tl, count=True))["nb_read_frames"])
        if os.path.exists(a.out):
            mj = ffprobe_json(a.ffprobe, a.out)
            rep["master"] = dict(duration=float(mj["format"]["duration"]),
                                 streams=[(s["codec_type"], s.get("codec_name")) for s in mj["streams"]])
        print(json.dumps(rep, indent=1)); return
    todo = jobs_for(p, a.build)
    if a.jobs:
        want = a.jobs.split(","); todo = [j for j in todo if j[0] in want]
        missing = set(want) - set(j[0] for j in todo)
        assert not missing, "unknown jobs: %s" % missing
    for name, arg in todo:
        t0 = time.time()
        r = run_job(name, arg, p, a.build, a.ffmpeg, a.ffprobe, a.seek_mode, a.out, a.force)
        print("%-14s %-18s %5.1fs" % (name, r, time.time() - t0))

if __name__ == "__main__":
    main()
