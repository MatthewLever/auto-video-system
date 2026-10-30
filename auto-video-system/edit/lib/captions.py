#!/usr/bin/env python3
"""captions.py - whisper words -> phrase cards -> styled ASS (1080x1920).

Empirically validated facts baked in (see research log):
- libass 0.17.1 parses Style: lines POSITIONALLY against the declared Format: line.
  A 22-field style under the standard 23-field V4+ Format shifts every field after
  OutlineColour left by one, landing 0 in ScaleY -> text renders as NOTHING, silently.
  This emitter therefore always writes a full 23-field Format line and asserts every
  Style: line it emits has exactly 23 fields.
- Fontsize N at PlayResY==output height renders DejaVu Sans cap height ~0.71*N px
  (62 -> 44 px), full ascender-descender ink ~0.92*N. Scales linearly with
  out_h/PlayResY.
"""
import argparse, json, os, re, sys

D = os.path.dirname(os.path.abspath(__file__))
SAFE_AREAS_JSON = os.path.join(D, "safe_areas.json")

# ---------------- config ----------------
DEFAULTS = dict(
    max_words=4,            # words per card
    gap_break=0.40,         # s: silence gap forcing a card break
    extend_cap=0.30,        # s: extend card end toward next card start, capped
    end_punct=".!?",        # break after these
    platform="engine_default" # safe-area profile to respect
)
PROPER_NOUNS = {  # lowercased key -> canonical
    "lebron": "LeBron", "cheerios": "Cheerios", "youngsponge": "YoungSponge",
    "iphone": "iPhone", "tiktok": "TikTok", "shopify": "Shopify",
}
PROPER_BIGRAMS = {("young", "sponge"): "YoungSponge", ("le", "bron"): "LeBron"}
NUMBER_FIX = {}

# Brand terms live in config/terms.json so they can be corrected without editing
# code. Numbers matter most: whisper heard "$75" as "$70" on a real take, and a
# wrong price burned into a reel is worse than any typo.
_TERMS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "terms.json")
if os.path.exists(_TERMS):
    try:
        _t = json.load(open(_TERMS))
        PROPER_NOUNS.update({k.lower(): v for k, v in _t.get("unigrams", {}).items()})
        PROPER_BIGRAMS.update({tuple(k.split()): v for k, v in _t.get("bigrams", {}).items()})
        NUMBER_FIX = {k.lower(): v for k, v in _t.get("numbers", {}).items()}
    except Exception:
        pass

FALLBACK_SAFE = {  # conservative fallback if safe_areas.json missing (px on 1080x1920)
    "intersection": {"top": 220, "bottom": 440, "left": 70, "right": 140}
}

def load_safe(platform):
    data = FALLBACK_SAFE
    if os.path.exists(SAFE_AREAS_JSON):
        data = json.load(open(SAFE_AREAS_JSON))["platforms"]
    p = data.get(platform) or FALLBACK_SAFE["intersection"]
    return {k: int(p[k]) for k in ("top", "bottom", "left", "right")}

# ---------------- word post-processing ----------------
def clean_words(words):
    """Merge hyphen continuations, strip whitespace, fix proper nouns."""
    out = []
    for w in words:
        t = w["word"].strip()
        if not t:
            continue
        if out and (t.startswith("-") or out[-1]["word"].endswith("-")):
            out[-1]["word"] = (out[-1]["word"].rstrip("-") + t.lstrip("-"))
            out[-1]["end"] = w["end"]
            continue
        out.append({"word": t, "start": float(w["start"]), "end": float(w["end"])})
    # proper nouns: bigrams first (merge), then unigrams
    merged = []
    i = 0
    while i < len(out):
        key2 = None
        if i + 1 < len(out):
            key2 = (_bare(out[i]["word"]).lower(), _bare(out[i+1]["word"]).lower())
        if key2 in PROPER_BIGRAMS:
            tail = re.sub(r"^[\w']+", "", out[i+1]["word"])  # keep trailing punct
            merged.append({"word": PROPER_BIGRAMS[key2] + tail,
                           "start": out[i]["start"], "end": out[i+1]["end"]})
            i += 2
        else:
            w = out[i]
            # key on the alphanumeric core: _bare() leaves "$" and "." attached
            # ("$70." stays "$70."), so match on "70" and swap only the digits,
            # preserving the "$" and any trailing punctuation.
            core = re.sub(r"[^\w]", "", w["word"]).lower()
            if core and core in NUMBER_FIX:
                w = dict(w, word=w["word"].replace(core, NUMBER_FIX[core], 1))
            merged.append(w); i += 1
    for w in merged:
        bare = _bare(w["word"]).lower()
        if bare in PROPER_NOUNS:
            w["word"] = re.sub(re.escape(_bare(w["word"])), PROPER_NOUNS[bare], w["word"], count=1)
    return merged

def _bare(t):
    m = re.match(r"[\w']+", t)
    return m.group(0) if m else t

# ---------------- grouping ----------------
def group_words(words, cfg):
    cards, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        gap = (words[i+1]["start"] - w["end"]) if i + 1 < len(words) else 0.0
        if (len(cur) >= cfg["max_words"]
                or w["word"].rstrip("\"'").endswith(tuple(cfg["end_punct"]))
                or gap > cfg["gap_break"]):
            cards.append(cur); cur = []
    if cur:
        cards.append(cur)
    out = []
    for j, c in enumerate(cards):
        start, end = c[0]["start"], c[-1]["end"]
        nxt = cards[j+1][0]["start"] if j + 1 < len(cards) else end + cfg["extend_cap"]
        out.append({"words": c, "start": start,
                    "end": min(nxt, end + cfg["extend_cap"]),
                    "text": " ".join(w["word"] for w in c)})
    return out

# ---------------- ASS emit ----------------
FORMAT_LINE = ("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
               "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
               "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
               "MarginL, MarginR, MarginV, Encoding")
N_FIELDS = 23

def style_line(name, font, size, primary, secondary, outline_c, back, bold, italic,
               underline, strikeout, sx, sy, spacing, angle, borderstyle, outline,
               shadow, align, ml, mr, mv, enc=1):
    vals = [name, font, size, primary, secondary, outline_c, back, bold, italic,
            underline, strikeout, sx, sy, spacing, angle, borderstyle, outline,
            shadow, align, ml, mr, mv, enc]
    assert len(vals) == N_FIELDS, f"style field count {len(vals)} != {N_FIELDS}"
    return "Style: " + ",".join(str(v) for v in vals)

def ts(t):
    t = max(0.0, t)
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"

def header(styles):
    for s in styles:
        n = len(s.split(":", 1)[1].split(","))
        assert n == N_FIELDS, f"emitted style has {n} fields, need {N_FIELDS}: {s}"
    return ("[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
            "WrapStyle: 0\nScaledBorderAndShadow: yes\nYCbCr Matrix: TV.709\n\n"
            "[V4+ Styles]\n" + FORMAT_LINE + "\n" + "\n".join(styles) +
            "\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, "
            "MarginV, Effect, Text\n")

def preset_clean(cards, safe):
    mv = safe["bottom"] + 20  # +20: shadow/descender clearance (measured)
    ml, mr = max(90, safe["left"]), max(90, safe["right"])
    st = style_line("Cap", "DejaVu Sans", 62, "&H00FFFFFF", "&H00000000",
                    "&H00000000", "&H90000000", -1, 0, 0, 0, 100, 100, 0, 0,
                    1, 0, 4, 2, ml, mr, mv)
    ev = [f"Dialogue: 0,{ts(c['start'])},{ts(c['end'])},Cap,,0,0,0,,{c['text']}"
          for c in cards]
    return header([st]) + "\n".join(ev) + "\n"

def _fit_fs(text, base_fs, max_w, font_path):
    """Shrink fontsize so rendered width (real TTF metrics via PIL, +outline
    cushion) fits max_w. Never enlarges. PIL advance widths track libass within
    a few %, cushion covers the difference + \\bord."""
    from PIL import ImageFont
    f = ImageFont.truetype(font_path, base_fs)
    w = f.getlength(text) + 0.10 * base_fs * 2  # outline + AA cushion
    if w <= max_w:
        return base_fs
    return max(int(base_fs * max_w / w), int(base_fs * 0.55))

def preset_wordpop(cards, safe):
    # 1-2 word cards, heavy font, center \pos inside safe area, 80->100 pop over 80 ms
    y = 1920 - safe["bottom"] - 260  # block center, clear of bottom UI
    base = 150
    max_w = 2 * min(540 - safe["left"], 540 - safe["right"]) - 30
    fpath = os.path.join(D, "fonts", "Montserrat-ExtraBold.ttf")
    st = style_line("Pop", "Montserrat ExtraBold", base, "&H00FFFFFF", "&H00000000",
                    "&H00000000", "&H80000000", 0, 0, 0, 0, 100, 100, 0, 0,
                    1, 7, 2, 5, safe["left"], safe["right"], 0)
    ev = []
    for c in cards:
        txt = c["text"].upper()
        fs = _fit_fs(txt, base, max_w, fpath)
        fso = f"\\fs{fs}" if fs != base else ""
        ev.append(f"Dialogue: 0,{ts(c['start'])},{ts(c['end'])},Pop,,0,0,0,,"
                  f"{{\\pos(540,{y}){fso}\\fscx80\\fscy80\\t(0,80,\\fscx100\\fscy100)}}"
                  f"{txt}")
    return header([st]) + "\n".join(ev) + "\n"

def preset_karaoke(cards, safe):
    # 3-4 word phrase; active word yellow via per-word Dialogue events (not \k)
    mv = safe["bottom"] + 20
    ml, mr = max(90, safe["left"]), max(90, safe["right"])
    st = style_line("Kar", "Montserrat ExtraBold", 72, "&H00FFFFFF", "&H00000000",
                    "&H00000000", "&H80000000", 0, 0, 0, 0, 100, 100, 0, 0,
                    1, 5, 2, 2, ml, mr, mv)
    HL, WH = "{\\1c&H00FFFF&}", "{\\1c&HFFFFFF&}"  # BGR: yellow / white
    ev = []
    for c in cards:
        ws = c["words"]
        for i, w in enumerate(ws):
            t0 = w["start"] if i else c["start"]
            t1 = ws[i+1]["start"] if i + 1 < len(ws) else c["end"]
            if t1 - t0 < 0.01:
                continue
            txt = " ".join((HL + x["word"] + WH) if k == i else x["word"]
                           for k, x in enumerate(ws))
            ev.append(f"Dialogue: 0,{ts(t0)},{ts(t1)},Kar,,0,0,0,,{txt}")
    return header([st]) + "\n".join(ev) + "\n"

PRESETS = {"clean": (preset_clean, dict(max_words=4)),
           "wordpop": (preset_wordpop, dict(max_words=2)),
           "karaoke": (preset_karaoke, dict(max_words=4))}

def burn_cmd(video_in, ass, out, fontsdir=os.path.join(D, "fonts")):
    return (f"ffmpeg -y -i {video_in} -vf \"subtitles={ass}:fontsdir={fontsdir}\" "
            f"-c:a copy {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", help="whisper words JSON")
    ap.add_argument("--wav", help="run faster-whisper base.en on this audio")
    ap.add_argument("--preset", default="clean", choices=list(PRESETS))
    ap.add_argument("--platform", default=DEFAULTS["platform"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-words", type=int)
    ap.add_argument("--gap-break", type=float)
    ap.add_argument("--extend-cap", type=float)
    a = ap.parse_args()
    if a.wav:
        from faster_whisper import WhisperModel
        m = WhisperModel("base.en", device="cpu", compute_type="int8")
        segs, _ = m.transcribe(a.wav, word_timestamps=True, language="en")
        words = [{"word": w.word, "start": w.start, "end": w.end}
                 for s in segs for w in s.words]
    else:
        words = json.load(open(a.words))["words"]
    fn, over = PRESETS[a.preset]
    cfg = dict(DEFAULTS); cfg.update(over)
    for k in ("max_words", "gap_break", "extend_cap"):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    cards = group_words(clean_words(words), cfg)
    ass = fn(cards, load_safe(a.platform))
    open(a.out, "w").write(ass)
    print(f"{a.preset}: {len(cards)} cards -> {a.out}")
    print("burn:", burn_cmd("<in.mp4>", a.out, "<out.mp4>"))

if __name__ == "__main__":
    main()
