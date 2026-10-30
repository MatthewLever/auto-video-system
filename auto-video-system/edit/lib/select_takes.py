#!/usr/bin/env python3
"""select_takes.py — zero-model-call VO take selection for the edit engine.
Input: VO files (m4a/wav) + script lines. Chain: silencedetect -> per-segment faster-whisper
(word timestamps) -> fuzzy align each script line to word spans -> score -> vo_selects.json.
Loudness is NOT scored (correctable): per-pick gain_db targets -18 LUFS.
"""
import subprocess, json, re, os, sys, time, argparse, tempfile
import numpy as np
from rapidfuzz import fuzz

SIL_N, SIL_D = "-32dB", 0.45          # proven silencedetect params
PAD_IN, PAD_OUT = 0.12, 0.18          # proven edge padding
TARGET_LUFS = -18.0
FILLERS = {"um", "uh", "er", "erm", "hmm", "mmm", "ah", "eh"}
WPS = 3.0                             # expected words/sec for duration sanity

def sh(cmd):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True)

def silence_segments(path):
    r = sh(f'ffmpeg -nostdin -hide_banner -i "{path}" -af silencedetect=n={SIL_N}:d={SIL_D} -f null -')
    dur = float(sh(f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{path}"').stdout.strip())
    ev = re.findall(r"silence_(start|end): ([\d.]+)", r.stderr)
    bounds, cur = [], 0.0
    for kind, t in ev:
        t = float(t)
        if kind == "start":
            if t - cur > 0.30: bounds.append((cur, t))
        else: cur = t
    if dur - cur > 0.30: bounds.append((cur, dur))
    return [(max(0.0, s - PAD_IN), min(dur, e + PAD_OUT)) for s, e in bounds]

def seg_loudness(path, s, e):
    r = sh(f'ffmpeg -nostdin -hide_banner -ss {s:.3f} -to {e:.3f} -i "{path}" -af ebur128 -f null -')
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    if m and float(m[-1]) > -70: return float(m[-1])
    r = sh(f'ffmpeg -nostdin -hide_banner -ss {s:.3f} -to {e:.3f} -i "{path}" '
           f'-af astats=measure_overall=RMS_level:measure_perchannel=none -f null -')
    m = re.findall(r"RMS level dB:\s*(-?[\d.]+)", r.stderr)
    return float(m[-1]) if m else -30.0

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()

def _num_words(n):
    if n < 20: return [_ONES[n]]
    if n < 100: return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (_num_words(n % 100) if n % 100 else [])
    return [str(n)]

def norm_tokens(text):
    out = []
    for w in text.lower().split():
        dollar = "$" in w
        m = re.fullmatch(r"\$?(\d+)[.,!?]*", w)
        if m:
            out += _num_words(int(m.group(1)))
            if dollar: out.append("dollars")
            continue
        t = re.sub(r"[^a-z']", "", w)
        if not t: continue
        out += _CONTR.get(t, [t])
    return out

_CONTR = {"i'm": ["i", "am"], "i've": ["i", "have"], "i'll": ["i", "will"],
          "don't": ["do", "not"], "can't": ["can", "not"], "won't": ["will", "not"],
          "it's": ["it", "is"], "that's": ["that", "is"], "we're": ["we", "are"],
          "you're": ["you", "are"], "they're": ["they", "are"], "gonna": ["going", "to"],
          "gotta": ["got", "to"], "wanna": ["want", "to"], "isn't": ["is", "not"],
          "didn't": ["did", "not"], "doesn't": ["does", "not"], "there's": ["there", "is"]}

def transcribe_segments(model, path, segs, tmpdir):
    out = []
    for i, (s, e) in enumerate(segs):
        wav = os.path.join(tmpdir, f"seg{i}.wav")
        sh(f'ffmpeg -nostdin -y -ss {s:.3f} -to {e:.3f} -i "{path}" -ar 16000 -ac 1 "{wav}"')
        segments, _ = model.transcribe(wav, word_timestamps=True, condition_on_previous_text=False, beam_size=5)
        words = []
        for seg in segments:
            for w in seg.words:
                words.append(dict(w=w.word.strip(), t0=round(s + w.start, 2), t1=round(s + w.end, 2),
                                  p=round(w.probability, 3), lp=round(seg.avg_logprob, 3)))
        lufs = seg_loudness(path, s, e)
        out.append(dict(i=i, t=(round(s, 2), round(e, 2)), words=words, lufs=round(lufs, 1),
                        text=" ".join(w["w"] for w in words)))
    return out

def disfluencies(tokens):
    n = 0
    for j, t in enumerate(tokens):
        if t in FILLERS: n += 1
        if j and t == tokens[j - 1] and t not in FILLERS: n += 1   # immediate word repeat
    return n

def align_line(line, seg):
    """Best word-span in segment for a script line. Returns (ratio, i0, i1, span metrics)."""
    ltok = norm_tokens(line); k = len(ltok)
    stok_full = [norm_tokens(w["w"]) for w in seg["words"]]
    stok = [t[0] if t else "" for t in stok_full]
    best = None
    n_lo = max(1, min(k - 2, len(stok)))
    n_hi = min(k + 4, len(stok) + 1)
    for n in range(n_lo, n_hi):
        for i0 in range(0, len(stok) - n + 1):
            cand = [t for ts in stok_full[i0:i0 + n] for t in ts]
            r = fuzz.ratio(" ".join(ltok), " ".join(cand)) / 100.0
            if best is None or r > best[0]: best = (r, i0, i0 + n)
    if best is None: return None
    r, i0, i1 = best
    words = seg["words"][i0:i1]
    span_tok = stok[i0:i1]
    d_in = disfluencies(span_tok)
    # ratio-optimal spans dodge flanking stutters/fillers: count boundary-adjacent ones too
    if i0 > 0 and stok[i0 - 1] and (stok[i0 - 1] in FILLERS or stok[i0 - 1] == span_tok[0]) \
            and words and seg["words"][i0 - 1]["t1"] > words[0]["t0"] - 0.4: d_in += 1
    if i1 < len(stok) and stok[i1] and (stok[i1] in FILLERS or stok[i1] == span_tok[-1]) \
            and words and seg["words"][i1]["t0"] < words[-1]["t1"] + 0.4: d_in += 1
    d_seg = disfluencies(stok)
    lp = float(np.mean([w["lp"] for w in words])) if words else -1.0
    dur = words[-1]["t1"] - words[0]["t0"] if words else 0.0
    exp = len(ltok) / WPS
    durpen = max(0.0, abs(dur - exp) / max(exp, 0.5) - 0.35)
    return dict(ratio=round(r, 3), t0=float(words[0]["t0"]) if words else None, t1=float(words[-1]["t1"]) if words else None,
                lp=round(lp, 3), disfl=d_in, disfl_seg=d_seg, durpen=round(durpen, 3), dur=round(dur, 2))

def score(al, take_disfl_out=0):
    logp_n = float(np.clip((al["lp"] + 1.0) / 0.8, 0, 1))
    return round(al["ratio"] + 0.25 * logp_n - 0.15 * al["disfl"]
                 - 0.05 * take_disfl_out - 0.10 * al["durpen"], 3)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vo", nargs="+", required=True); ap.add_argument("--script", required=True)
    ap.add_argument("--out", default="vo_selects.json"); ap.add_argument("--model", default="base.en")
    ap.add_argument("--debug", default=None)
    a = ap.parse_args()
    lines = [l.strip() for l in open(a.script) if l.strip()]
    from faster_whisper import WhisperModel
    t0 = time.time()
    model = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=2)
    allsegs = []
    with tempfile.TemporaryDirectory() as td:
        for f in a.vo:
            segs = silence_segments(f)
            print(f"{os.path.basename(f)}: {len(segs)} speech segments {[ (round(s,1),round(e,1)) for s,e in segs ]}")
            for seg in transcribe_segments(model, f, segs, td):
                seg["f"] = os.path.basename(f); allsegs.append(seg)
    # cluster segments into TAKES: same file, gap < 1.0s => same take (reads are separated by ~2s silence)
    take_id, last = -1, None
    for seg in allsegs:
        if last is None or seg["f"] != last["f"] or seg["t"][0] - last["t"][1] > 1.0: take_id += 1
        seg["take"] = take_id; last = seg
    take_disfl = {}
    for seg in allsegs:
        toks = [t for w in seg["words"] for t in norm_tokens(w["w"])]
        take_disfl[seg["take"]] = take_disfl.get(seg["take"], 0) + disfluencies(toks)
    picks, dbg = [], []
    for li, line in enumerate(lines):
        cands = []
        for seg in allsegs:
            al = align_line(line, seg)
            if al is None: continue
            spill = max(0, take_disfl[seg["take"]] - al["disfl"])   # flaws elsewhere in same take
            al.update(f=seg["f"], seg=seg["i"], take=seg["take"], lufs=seg["lufs"],
                      score=score(al, spill), gain=round(TARGET_LUFS - seg["lufs"], 1))
            cands.append(al)
        cands.sort(key=lambda c: (-c["score"], c["seg"]))
        if not cands:
            raise SystemExit(
                "line %d %r: no candidate take found. Heard segments:\n%s" %
                (li, line[:60], "\n".join("  %s#%d %r" % (s2["f"], s2["i"],
                 " ".join(w["w"] for w in s2["words"])[:90]) for s2 in allsegs)))
        p = cands[0]
        picks.append({"line": li, "text": line,
                      "pick": {"f": p["f"], "seg": p["seg"], "t": [p["t0"], p["t1"]], "gain": p["gain"],
                               "score": p["score"], "ratio": p["ratio"], "disfl": p["disfl"]},
                      "alts": [{"f": c["f"], "seg": c["seg"], "t": [c["t0"], c["t1"]], "gain": c["gain"],
                                "score": c["score"]} for c in cands[1:3]]})
        dbg.append({"line": li, "cands": cands})
    out = {"target_lufs": TARGET_LUFS, "model": a.model, "lines": picks}
    with open(a.out, "w") as f: json.dump(out, f, separators=(",", ":"))
    if a.debug:
        with open(a.debug, "w") as f: json.dump(dbg, f, indent=1)
    print(f"wrote {a.out} ({time.time()-t0:.1f}s total incl model load)")
    for p in picks:
        print(f"L{p['line']} -> {p['pick']['f']} seg{p['pick']['seg']} t={p['pick']['t']} "
              f"gain={p['pick']['gain']:+.1f}dB score={p['pick']['score']} (alts: "
              + ", ".join(f"seg{a_['seg']}:{a_['score']}" for a_ in p["alts"]) + ")")

if __name__ == "__main__":
    main()
