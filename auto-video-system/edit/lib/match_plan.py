#!/usr/bin/env python3.11
"""
match_plan.py -- the SLOT-MATCHING BRAIN of the edit engine.

Fills a reference-derived BLUEPRINT (ordered slots) with windows from a footage
CATALOG, guided by a VO plan (word timings -> per-slot keyword hints) and a
cut-profile (family weighting, e.g. "PORTRAIT-led"). Pure deterministic code:
no model calls, no randomness, no wall-clock. Same inputs -> byte-identical
edl.json.

Architecture (mirrors Write-A-Video [Wang et al., TOG 2019]: prune candidates
per text segment to top-K, then optimize a sequence energy with unary +
pairwise terms; and Leake et al. SIGGRAPH 2017: idioms as soft costs over a
label sequence solved by DP):

  1. Derive slot keywords from VO words overlapping each slot (+ lookahead).
  2. Build slot x window UNARY cost matrix; prune to top-K=8 per slot.
  3. Solve sequence with one of 4 solvers (greedy / hungarian / hungarian_dup /
     beam DP). All are scored by ONE true-objective evaluator that includes the
     sequential terms (diversity, continuity, reuse) Hungarian cannot see.
  4. Emit edl.json with one source window per slot + one-line rationale.

COST MODEL (lower = better). Weights chosen so intent (affinity) dominates
polish (quality): max quality gap costs W_Q*1.0 = 1.0, while a family miss
costs up to W_AFF*1.0 = 3.0 -> a gorgeous wrong-family shot loses to a decent
right-family shot (validated by test case B).

  unary(slot, window):
    infeasible        BIG        window shorter than slot.dur + 2*HANDLE (0.2s)
    affinity          W_AFF  * (1 - aff)   aff = 0.65*profile_family_weight
                                            + 0.35*keyword hit (VO words ->
                                            lexicon family / window tags)
    motion match      W_MOT  * |slot.motion - window.motion|      (both 0..1)
    shot-scale match  W_SCALE* ordinal_dist(MACRO,CU,MCU,MS,WS)/4
    quality           W_Q    * (1 - window.quality)
    hook bonus        W_HOOK * (1 - window.hero)     only slot.role == "hook"
    hook forbid       HOOK_FORBID if window.family in profile.forbid_hook
                      (brand rule: product never the hook/pitch)
    conclusion        W_CONC * (1 - is_conclusion_family)  only role ==
                      "conclusion" (brand rule: product appears as conclusion)
    conclusion forbid HOOK_FORBID if window.family in
                      profile.forbid_conclusion_families (brand rule, mirrors
                      hook forbid: e.g. STOCK must never win the conclusion)

  sequential (need knowledge of neighbours / whole sequence):
    diversity penalty  +P_DIV   adjacent slots cut from the same clip
    continuity bonus   -B_CONT  adjacent slots use CONSECUTIVE windows of one
                                clip AND both slots are a fast cut-run
                                (dur <= 0.9s) -> montage stays inside a move
    reuse penalty      +P_REUSE * (uses-1) per extra use of the same window

SOLVER CHOICE (see compare_solvers / tests): Hungarian optimizes the WRONG
objective: it enforces hard one-to-one use (reuse is legal in our domain, only
discouraged) and is structurally blind to the sequential terms above, because
linear assignment requires cost(slot,window) independent of other rows.
Duplicating columns (hungarian_dup) re-legalizes reuse with escalating cost but
still cannot see adjacency. The true objective is a chain -> DP. `beam` is a
Viterbi-style DP over the pruned top-K with the running usage multiset kept in
the beam state (exact for our sizes at width 64). beam wins or ties every test;
it is the default.

CLI:
  python3.11 match_plan.py --blueprint b.json --catalog c.json --vo vo.json \
      --profile p.json --out edl.json [--solver beam|greedy|hungarian|hungarian_dup]
"""

from __future__ import annotations
import argparse
import json
import sys

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False

# ----------------------------------------------------------------- constants
HANDLE = 0.2          # editorial handles either side of the slot cut, seconds
BIG = 1.0e6           # infeasible marker (finite: Hungarian-safe)
TOP_K = 8             # candidates kept per slot after unary prune
CUT_RUN_MAX = 0.9     # slot dur <= this on both sides of a cut => cut-run
BEAM_WIDTH = 64

W_AFF = 3.0           # setup/semantic affinity           (dominant: intent)
W_MOT = 1.5           # motion-energy match               (feel of the cut)
W_SCALE = 1.0         # shot-scale match                  (framing grammar)
W_Q = 1.0             # window quality score              (polish)
W_HOOK = 2.0          # hero-rank bonus on the hook slot  (strongest shot 1st)
HOOK_FORBID = 5.0     # brand rule: forbidden family in hook (soft-hard)
W_CONC = 1.5          # conclusion-family pull on the last-slot role
P_DIV = 0.8           # adjacent slots from same clip (non-continuity)
B_CONT = 1.2          # consecutive windows of one clip across a cut-run
P_REUSE = 2.5         # per extra use of the exact same window

SCALES = ["MACRO", "CU", "MCU", "MS", "WS"]

# VO word -> setup-family hints. Deterministic micro-lexicon; extend per brand.
LEXICON = {
    "PORTRAIT": ["i", "me", "my", "story", "why", "believe", "founder", "face",
                 "talk", "honest", "started", "person", "myself", "ask"],
    "DESK": ["print", "printer", "printing", "printed", "build", "built",
             "building", "making", "work", "working", "desk", "design",
             "designing", "computer", "fail", "failed", "night", "iterate"],
    "PRODUCT": ["shoe", "shoes", "sole", "flip", "flop", "product", "material",
                "texture", "detail", "macro", "layer", "layers", "rubber",
                "flex", "flexes"],
    "FEET": ["walk", "walking", "walked", "step", "steps", "wear", "wearing",
             "street", "ground", "feet"],
    "MOVEMENT": ["move", "moving", "moved", "go", "going", "run", "running",
                 "journey", "forward", "outside", "city", "kept"],
}


# ------------------------------------------------------------------- helpers
def _r(x: float) -> float:
    """Round for stable float text in JSON + stable comparisons."""
    return round(x + 0.0, 4)


def slot_keywords(slot: dict, vo_plan: dict) -> list[str]:
    """Words whose timing overlaps [t0, t0+dur+lookahead). B-roll illustrates
    what is being said (QuickCut aligns narration to footage the same way), so
    a small lookahead lets the image anticipate the line."""
    if not vo_plan:
        return []
    t0 = float(slot["t0"])
    t1 = t0 + float(slot["dur"]) + 0.35  # lookahead
    words = []
    for line in vo_plan.get("lines", []):
        for wd in line.get("words", []):
            if float(wd["t0"]) < t1 and float(wd["t1"]) > t0:
                words.append(str(wd["w"]).lower().strip(".,!?"))
    return sorted(set(words))


def keyword_family_hits(keywords: list[str]) -> dict[str, int]:
    hits: dict[str, int] = {}
    for fam, vocab in LEXICON.items():
        n = sum(1 for k in keywords if k in vocab)
        if n:
            hits[fam] = n
    return hits


def keyword_tag_score(keywords: list[str], tags: list[str]) -> float:
    """Fuzzy prefix match VO keywords vs window tags (stem-ish, len>=4)."""
    if not keywords or not tags:
        return 0.0
    tl = [t.lower() for t in tags]
    hit = 0
    for k in keywords:
        for t in tl:
            if k == t or (len(k) >= 4 and (t.startswith(k[:4]) or k.startswith(t[:4]))):
                hit += 1
                break
    return min(1.0, hit / 2.0)  # 2+ tag hits saturate


# ------------------------------------------------------------------ unary cost
def unary_cost(slot: dict, win: dict, profile: dict, kws: list[str]) -> tuple[float, str]:
    """Returns (cost, rationale fragment)."""
    dur_avail = float(win["t_out"]) - float(win["t_in"])
    need = float(slot["dur"]) + 2 * HANDLE
    if dur_avail + 1e-9 < need:
        return BIG, f"INFEASIBLE need {need:.2f}s have {dur_avail:.2f}s"

    bits = []

    # --- affinity: profile family weight + VO-keyword evidence
    fam = win.get("family", "UNKNOWN")
    base = float(profile.get("family_weights", {}).get(fam, 0.3))
    fam_hits = keyword_family_hits(kws)
    kw_fam = min(1.0, fam_hits.get(fam, 0) / 2.0)          # lexicon -> family
    kw_tag = keyword_tag_score(kws, win.get("tags", []))    # direct tag match
    kw_score = max(kw_fam, kw_tag)
    aff = min(1.0, 0.65 * base + 0.35 * kw_score)
    c_aff = W_AFF * (1.0 - aff)
    bits.append(f"aff{aff:.2f}(fam {fam} w{base:.2f} kw{kw_score:.2f})")

    # --- motion
    dm = abs(float(slot.get("motion", 0.5)) - float(win.get("motion", 0.5)))
    c_mot = W_MOT * dm
    bits.append(f"mot|d{dm:.2f}|")

    # --- shot scale
    s_sc, w_sc = slot.get("scale"), win.get("scale")
    if s_sc in SCALES and w_sc in SCALES:
        dist = abs(SCALES.index(s_sc) - SCALES.index(w_sc)) / (len(SCALES) - 1)
    else:
        dist = 0.5
    c_scale = W_SCALE * dist
    bits.append(f"scale {w_sc or '?'}vs{s_sc or '?'}d{dist:.2f}")

    # --- quality
    q = float(win.get("quality", 0.5))
    c_q = W_Q * (1.0 - q)
    bits.append(f"q{q:.2f}")

    # --- hook slot: hero shot demanded, forbidden families punished
    c_hook = 0.0
    if slot.get("role") == "hook":
        hero = float(win.get("hero", q))
        c_hook = W_HOOK * (1.0 - hero)
        bits.append(f"hook hero{hero:.2f}")
        if fam in profile.get("forbid_hook_families", []):
            c_hook += HOOK_FORBID
            bits.append("HOOK-FORBID")

    # --- conclusion slot: brand rule, product as conclusion; forbidden
    # families are hard-barred the same way forbid_hook_families bars the
    # hook slot (same mechanism, mirrored for the other brand-critical slot:
    # e.g. STOCK must never win hook OR conclusion).
    c_conc = 0.0
    conc_fam = profile.get("conclusion_family")
    if slot.get("role") == "conclusion":
        if conc_fam:
            match = 1.0 if fam == conc_fam else 0.0
            c_conc = W_CONC * (1.0 - match)
            if match:
                bits.append("concl-fam+")
        if fam in profile.get("forbid_conclusion_families", []):
            c_conc += HOOK_FORBID
            bits.append("CONCL-FORBID")

    total = c_aff + c_mot + c_scale + c_q + c_hook + c_conc
    return _r(total), " ".join(bits)


def is_cut_run(a: dict, b: dict) -> bool:
    return float(a["dur"]) <= CUT_RUN_MAX and float(b["dur"]) <= CUT_RUN_MAX


def pair_cost(prev_slot: dict, slot: dict, prev_win: dict, win: dict) -> tuple[float, str]:
    """Adjacency terms: diversity penalty vs continuity bonus."""
    if prev_win["clip"] == win["clip"]:
        if is_cut_run(prev_slot, slot) and \
                int(win.get("clip_seq", -9)) == int(prev_win.get("clip_seq", -99)) + 1:
            return -B_CONT, f"cont+(follows {prev_win['id']})"
        return P_DIV, f"div-(same clip as {prev_win['id']})"
    return 0.0, ""


# ------------------------------------------------------- objective evaluation
def true_cost(assign: list[int], slots: list[dict], wins: list[dict],
              U: list[list[float]]) -> float:
    """The single source of truth all solvers are judged by."""
    tot = sum(U[i][assign[i]] for i in range(len(slots)))
    for i in range(1, len(slots)):
        c, _ = pair_cost(slots[i - 1], slots[i], wins[assign[i - 1]], wins[assign[i]])
        tot += c
    seen: dict[int, int] = {}
    for j in assign:
        seen[j] = seen.get(j, 0) + 1
    tot += sum(P_REUSE * (n - 1) for n in seen.values() if n > 1)
    return _r(tot)


# ------------------------------------------------------------------- solvers
def prune_topk(U: list[list[float]], wins: list[dict], slots: list[dict],
               k: int = TOP_K) -> list[list[int]]:
    """Per slot: indices of k cheapest feasible windows (ties -> window id),
    PLUS chain closure: unary pruning is blind to the continuity bonus, so a
    montage chain's later links would die before the pairwise term could save
    them. For every cut-run transition i->i+1, the successor window
    (same clip, clip_seq+1) of each slot-i candidate is force-kept in slot
    i+1's candidate set if feasible. Deterministic."""
    out = []
    for row in U:
        idx = [j for j in range(len(row)) if row[j] < BIG]
        idx.sort(key=lambda j: (row[j], wins[j]["id"]))
        out.append(idx[:k])
    succ = {}  # (clip, seq) -> window index
    for j, w in enumerate(wins):
        succ[(w["clip"], int(w.get("clip_seq", -1)))] = j
    for i in range(len(slots) - 1):
        if not is_cut_run(slots[i], slots[i + 1]):
            continue
        add = []
        for j in out[i]:
            w = wins[j]
            nj = succ.get((w["clip"], int(w.get("clip_seq", -1)) + 1))
            if nj is not None and U[i + 1][nj] < BIG and nj not in out[i + 1]:
                add.append(nj)
        out[i + 1] = out[i + 1] + sorted(set(add), key=lambda j: wins[j]["id"])
    return out


def solve_greedy(slots, wins, U, cand) -> list[int]:
    assign: list[int] = []
    used: dict[int, int] = {}
    for i in range(len(slots)):
        best, best_key = None, None
        for j in cand[i]:
            c = U[i][j] + P_REUSE * used.get(j, 0)
            if i > 0:
                pc, _ = pair_cost(slots[i - 1], slots[i], wins[assign[i - 1]], wins[j])
                c += pc
            key = (_r(c), wins[j]["id"])           # deterministic tie-break
            if best_key is None or key < best_key:
                best, best_key = j, key
        if best is None:
            raise RuntimeError(f"slot {i}: no feasible candidate")
        assign.append(best)
        used[best] = used.get(best, 0) + 1
    return assign


def solve_hungarian(slots, wins, U, cand) -> list[int]:
    """Classic one-to-one assignment on the unary matrix. Structurally cannot
    model reuse (hard-forbidden) or adjacency terms (rows must be independent).
    Kept as a benchmark to show WHY it is the wrong tool."""
    if not HAVE_SCIPY:
        raise RuntimeError("scipy missing")
    S, N = len(slots), len(wins)
    if N < S:
        raise RuntimeError("hungarian: fewer windows than slots")
    M = [[U[i][j] for j in range(N)] for i in range(S)]
    rows, cols = linear_sum_assignment(M)
    assign = [0] * S
    for r, c in zip(rows.tolist(), cols.tolist()):
        assign[r] = c
    return assign


def solve_hungarian_dup(slots, wins, U, cand, repeats: int = 2) -> list[int]:
    """Reuse-tolerant Hungarian: each window duplicated `repeats` times, k-th
    copy priced +k*P_REUSE. Fixes the hard no-reuse constraint, still blind to
    diversity/continuity adjacency."""
    if not HAVE_SCIPY:
        raise RuntimeError("scipy missing")
    S, N = len(slots), len(wins)
    cols = []  # (win_idx, copy_k)
    for j in range(N):
        for k in range(repeats):
            cols.append((j, k))
    M = [[U[i][j] + P_REUSE * k for (j, k) in cols] for i in range(S)]
    rows, cids = linear_sum_assignment(M)
    assign = [0] * S
    for r, c in zip(rows.tolist(), cids.tolist()):
        assign[r] = cols[c][0]
    return assign


def solve_beam(slots, wins, U, cand, width: int = BEAM_WIDTH) -> list[int]:
    """Viterbi-style DP over pruned candidates; beam state carries the usage
    multiset so the global reuse penalty is exact inside the beam. With
    K=8 and width 64 this explores the full product space for typical reels
    (K^2=64 transitions per step) -> effectively exact. Deterministic order."""
    # state: (cost, path tuple)
    beams: list[tuple[float, tuple[int, ...]]] = [(0.0, ())]
    for i in range(len(slots)):
        nxt: dict[tuple[int, ...], float] = {}
        for cost, path in beams:
            used: dict[int, int] = {}
            for j in path:
                used[j] = used.get(j, 0) + 1
            for j in cand[i]:
                c = cost + U[i][j] + P_REUSE * used.get(j, 0)
                if path:
                    pc, _ = pair_cost(slots[i - 1], slots[i], wins[path[-1]], wins[j])
                    c += pc
                p2 = path + (j,)
                if p2 not in nxt or c < nxt[p2]:
                    nxt[p2] = _r(c)
        if not nxt:
            raise RuntimeError(f"slot {i}: beam empty (no feasible candidate)")
        ranked = sorted(nxt.items(), key=lambda kv: (kv[1], tuple(wins[j]["id"] for j in kv[0])))
        beams = [(c, p) for p, c in ranked[:width]]
    best_path = beams[0][1]
    return list(best_path)


SOLVERS = {
    "greedy": solve_greedy,
    "hungarian": solve_hungarian,
    "hungarian_dup": solve_hungarian_dup,
    "beam": solve_beam,
}


# ------------------------------------------------------------------ pipeline
def build_unary(slots, wins, profile, vo_plan):
    U, R = [], []
    for s in slots:
        kws = slot_keywords(s, vo_plan)
        row, rrow = [], []
        for w in wins:
            c, why = unary_cost(s, w, profile, kws)
            row.append(c)
            rrow.append(why)
        U.append(row)
        R.append(rrow)
    return U, R


def make_edl(blueprint, catalog, vo_plan, profile, solver_name="beam"):
    slots = blueprint["slots"]
    wins = catalog["windows"]
    U, R = build_unary(slots, wins, profile, vo_plan)
    cand = prune_topk(U, wins, slots)
    assign = SOLVERS[solver_name](slots, wins, U, cand)
    tc = true_cost(assign, slots, wins, U)

    events = []
    for i, j in enumerate(assign):
        w, s = wins[j], slots[i]
        src_in = _r(float(w["t_in"]) + HANDLE)
        src_out = _r(src_in + float(s["dur"]))
        rat = R[i][j]
        if i > 0:
            pc, note = pair_cost(slots[i - 1], s, wins[assign[i - 1]], w)
            if note:
                rat += " " + note
        n_use = assign[:i + 1].count(j)
        if n_use > 1:
            rat += f" reuse#{n_use}"
        events.append({
            "slot": int(s["id"]),
            "t0": _r(float(s["t0"])),
            "dur": _r(float(s["dur"])),
            "window": w["id"],
            "clip": w["clip"],
            "family": w.get("family", "UNKNOWN"),
            "src_in": src_in,
            "src_out": src_out,
            "unary_cost": _r(U[i][j]),
            "rationale": rat,
        })

    edl = {
        "meta": {
            "blueprint": blueprint.get("meta", {}).get("ref_id", "unknown"),
            "shoot": catalog.get("meta", {}).get("shoot_id", "unknown"),
            "profile": profile.get("name", "unnamed"),
            "solver": solver_name,
            "true_cost": tc,
            "weights": {"W_AFF": W_AFF, "W_MOT": W_MOT, "W_SCALE": W_SCALE,
                        "W_Q": W_Q, "W_HOOK": W_HOOK, "HOOK_FORBID": HOOK_FORBID,
                        "W_CONC": W_CONC, "P_DIV": P_DIV, "B_CONT": B_CONT,
                        "P_REUSE": P_REUSE, "HANDLE": HANDLE, "TOP_K": TOP_K},
        },
        "events": events,
    }
    return edl, U, cand, assign


def compare_solvers(blueprint, catalog, vo_plan, profile):
    slots, wins = blueprint["slots"], catalog["windows"]
    U, _ = build_unary(slots, wins, profile, vo_plan)
    cand = prune_topk(U, wins, slots)
    out = {}
    for name, fn in SOLVERS.items():
        try:
            a = fn(slots, wins, U, cand)
            out[name] = {"assign": [wins[j]["id"] for j in a],
                         "true_cost": true_cost(a, slots, wins, U)}
        except RuntimeError as e:
            out[name] = {"error": str(e)}
    return out


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, sort_keys=True, indent=1, separators=(",", ": "))
        f.write("\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--blueprint", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--vo", default=None)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--solver", default="beam", choices=sorted(SOLVERS))
    a = ap.parse_args(argv)
    bp = json.load(open(a.blueprint))
    cat = json.load(open(a.catalog))
    vo = json.load(open(a.vo)) if a.vo else {}
    prof = json.load(open(a.profile))
    edl, _, _, _ = make_edl(bp, cat, vo, prof, a.solver)
    dump_json(edl, a.out)
    print(f"wrote {a.out} solver={a.solver} true_cost={edl['meta']['true_cost']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
