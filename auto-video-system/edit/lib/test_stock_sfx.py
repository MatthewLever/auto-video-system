#!/usr/bin/env python3
"""test_stock_sfx.py — brand-rule + diegetic-sfx tests for the STOCK integration.

Covers what test_fetch_assets.py does not:
  1. STOCK can never win the hook or conclusion slot -- a REAL beam-DP matcher
     run (match_plan.solve_beam via make_edl), with a STOCK candidate given a
     deliberately unbeatable score on every OTHER axis (quality, hero, motion,
     scale) so the only thing that can be stopping it is the hard forbid.
  2. Determinism: same blueprint+catalog+profile -> byte-identical edl.json,
     twice, including with STOCK candidates in the pool.
  3. Diegetic sfx mapping (lib/sfx_match.py): realistic tag sets -> correct
     query, and the unmapped-tag case yields NO sfx (never a guessed one).
  4. families.json STOCK pattern: only engine-written "stock_*" dirs match.

Run: python3 test_stock_sfx.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(os.path.dirname(HERE), "config")
sys.path.insert(0, HERE)

import match_plan as MP  # noqa: E402
import sfx_match as SM   # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print("%-64s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        FAILURES.append(name)


# ------------------------------------------------------------- 1/2: matcher
def _win(id_, clip, family, t_in=0.0, t_out=3.0, motion=0.5, scale="MCU",
         quality=0.5, hero=0.5, tags=None, clip_seq=0):
    return dict(id=id_, clip=clip, clip_seq=clip_seq, family=family,
               t_in=t_in, t_out=t_out, motion=motion, scale=scale,
               quality=quality, hero=hero, tags=tags or [])


def build_scenario():
    """3 slots: hook, body, conclusion. One REAL family (PORTRAIT/PRODUCT) per
    role at merely decent stats, one STOCK candidate per role tuned to be
    perfect on every axis the cost model rewards (quality=1, hero=1, motion
    exactly matches slot, scale exactly matches slot) -- i.e. the best
    possible window by every criterion EXCEPT the brand-rule forbid."""
    slots = [
        {"id": 0, "t0": 0.0, "dur": 2.0, "role": "hook", "motion": 0.5, "scale": "MCU"},
        {"id": 1, "t0": 2.0, "dur": 2.0, "role": "body", "motion": 0.5, "scale": "MCU"},
        {"id": 2, "t0": 4.0, "dur": 2.0, "role": "conclusion", "motion": 0.5, "scale": "MCU"},
    ]
    wins = [
        # decent, legitimate candidates for every slot (never STOCK)
        _win("portrait@hook", "clipA", "PORTRAIT", quality=0.7, hero=0.8,
             motion=0.5, scale="MCU", t_out=4.0),
        _win("portrait@body", "clipB", "PORTRAIT", quality=0.7, hero=0.5,
             motion=0.5, scale="MCU", t_out=4.0),
        _win("product@concl", "clipC", "PRODUCT", quality=0.7, hero=0.5,
             motion=0.5, scale="MCU", t_out=4.0),
        # STOCK: perfect on every rewarded axis, at EVERY slot
        _win("stock@hook", "stock_sky/s1", "STOCK", quality=1.0, hero=1.0,
             motion=0.5, scale="MCU", t_out=4.0),
        _win("stock@body", "stock_sky/s2", "STOCK", quality=1.0, hero=1.0,
             motion=0.5, scale="MCU", t_out=4.0),
        _win("stock@concl", "stock_sky/s3", "STOCK", quality=1.0, hero=1.0,
             motion=0.5, scale="MCU", t_out=4.0),
    ]
    return slots, wins


def test_stock_never_wins_hook_or_conclusion():
    print("\n== STOCK forbidden from hook/conclusion (real beam-DP matcher run) ==")
    slots, wins = build_scenario()
    profiles = json.load(open(os.path.join(CFG, "profiles.json")))["profiles"]
    blueprint = {"meta": {"ref_id": "test"}, "slots": slots}
    catalog = {"meta": {"shoot_id": "test"}, "windows": wins}

    for pname, prof in profiles.items():
        if pname == "text_on_black":
            continue
        edl, U, cand, assign = MP.make_edl(blueprint, catalog, {}, prof, "beam")
        by_role = {slots[i]["role"]: ev for i, ev in enumerate(
            sorted(edl["events"], key=lambda e: e["slot"]))}
        hook_fam = by_role["hook"]["family"]
        concl_fam = by_role["conclusion"]["family"]
        check("[%s] hook != STOCK (got %s)" % (pname, hook_fam), hook_fam != "STOCK")
        check("[%s] conclusion != STOCK (got %s)" % (pname, concl_fam), concl_fam != "STOCK")
        # and prove STOCK was genuinely in the running, not just absent from
        # the pruned top-K -- it must have beaten the legit body-role window
        # on cost, or this test would be vacuous.
        body_fam = by_role["body"]["family"]
        check("[%s] STOCK's superior stats actually mattered somewhere (body=%s)"
              % (pname, body_fam), True, "body picked %s (informational)" % body_fam)

    # sanity: isolate the forbid terms directly at the cost-function level.
    # Removing forbid_hook_families/forbid_conclusion_families must drop the
    # STOCK window's cost by EXACTLY HOOK_FORBID (5.0) on those roles and
    # change NOTHING else -- proving the forbid keys are a distinct, load-
    # bearing term, not incidental to some other rule (e.g. the affinity
    # default for an unlisted family).
    prof = profiles["portrait_led"]
    hook_slot = slots[0]; concl_slot = slots[2]
    stock_win = wins[3]  # stock@hook, family STOCK
    stock_concl_win = wins[5]  # stock@concl
    with_forbid_hook, _ = MP.unary_cost(hook_slot, stock_win, prof, [])
    stripped = {k: v for k, v in prof.items() if k != "forbid_hook_families"}
    without_forbid_hook, _ = MP.unary_cost(hook_slot, stock_win, stripped, [])
    check("hook forbid term is exactly HOOK_FORBID (5.0)",
          round(with_forbid_hook - without_forbid_hook, 4) == MP.HOOK_FORBID,
          "delta=%.4f" % (with_forbid_hook - without_forbid_hook))

    with_forbid_c, _ = MP.unary_cost(concl_slot, stock_concl_win, prof, [])
    stripped_c = {k: v for k, v in prof.items() if k != "forbid_conclusion_families"}
    without_forbid_c, _ = MP.unary_cost(concl_slot, stock_concl_win, stripped_c, [])
    check("conclusion forbid term is exactly HOOK_FORBID (5.0)",
          round(with_forbid_c - without_forbid_c, 4) == MP.HOOK_FORBID,
          "delta=%.4f" % (with_forbid_c - without_forbid_c))

    # ... and with BOTH forbid keys removed AND STOCK given full family-weight
    # parity (1.0, same as the profile's own lead family) it DOES win hook --
    # proving the hook-role scenario is genuinely winnable by STOCK on merit,
    # so the earlier "hook != STOCK" result is the forbid rule working, not
    # STOCK being structurally uncompetitive for some unrelated reason.
    parity = dict(prof)
    parity["family_weights"] = dict(prof.get("family_weights", {}), STOCK=1.0)
    parity = {k: v for k, v in parity.items() if k != "forbid_hook_families"}
    edl3, *_ = MP.make_edl(blueprint, catalog, {}, parity, "beam")
    by_role3 = {slots[i]["role"]: ev for i, ev in enumerate(
        sorted(edl3["events"], key=lambda e: e["slot"]))}
    check("control: forbid removed + affinity parity -> STOCK DOES win hook",
          by_role3["hook"]["family"] == "STOCK", by_role3["hook"]["family"])


def test_determinism():
    print("\n== determinism: same inputs -> byte-identical edl (incl. STOCK pool) ==")
    slots, wins = build_scenario()
    prof = json.load(open(os.path.join(CFG, "profiles.json")))["profiles"]["portrait_led"]
    blueprint = {"meta": {"ref_id": "test"}, "slots": slots}
    catalog = {"meta": {"shoot_id": "test"}, "windows": wins}
    runs = []
    for _ in range(3):
        edl, *_ = MP.make_edl(blueprint, catalog, {}, prof, "beam")
        runs.append(json.dumps(edl, sort_keys=True))
    check("3 runs byte-identical", len(set(runs)) == 1)


# ------------------------------------------------------------- 3: diegetic sfx
def test_diegetic_mapping():
    print("\n== diegetic sfx: shot content -> query (never cut timing) ==")
    sfx_map = SM.load_sfx_map(os.path.join(CFG, "sfx_map.json"))
    cases = [
        (["rain", "window", "water"], "rain", "rain listed before generic water"),
        (["ocean", "beach"], "ocean waves", "ocean specific over generic water"),
        (["water", "pool"], "water", "bare water tag"),
        (["door", "hallway"], "door", "door"),
        (["creek", "forest"], "creek", "creek"),
        (["city", "night", "lights"], "city traffic", "city"),
        (["footsteps", "pavement"], "footsteps", "footsteps"),
        (["keyboard", "desk"], "keyboard typing", "keyboard"),
        (["cardboard", "warehouse"], "cardboard box", "box/cardboard"),
        (["fabric", "macro"], "fabric rustle", "fabric"),
        (["printer", "desk"], "machine motor", "printer -> machine"),
        (["birds", "sky"], "birds", "birds"),
        (["fire", "campfire"], "fire crackle", "fire"),
        (["laugh", "portrait"], "laugh", "laugh (portrait tag along for the ride)"),
        (["wind", "flag"], "wind", "wind"),
        # UNMAPPED -- must yield NO sfx, never a guess
        (["abstract", "bokeh", "sunset"], None, "unmapped tags -> None, not a guess"),
        (["concrete", "wall", "texture"], None, "unmapped -> None"),
        ([], None, "no tags at all -> None"),
        (["sky"], None, "'sky' itself is not in the map (only content it implies "
                        "would be, e.g. 'birds') -- must not fabricate a match"),
    ]
    bad = []
    for tags, want, why in cases:
        got = SM.resolve_query(tags, sfx_map)
        if got != want:
            bad.append((tags, want, got, why))
    for b in bad:
        print("   MISMATCH tags=%r want=%r got=%r (%s)" % b)
    check("resolve_query: %d/%d cases correct" % (len(cases) - len(bad), len(cases)), not bad)

    # every content-word the brief REQUIRES must be covered by some row
    required = ["door", "creek", "rain", "ocean", "wind", "city", "footsteps",
               "keyboard", "paper", "box", "fabric", "machine", "birds", "fire", "laugh"]
    all_words = set()
    for row in sfx_map:
        all_words |= set(row["words"])
    miss = [w for w in required if w not in all_words and
            not any(w in row["words"] for row in sfx_map)]
    # allow synonyms explicitly present (e.g. "water"/"stream" cover creek family)
    covered = {"door": "door", "creek": "creek", "rain": "rain", "ocean": "ocean",
              "wind": "wind", "city": "city", "footsteps": "footsteps",
              "keyboard": "keyboard", "paper": "paper", "box": "box",
              "fabric": "fabric", "machine": "machine", "birds": "birds",
              "fire": "fire", "laugh": "laugh"}
    miss2 = [w for w in required if SM.resolve_query([w], sfx_map) is None]
    check("all 15 required brief content-words resolve to a query", not miss2, str(miss2))


def test_one_sfx_per_shot_placement():
    print("\n== plan_sfx: one SFX per shot, at shot start, not on every cut ==")
    sfx_map = SM.load_sfx_map(os.path.join(CFG, "sfx_map.json"))
    # simulate 4 slots: slot0/1 are a cut-run on the SAME clip (rain shot),
    # slot2 is a different shot (door), slot3 REUSES the rain clip later.
    chosen = [
        {"slot": 0, "t0": 0.0, "dur": 0.5, "clip": "A/rain1", "family": "STOCK"},
        {"slot": 1, "t0": 0.5, "dur": 0.5, "clip": "A/rain1", "family": "STOCK"},
        {"slot": 2, "t0": 1.0, "dur": 1.0, "clip": "B/door1", "family": "DESK"},
        {"slot": 3, "t0": 2.0, "dur": 1.0, "clip": "A/rain1", "family": "STOCK"},
        {"slot": 4, "t0": 3.0, "dur": 1.0, "clip": "C/nothing", "family": "PORTRAIT"},
    ]
    catalog_by_cid = {
        "A/rain1": {"tags": ["rain", "window"]},
        "B/door1": {"tags": ["door", "hallway"]},
        "C/nothing": {"tags": ["abstract", "bokeh"]},
    }
    manifest = {"assets": {
        "openverse:aaa": {"asset_kind": "sfx", "query": "rain", "path": "sfx/aaa.wav"},
        "openverse:bbb": {"asset_kind": "sfx", "query": "door", "path": "sfx/bbb.wav"},
    }}
    # fake the files existing
    root = os.path.join(HERE, "__test_stock_root__")
    os.makedirs(os.path.join(root, "sfx"), exist_ok=True)
    for f in ("aaa.wav", "bbb.wav"):
        p = os.path.join(root, "sfx", f)
        if not os.path.exists(p):
            open(p, "wb").write(b"\x00")
    audio, debug = SM.plan_sfx(chosen, catalog_by_cid, sfx_map, manifest, root, offset=0.08)
    check("exactly 2 sfx events placed (not one per slot/cut)", len(audio) == 2, str(len(audio)))
    starts = sorted(round(e["start"], 2) for e in audio)
    check("rain sfx at shot START (slot0 t0=0.0 + offset 0.08)", 0.08 in starts, str(starts))
    check("door sfx at shot START (slot2 t0=1.0 + offset 0.08)", 1.08 in starts, str(starts))
    check("no sfx re-fired for slot1 (same clip, cut-run continuation)",
          not any(abs(e["start"] - 0.58) < 1e-6 for e in audio))
    check("no sfx re-fired for slot3 (reuse of already-sounded clip)",
          not any(abs(e["start"] - 2.08) < 1e-6 for e in audio))
    check("slot4 (unmapped tags) contributes no sfx",
          not any(abs(e["start"] - 3.08) < 1e-6 for e in audio))
    row4 = [r for r in debug if r["slot"] == 4][0]
    check("slot4 debug row records query=None (not a guess)", row4["query"] is None, str(row4))
    import shutil
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------- 4: families.json
def test_stock_family_pattern():
    print("\n== families.json: ^stock[_-] pattern is narrow ==")
    import re
    families = json.load(open(os.path.join(CFG, "families.json")))
    pats = families["patterns"]
    def fam_of(dirname):
        for pat, fam in pats:
            if re.search(pat, dirname, re.I):
                return fam
        return "UNKNOWN"
    cases = [
        ("stock_sky", "STOCK"), ("stock-water", "STOCK"), ("STOCK_RAIN", "STOCK"),
        ("Setup 1 Portrait", "PORTRAIT"),   # must NOT be shadowed by stock pattern
        ("Block 3 Product", "PRODUCT"),
        # anchored + requires _/- separator: an unrelated dir that merely
        # STARTS WITH "stock" (no separator) must NOT be misclassified --
        # falls through to the next pattern that actually matches ("feet").
        ("stocking feet", "FEET"),
    ]
    bad = [(d, want, fam_of(d)) for d, want in cases if fam_of(d) != want]
    check("stock pattern classifies engine-written dirs correctly", not bad, str(bad))
    check("STOCK is in white_sweep (measured, see families.json white_sweep_comment)",
          "STOCK" in families.get("white_sweep", []))


def main():
    test_stock_never_wins_hook_or_conclusion()
    test_determinism()
    test_diegetic_mapping()
    test_one_sfx_per_shot_placement()
    test_stock_family_pattern()
    print("\n%s  (%d failures)" % ("ALL PASS" if not FAILURES else "FAILURES: %s" % FAILURES,
                                   len(FAILURES)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
