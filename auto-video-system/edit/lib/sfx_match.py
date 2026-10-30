#!/usr/bin/env python3
"""sfx_match.py — DIEGETIC sfx selection: shot content -> sound, never cut timing.

His explicit ask: "a door opening, a creek, a laugh" -- a sound that matches
what is ON SCREEN. NOT a transition whoosh at a cut point. So this module never
looks at cut positions; it only ever looks at a shot's own content words:

  * his own footage        -> catalog.json "tags" (a human adds these once per
                               shoot; see the edit-engine skill's semantic-tag step)
  * fetched Pexels stock    -> the category it was fetched under, injected as a
                               tag by `engine.py stock` (config/families.json's
                               ^stock[_-] pattern is how those clips even reach
                               the catalog at all -- same path, not a parallel one)

The mapping (config/sfx_map.json) is a flat, auditable table on purpose: every
content-word this engine will ever attach a sound to is visible in one file,
in the order it is checked. `resolve_query` returns None -- no sfx -- for
anything not in that table. There is no fuzzy/embedding fallback: a wrong
guessed sound (a "whoosh" on an unrecognised tag) is worse than silence,
because it would break the diegetic promise the whole feature exists for.

Placement rule (enforced by `plan_sfx`, not by the caller): one sound at the
START of a shot's FIRST appearance in the timeline (+ a small configurable
offset), never repeated for a cut-run continuation or a later reuse of the
same underlying clip. That is "one SFX per shot maximum."
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DEFAULT = os.path.join(os.path.dirname(HERE), "config", "sfx_map.json")

SFX_DEFAULT_OFFSET = 0.08   # seconds after the shot's own slot t0
SFX_DUCK = dict(threshold=0.06, ratio=6, attack=15, release=300, makeup=1)
# lighter than the music duck (threshold=0.05,ratio=8): a one-shot SFX only
# needs to duck for its own brief duration, not hold a bed down for a scene.


def load_sfx_map(path: str = CFG_DEFAULT) -> list:
    with open(path) as f:
        return json.load(f)["map"]


def resolve_query(tags, sfx_map) -> str | None:
    """First sfx_map row (in file order) whose word set intersects `tags`.

    Word-boundary equality against the tag list, not substring: tags are a
    closed, literal vocabulary (see catalog_footage / the skill's tagging
    step), so this stays simple and auditable rather than clever."""
    tagset = {str(t).strip().lower() for t in (tags or []) if str(t).strip()}
    if not tagset:
        return None
    for row in sfx_map:
        if tagset & set(row["words"]):
            return row["query"]
    return None


def select_sfx_asset(manifest: dict, root: str, query: str):
    """Deterministic pick: sorted by manifest key (source:id), first match.

    Same manifest + same query -> same asset, forever, until the library
    changes -- that is the determinism guarantee `plan` depends on."""
    cands = sorted(
        k for k, a in (manifest.get("assets") or {}).items()
        if a.get("asset_kind") == "sfx" and a.get("query") == query
    )
    if not cands:
        return None
    a = manifest["assets"][cands[0]]
    return {"key": cands[0], "path": os.path.join(root, a["path"]), "asset": a}


def plan_sfx(chosen_events, catalog_by_cid, sfx_map, manifest, root,
             offset: float = SFX_DEFAULT_OFFSET):
    """chosen_events: match_edl.json['events'], sorted by slot (as cmd_plan
    already has them). Returns (audio_events, debug_rows).

    One sound per SHOT: a "shot" is identified by clip id (cid). The first
    slot in timeline order whose clip differs from every clip already sounded
    is eligible; a cut-run continuation of the SAME clip, or a later reuse of
    a clip already sounded, is silently skipped -- never a second SFX on one
    shot, never one on every cut."""
    audio, debug, sounded = [], [], set()
    for ev in chosen_events:
        cid = ev["clip"]
        if cid in sounded:
            continue
        c = catalog_by_cid.get(cid, {})
        tags = c.get("tags") or []
        q = resolve_query(tags, sfx_map)
        row = {"slot": ev["slot"], "clip": cid, "tags": tags, "query": q}
        if not q:
            debug.append(row)
            continue
        asset = select_sfx_asset(manifest, root, q)
        row["asset"] = asset["key"] if asset else None
        if not asset or not os.path.exists(asset["path"]):
            row["skipped"] = "no cached sfx asset for query %r" % q
            debug.append(row)
            continue
        sounded.add(cid)
        start = round(float(ev["t0"]) + offset, 3)
        audio.append({"file": asset["path"], "start": start, "gain_db": 0,
                      "duck": dict(SFX_DUCK), "_clip": cid, "_query": q,
                      "_asset": asset["key"]})
        debug.append(row)
    return audio, debug


# ------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("resolve", help="tags -> query (or none)")
    r.add_argument("--tags", nargs="*", default=[])
    r.add_argument("--map", default=CFG_DEFAULT)

    a = ap.parse_args(argv)
    if a.cmd == "resolve":
        sfx_map = load_sfx_map(a.map)
        q = resolve_query(a.tags, sfx_map)
        print(json.dumps({"tags": a.tags, "query": q}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
