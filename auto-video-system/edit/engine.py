#!/usr/bin/env python3
"""engine.py — the edit engine orchestrator.

Reference-driven vertical-reel editing: give it a reference reel once, a shoot's
footage once, a voiceover + script — it plans and renders any number of cuts
that clone the reference's structure with your footage, deterministically.

Stages (each cached, each re-runnable alone):

  analyze   reference.mp4      -> blueprints/<name>/reference_blueprint.json (+sheets)
  catalog   footage dirs       -> job/catalog.json + contact sheet   (proxies in)
  takes     VO files + script  -> job/vo_selects.json
  vo        selects            -> job/vo.wav + vo_layout + vo_words
  plan      blueprint+catalog+vo+profile -> job/edl.json + beatsheet.md + plan.json
  captions  vo words           -> job/captions.ass (+ overflow check) wired into edl
  render    edl                -> job/out/master.mp4   (draft here; final on Mac --calls)
  qc        master + plan      -> job/qc.json + output contact sheet
  encode    master -> cdn (<20MiB) / preview / master profile

Token economy: everything here is plain code. A model only ever needs to read
beatsheet.md (~1 KB), the QC verdict (~1 KB), and glance at two JPEG sheets.
"""
import argparse, json, os, re, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "lib")
CFG = os.path.join(HERE, "config")
sys.path.insert(0, LIB)

PY = sys.executable or "python3"
SCALE_MAP = {"close": "CU", "medium": "MCU", "wide": "WS"}
MIN_SLOT = 0.45
SNAP_WIN = 0.30
HANDLE = 0.4  # extra source seconds a window must hold beyond slot dur
# --- transitions (blueprint.transitions -> EDL). Conservative by construction:
# a spurious dissolve looks broken, a missed one just looks plainer.
TR_CONF_MIN = 0.60      # below this -> hard cut
TR_MAX_XFADE = 0.50     # never ask for a longer dissolve than this
TR_MIN_BODY = 0.30      # each neighbour keeps >= this many seconds outside the mix
TR_MATCH_WIN = 0.25     # scaled reference cut -> produced boundary match window
TR_HANDLE_EPS = 1.0 / 30


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode:
        sys.stderr.write((r.stderr or r.stdout)[-3000:] + "\n")
        raise SystemExit("FAILED: " + " ".join(str(c) for c in cmd[:6]))
    return r


def jload(p):
    with open(p) as f:
        return json.load(f)


def jdump(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


def cfg(name):
    return jload(os.path.join(CFG, name))


# ---------------------------------------------------------------- analyze
def cmd_analyze(a):
    # analyze_reference writes <workdir>/reference_blueprint.json, and cmd_plan
    # reads <blueprint_dir>/out/reference_blueprint.json — so the workdir IS the
    # "out" subdir. Passing the parent silently wrote blueprints nobody could load.
    out = os.path.join(a.blueprints, a.name, "out")
    bp = os.path.join(out, "reference_blueprint.json")
    if os.path.exists(bp) and not a.force:
        print("cached:", bp); return
    os.makedirs(out, exist_ok=True)
    cmd = [PY, os.path.join(LIB, "analyze_reference.py"), a.reference, "--workdir", out]
    if a.clean_footage:
        cmd.append("--no-chrome-crop")
    if getattr(a, "no_transitions", False):
        cmd.append("--no-transitions")
    if getattr(a, "merge_dissolves", False):
        cmd.append("--merge-dissolves")
    sh(cmd)
    print("blueprint:", bp)


# ---------------------------------------------------------------- catalog
def fam_of(dirname, families):
    for pat, fam in families["patterns"]:
        if re.search(pat, dirname, re.I):
            return fam
    return dirname.upper().replace(" ", "_") or "UNKNOWN"


def _catalog_roots(job, footage_roots, families):
    """Catalog every clip-bearing immediate subdir under each of `footage_roots`.

    Shared by cmd_catalog (his own footage) and cmd_stock (fetched Pexels
    stock) so stock flows through the ONE analysis path -- there is no
    parallel stock cataloger. Returns (merged_entries, debug_dict, roots)."""
    merged, debug_all = [], {}
    roots = []
    for root in footage_roots:
        subs = sorted(d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d)) and not d.startswith(("_", ".")))
        roots += [(os.path.join(root, d), d) for d in subs] or [(root, os.path.basename(root))]
    for path, dname in roots:
        vids = [f for f in os.listdir(path) if f.lower().endswith((".mp4", ".mov", ".m4v"))]
        if not vids:
            continue
        co, dbg = os.path.join(job, "_cat_%s.json" % re.sub(r"\W", "", dname)), \
                  os.path.join(job, "_dbg_%s.json" % re.sub(r"\W", "", dname))
        sh([PY, os.path.join(LIB, "catalog_footage.py"), path, "--out", co, "--debug", dbg])
        fam = fam_of(dname, families)
        dbg_d = jload(dbg)
        for c in jload(co):
            cid = "%s/%s" % (dname, os.path.splitext(c["f"])[0])
            c.update(cid=cid, dir=dname, family=fam, path=os.path.join(path, c["f"]))
            merged.append(c)
            debug_all[cid] = dbg_d.get(c["f"], {})
        os.remove(co); os.remove(dbg)
    return merged, debug_all, roots


def cmd_catalog(a):
    os.makedirs(a.job, exist_ok=True)
    families = cfg("families.json")
    merged, debug_all, roots = _catalog_roots(a.job, a.footage, families)
    jdump(merged, os.path.join(a.job, "catalog.json"))
    jdump(debug_all, os.path.join(a.job, "catalog_debug.json"))
    sh([PY, os.path.join(LIB, "contact_sheet.py"), os.path.join(a.job, "catalog.json"),
        ".", os.path.join(a.job, "footage_sheet")])
    print("catalog: %d clips in %d dirs -> %s" %
          (len(merged), len(roots), os.path.join(a.job, "catalog.json")))


# ---------------------------------------------------------------- stock (fetch + catalog, one step)
def cmd_stock(a):
    """Fetch Pexels stock (texture/atmosphere only, see lib/fetch_assets.py)
    and MERGE it into the job's existing catalog.json via the same
    catalog_footage.py path his own footage goes through -- no parallel
    analysis path. Stock never overwrites or removes his own catalog entries;
    it only appends new STOCK-family ones, tagged with their fetch category so
    the diegetic SFX matcher (lib/sfx_match.py) has content words to key on."""
    os.makedirs(a.job, exist_ok=True)
    root = os.path.expanduser(a.root)
    families = cfg("families.json")
    fetch_summary = []
    for cat in a.category:
        cmd = [PY, os.path.join(LIB, "fetch_assets.py"), "video",
               "--category", cat, "--n", str(a.n), "--root", root]
        if a.api_base:
            cmd += ["--api-base", a.api_base]
        if a.per_page:
            cmd += ["--per-page", str(a.per_page)]
        if a.dry_run:
            cmd += ["--dry-run"]
        r = sh(cmd)
        fetch_summary.append(json.loads(r.stdout))

    video_root = os.path.join(root, "video")
    if not os.path.isdir(video_root) or a.dry_run:
        print(json.dumps({"fetched": fetch_summary, "catalogued": 0}, indent=1))
        return

    new_entries, new_debug, _ = _catalog_roots(a.job, [video_root], families)
    # tag every newly-catalogued stock clip with its fetch category (dirname
    # "stock_<category>" -> "<category>") so the diegetic matcher has a
    # content word without needing a human to tag stock footage by hand.
    for c in new_entries:
        cat = c["dir"][len("stock_"):] if c["dir"].startswith("stock_") else c["dir"]
        tags = list(c.get("tags") or [])
        if cat not in tags:
            tags.append(cat)
        c["tags"] = tags

    cat_path = os.path.join(a.job, "catalog.json")
    dbg_path = os.path.join(a.job, "catalog_debug.json")
    existing = jload(cat_path) if os.path.exists(cat_path) else []
    existing_dbg = jload(dbg_path) if os.path.exists(dbg_path) else {}
    have = {c["cid"] for c in existing}
    added = [c for c in new_entries if c["cid"] not in have]
    existing.extend(added)
    existing_dbg.update(new_debug)
    jdump(existing, cat_path)
    jdump(existing_dbg, dbg_path)
    if os.path.isdir(a.job):
        sh([PY, os.path.join(LIB, "contact_sheet.py"), cat_path, ".",
            os.path.join(a.job, "footage_sheet")])
    print(json.dumps({"fetched": fetch_summary, "catalogued": len(added),
                      "skipped_already_in_catalog": len(new_entries) - len(added),
                      "catalog": cat_path}, indent=1))


# ---------------------------------------------------------------- takes / vo
def cmd_takes(a):
    os.makedirs(a.job, exist_ok=True)
    sh([PY, os.path.join(LIB, "select_takes.py"), "--vo", *a.vo, "--script", a.script,
        "--out", os.path.join(a.job, "vo_selects.json"),
        "--debug", os.path.join(a.job, "vo_debug.json")])
    jdump({os.path.basename(v): os.path.abspath(v) for v in a.vo},
          os.path.join(a.job, "vo_sources.json"))
    print("takes ->", os.path.join(a.job, "vo_selects.json"))


def cmd_vo(a):
    cmd = [PY, os.path.join(LIB, "build_vo.py"),
           "--selects", os.path.join(a.job, "vo_selects.json"),
           "--workdir", a.job, "--gap", str(a.gap), "--tail", str(a.tail)]
    for g in a.gap_after or []:
        cmd += ["--gap-after", g]
    sh(cmd)
    print("vo ->", os.path.join(a.job, "vo.wav"))


# ---------------------------------------------------------------- plan
def _layout_card(text, W=1080, H=1920, platform="engine_default"):
    """Wrap a text card to real measured glyph widths, inside the safe area.

    drawtext clips silently: a card wider than the frame just loses its first
    and last letters, and every programmatic check still passes. The planner has
    PIL and the actual font file, so it measures here and hands the renderer
    explicit lines + fontsize — keeping the Mac-side renderer stdlib-pure."""
    from PIL import ImageFont
    safe = jload(os.path.join(CFG, "safe_areas.json"))["platforms"].get(platform, {})
    left = int(safe.get("left", 110)); right = int(safe.get("right", 200))
    avail = W - left - right
    cands = [os.path.join(HERE, "assets", "fonts", "Montserrat-ExtraBold.ttf"),
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    path = next((f for f in cands if os.path.exists(f)), None)
    words = (text or "").split()
    if not words:
        return [" "], int(W * 0.07)
    for size in range(int(H * 0.048), 20, -2):
        try:
            fnt = ImageFont.truetype(path, size) if path else ImageFont.load_default()
        except Exception:
            fnt = ImageFont.load_default()
        wide = lambda t: fnt.getbbox(t)[2] - fnt.getbbox(t)[0]
        if any(wide(w) > avail for w in words):
            continue                      # a single word still overflows: shrink
        lines, cur = [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if wide(t) <= avail:
                cur = t
            else:
                lines.append(cur); cur = w
        if cur:
            lines.append(cur)
        if len(lines) <= 4:
            return lines, size
    return words[:1], 40


def slots_from_blueprint(bp, total, words):
    g = bp.get("global", {})
    ref_dur = g.get("loop_s") or g.get("duration_s") or bp["shots"][-1]["t1"]
    s = total / ref_dur
    tr_by_t = {round(b["t"], 3): b for b in ((bp.get("transitions") or {}).get("boundaries") or [])}
    scaled = [(c * s, tr_by_t.get(round(c, 3))) for c in bp["cuts"]["times"]]
    cuts = [c * s for c in bp["cuts"]["times"] if c * s < total - MIN_SLOT]
    # snap scaled cuts to this VO's word starts when the reference was VO-locked
    wa = (bp.get("audio", {}).get("cut_word_align") or {})
    if wa.get("frac_within_80ms", 0) >= 0.5 and words:
        starts = sorted(w["start"] for w in words)
        snapped = []
        for c in cuts:
            near = min(starts, key=lambda t: abs(t - c))
            snapped.append(near if abs(near - c) <= SNAP_WIN else c)
        cuts = snapped
    bounds = [0.0] + sorted(set(round(c, 3) for c in cuts)) + [round(total, 3)]
    # merge slots that fell under MIN_SLOT
    clean = [bounds[0]]
    for b in bounds[1:]:
        if b - clean[-1] >= MIN_SLOT or b == bounds[-1]:
            if b == bounds[-1] and b - clean[-1] < MIN_SLOT and len(clean) > 1:
                clean.pop()
            clean.append(b)
    # quantize boundaries to the 30fps frame grid so per-slot frame counts sum
    # exactly to the plan total (per-slot rounding drift is the -t trap reborn)
    fq = sorted(set(round(b * 30) for b in clean))
    clean = [f / 30.0 for f in fq]
    shots = bp["shots"]
    p95 = sorted(sh_["motion"] for sh_ in shots)[max(0, int(0.95 * len(shots)) - 1)] or 1.0
    slots = []
    for i in range(len(clean) - 1):
        t0, t1 = clean[i], clean[i + 1]
        ref_t = (t0 + (t1 - t0) / 2) / s          # midpoint back in reference time
        ref_shot = next((sh_ for sh_ in shots if sh_["t0"] <= ref_t < sh_["t1"]), shots[-1])
        role = "hook" if t0 < 1.5 else ("conclusion" if i == len(clean) - 2 else "body")
        # transition INTO this slot: nearest scaled reference cut
        tr = None
        if i > 0:
            cand = [(abs(st_ - t0), b) for st_, b in scaled if b and abs(st_ - t0) <= TR_MATCH_WIN]
            if cand:
                tr = min(cand, key=lambda x: x[0])[1]
        slots.append({"id": i, "t0": round(t0, 3), "dur": round(t1 - t0, 3), "role": role,
                      "motion": round(min(1.0, ref_shot["motion"] / p95), 3),
                      "scale": SCALE_MAP.get(ref_shot.get("scale_guess"), "MS"),
                      "ref_shot": ref_shot["i"], "static": ref_shot.get("static", False),
                      "punch": float(ref_shot.get("punch", 1.0) or 1.0),
                      "speed": float(ref_shot.get("speed", 1.0) or 1.0),
                      "transition": ({"type": tr["type"], "dur": tr["dur"],
                                      "confidence": tr["confidence"]} if tr else None)})
    return slots


def _best_spans_neutral(cf, ws, dur, cuts, hold, dead, d_target, topk=3):
    """best_spans with exposure/luma terms neutralized — white-sweep product
    footage clips highlights BY DESIGN and must not be scored as overexposed."""
    q = []
    for w in ws:
        sharp_n, _expo, _luma, steady = cf.qparts(w)
        q.append(sharp_n * steady)
    lo = hold[1] if hold else (cf.CONV_HEAD if dur > 6 else 0.0)
    hi = dead[0] if dead else dur
    if hi - lo < d_target:
        lo, hi = 0.0, dur
    n = max(1, int(round(d_target / cf.WIN)))
    cands = []
    import numpy as np
    for s in range(0, len(ws) - n + 1):
        t0, t1 = ws[s]["t0"], round(ws[s]["t0"] + d_target, 2)
        if t0 < lo - 1e-6 or t1 > hi + 1e-6:
            continue
        if any(t0 < c < t1 for c in cuts):
            continue
        seg = q[s:s + n]
        cands.append((round(float(np.mean(seg) - 0.25 * np.std(seg)), 3), t0, t1))
    cands.sort(key=lambda x: (-x[0], x[1]))
    out = []
    for sc, t0, t1 in cands:
        if all(min(t1, o[2]) - max(t0, o[1]) < 0.5 * d_target for o in out):
            out.append((sc, t0, t1))
        if len(out) >= topk:
            break
    return out


def windows_from_catalog(catalog, debug, slot_durs, families):
    import catalog_footage as cf
    fam_cfg = cfg("families.json")
    fam_scale = fam_cfg["scale"]
    white_sweep = set(fam_cfg.get("white_sweep", []))
    all_motion = sorted(w["motion"] for d in debug.values() for w in d.get("windows", []))
    mp95 = all_motion[max(0, int(0.95 * len(all_motion)) - 1)] if all_motion else 1.0
    wins = []
    for c in catalog:
        if c.get("junk"):
            continue
        dbg = debug.get(c["cid"]) or {}
        ws = dbg.get("windows") or []
        if not ws:
            continue
        hold = c.get("hold"); dead = c.get("dead")
        hero = c.get("hero") or [0, 0, 0.5]
        neutral = c["family"] in white_sweep
        for dur in slot_durs:
            fn = _best_spans_neutral if neutral else \
                (lambda *args, **kw: cf.best_spans(*args[1:], **kw))
            spans = fn(cf, ws, c["d"], c.get("cuts", []), hold, dead,
                       min(dur + HANDLE, c["d"]), topk=3)
            # clip_seq: +1 chains ONLY for truly time-adjacent spans within one
            # duration bucket — the matcher's continuity bonus keys on seq+1,
            # and rank-based numbering here once faked "consecutive footage"
            # across buckets and stole a -1.2 bonus it hadn't earned.
            bucket_base = int(dur * 100) * 1000
            prev_t1, seq = None, bucket_base
            for k, (sc, t0, t1) in enumerate(sorted(spans, key=lambda x: x[1])):
                seq = seq + 1 if (prev_t1 is not None and abs(t0 - prev_t1) <= 0.3) \
                    else bucket_base + (k + 1) * 50
                prev_t1 = t1
                seg = [w for w in ws if t0 <= w["t0"] < t1]
                mot = (sum(w["motion"] for w in seg) / len(seg) / mp95) if seg else 0.3
                overlaps_hero = not (t1 <= hero[0] or t0 >= hero[1])
                wins.append({"id": "%s@%.2g#%d" % (c["cid"], dur, k),
                             "clip": c["cid"], "clip_seq": seq, "family": c["family"],
                             "t_in": round(t0, 2), "t_out": round(t1, 2),
                             "motion": round(min(1.0, mot), 3),
                             "scale": fam_scale.get(c["family"], "MS"),
                             "quality": round(min(1.0, sc), 3),
                             "hero": round(hero[2] if overlaps_hero else 0.6 * hero[2], 3),
                             "tags": c.get("tags", []), "_dur": dur})
    return wins



def _apply_transitions(edl_slots, slots, fps, tail_avail, speed_hints=False):
    """Blueprint transition hints -> the renderer's existing vocabulary.

    Safety (all measured against render_edit.plan()'s own asserts):
      * xfade INTO slot k extends slot k-1 by xf frames of REAL source material,
        so slot k-1 must actually hold that much footage after its out point;
      * both neighbours keep >= TR_MIN_BODY seconds outside the mix;
      * confidence < TR_CONF_MIN -> hard cut;
      * whip has no renderer primitive -> hard cut (recorded in the report).
    """
    rep = []
    minbody = int(round(TR_MIN_BODY * fps))
    for k in range(1, min(len(edl_slots), len(slots))):
        hint = slots[k].get("transition")
        n_k = int(round(float(edl_slots[k]["dur"]) * fps))
        n_p = int(round(float(edl_slots[k - 1]["dur"]) * fps))
        if not hint:
            continue
        r = dict(slot=edl_slots[k]["id"], ref_type=hint["type"],
                 ref_dur=hint["dur"], conf=hint["confidence"], applied="cut")
        if hint["confidence"] < TR_CONF_MIN:
            r["reason"] = "confidence %.2f < %.2f" % (hint["confidence"], TR_CONF_MIN)
        elif hint["type"] == "whip":
            r["reason"] = "no whip primitive in renderer"
        elif hint["type"] == "dissolve":
            xf = int(round(min(float(hint["dur"]), TR_MAX_XFADE) * fps))
            avail = tail_avail[k - 1] if tail_avail else None
            if xf < 1:
                r["reason"] = "dissolve shorter than one frame"
            elif min(n_k, n_p) - xf < minbody:
                r["reason"] = "slot too short (%d,%d frames, need %d body)" % (n_k, n_p, minbody)
            elif avail is not None and avail + TR_HANDLE_EPS < xf / float(fps):
                r["reason"] = "only %.2fs of handle after slot %s, need %.2fs" % (
                    avail, edl_slots[k - 1]["id"], xf / float(fps))
            else:
                edl_slots[k]["transition"] = {"type": "xfade", "dur": round(xf / float(fps), 4)}
                r.update(applied="xfade", dur=round(xf / float(fps), 4),
                         handle_s=(None if avail is None else round(avail, 3)))
        elif hint["type"] == "flash":
            fl = max(1, int(round(float(hint["dur"]) / 2.0 * fps)))   # blueprint dur = total span
            if min(n_k, n_p) - fl < minbody:
                r["reason"] = "slot too short for flash (%d,%d frames)" % (n_k, n_p)
            else:
                edl_slots[k]["transition"] = {"type": "flash", "dur": round(fl / float(fps), 4)}
                r.update(applied="flash", dur=round(fl / float(fps), 4))
        rep.append(r)
    # within-shot moves
    for k in range(min(len(edl_slots), len(slots))):
        if edl_slots[k].get("card"):
            continue                      # punch/speed are meaningless on a text card
        pz = float(slots[k].get("punch", 1.0) or 1.0)
        if pz > 1.001:
            edl_slots[k]["punch"] = round(min(1.20, pz), 3)
        sp = float(slots[k].get("speed", 1.0) or 1.0)
        if speed_hints and abs(sp - 1.0) > 1e-6:
            edl_slots[k]["speed"] = round(sp, 3)
    return rep


def cmd_plan(a):
    job = a.job
    profiles = cfg("profiles.json")
    prof = profiles["profiles"][a.profile]
    bp = jload(os.path.join(a.blueprint) if a.blueprint.endswith(".json")
               else os.path.join(a.blueprint, "out", "reference_blueprint.json"))
    layout = jload(os.path.join(job, "vo_layout.json"))
    words = jload(os.path.join(job, "vo_words.json"))["words"]
    total = round(layout[-1]["t1"] + a.tail, 3) if layout else 20.0

    slots = slots_from_blueprint(bp, total, words)
    cwords = [{"t0": w["start"], "t1": w["end"], "w": w["word"].strip()} for w in words]
    vo_c = {"lines": [{"text": l["text"],
                       "words": [w for w in cwords if l["t0"] - 0.05 <= w["t0"] < l["t1"] + 0.05]}
                      for l in layout]}

    if prof.get("text_on_black"):
        edl_slots, planned_black, tail_avail = [], [], None
        for s in slots:
            in_slot = [w["w"] for w in cwords if s["t0"] <= w["t0"] < s["t0"] + s["dur"]]
            ctext = " ".join(in_slot).upper() or " "
            clines, csize = _layout_card(ctext, platform=prof.get("platform", "engine_default"))
            edl_slots.append({"id": "s%02d" % s["id"], "dur": s["dur"],
                              "card": {"text": ctext, "lines": clines, "size": csize,
                                       "style": prof.get("card_style", "default")}})
            planned_black.append([round(s["t0"], 2), round(s["t0"] + s["dur"], 2)])
        chosen = None
    else:
        catalog = jload(os.path.join(job, "catalog.json"))
        debug = jload(os.path.join(job, "catalog_debug.json"))
        # STOCK is DEFAULT DISABLED: fetched b-roll only enters the candidate
        # pool at all with --stock. Filtering it out here (before the matcher
        # ever sees it) is a stronger guarantee than any cost-based forbid --
        # an opted-out reel cannot contain a stock frame, full stop.
        if not getattr(a, "stock", False):
            catalog = [c for c in catalog if c.get("family") != "STOCK"]
        import math
        slot_durs = sorted(set(max(0.25, math.ceil(s["dur"] * 4) / 4) for s in slots))
        wins = windows_from_catalog(catalog, debug, slot_durs, None)
        # matcher inputs
        tmp = os.path.join(job, "_match"); os.makedirs(tmp, exist_ok=True)
        jdump({"meta": {"ref_id": os.path.basename(a.blueprint)},
               "slots": [{k: s[k] for k in ("id", "t0", "dur", "role", "motion", "scale")}
                         for s in slots]}, os.path.join(tmp, "blueprint.json"))
        jdump({"meta": {"shoot_id": "job"}, "windows":
               [{k: w[k] for k in ("id", "clip", "clip_seq", "family", "t_in", "t_out",
                                   "motion", "scale", "quality", "hero", "tags")}
                for w in wins]}, os.path.join(tmp, "catalog.json"))
        jdump(vo_c, os.path.join(tmp, "vo.json"))
        jdump({"name": a.profile, **{k: prof[k] for k in
               ("family_weights", "forbid_hook_families", "conclusion_family",
                "forbid_conclusion_families") if k in prof}},
              os.path.join(tmp, "profile.json"))
        sh([PY, os.path.join(LIB, "match_plan.py"),
            "--blueprint", os.path.join(tmp, "blueprint.json"),
            "--catalog", os.path.join(tmp, "catalog.json"),
            "--vo", os.path.join(tmp, "vo.json"),
            "--profile", os.path.join(tmp, "profile.json"),
            "--out", os.path.join(job, "match_edl.json")])
        match = jload(os.path.join(job, "match_edl.json"))
        bycid = {c["cid"]: c for c in catalog}
        edl_slots, planned_black, tail_avail = [], [], []
        chosen = sorted(match["events"], key=lambda e: e["slot"])
        for ev in chosen:
            src = bycid[ev["clip"]]["path"] if not a.src_map else None
            if a.src_map:
                smap = jload(a.src_map)
                src = smap.get(os.path.basename(bycid[ev["clip"]]["path"]),
                               bycid[ev["clip"]]["path"])
            edl_slots.append({"id": "s%02d" % ev["slot"], "src": src,
                              "src_in": ev["src_in"], "dur": ev["dur"]})
            c_ = bycid[ev["clip"]]
            out_t = float(ev["src_in"]) + float(ev["dur"])
            lim = float(c_.get("d", out_t))
            for cx in sorted(c_.get("cuts") or []):
                if cx > out_t + 1e-6:
                    lim = min(lim, float(cx)); break
            tail_avail.append(max(0.0, lim - out_t))

    tr_report = []
    if not getattr(a, "no_transitions", False):
        tr_report = _apply_transitions(edl_slots, slots, 30, tail_avail,
                                       speed_hints=getattr(a, "speed_hints", False))
    edl = {"version": 1, "fps": 30, "width": 1080, "height": 1920,
           "profile": "final" if a.final else "draft",
           "slots": edl_slots,
           "audio": {"vo": os.path.join(job, "vo.wav"), "target_lufs": -14}}
    if os.path.exists(os.path.join(job, "captions.ass")):
        edl["subtitles"] = os.path.join(job, "captions.ass")  # keep captions wired across re-plans
    if a.music:
        edl["audio"]["music"] = {"file": a.music, "target_lufs": prof.get("music_lufs", -22),
                                 "duck": {"threshold": 0.05, "ratio": 8,
                                          "attack": 20, "release": 400}}
    if getattr(a, "sfx", False) and chosen is not None:
        import sfx_match
        sfx_map = sfx_match.load_sfx_map(os.path.join(CFG, "sfx_map.json"))
        stock_root = os.path.expanduser(getattr(a, "stock_root", None) or "~/engine/stock")
        man_path = os.path.join(stock_root, "manifest.json")
        manifest = jload(man_path) if os.path.exists(man_path) else {"assets": {}}
        sfx_offset = a.sfx_offset if getattr(a, "sfx_offset", None) is not None \
            else sfx_match.SFX_DEFAULT_OFFSET
        sfx_events, sfx_debug = sfx_match.plan_sfx(
            chosen, bycid, sfx_map, manifest, stock_root, offset=sfx_offset)
        if sfx_events:
            edl["audio"]["sfx"] = sfx_events
        jdump({"offset": sfx_offset, "placed": len(sfx_events), "rows": sfx_debug},
              os.path.join(job, "sfx_plan.json"))
    jdump(edl, os.path.join(job, "edl.json"))
    jdump({"width": 1080, "height": 1920, "fps": 30, "duration": total,
           "frames": round(total * 30), "profile": "master", "lufs_target": -14.0,
           "planned_black": planned_black, "planned_freeze": planned_black,
           # only claim an ASS file when this cut actually burns one, or QC's
           # caption check dies on a path that was never written
           "ass": (None if prof.get("text_on_black") else os.path.join(job, "captions.ass")),
           "platform": prof.get("platform", "engine_default"),
           "transitions": tr_report,
           "size_cap_mib": None}, os.path.join(job, "plan.json"))

    # beat sheet — the approval artifact
    lines = ["# Beat sheet — %s  (profile: %s)" % (os.path.basename(job), a.profile),
             "", "Total %.2fs · %d slots · reference `%s` (%.2fs, x%.2f)" %
             (total, len(slots), os.path.basename(a.blueprint),
              bp.get("global", {}).get("duration_s", 0),
              total / max(0.01, bp.get("global", {}).get("loop_s")
                          or bp.get("global", {}).get("duration_s", total))),
             "", "| slot | time | dur | VO | source | why |", "|---|---|---|---|---|---|"]
    for i, s in enumerate(slots):
        in_slot = " ".join(w["w"] for w in cwords if s["t0"] <= w["t0"] < s["t0"] + s["dur"])
        if chosen is None:
            srcd = "CARD: " + (in_slot.upper()[:40] or "—")
            why = "text-on-black"
        else:
            ev = chosen[i]
            srcd = "%s [%.1f–%.1f]" % (ev["clip"], ev["src_in"], ev["src_in"] + ev["dur"])
            why = ev.get("rationale", "")[:60]
        lines.append("| S%02d %s | %.2f | %.2f | %s | %s | %s |" %
                     (s["id"], s["role"][:4], s["t0"], s["dur"],
                      (in_slot[:44] or "—"), srcd, why))
    applied = [r for r in tr_report if r["applied"] != "cut"]
    if tr_report:
        lines += ["", "Transitions: %d applied (%s), %d fell back to hard cut" % (
            len(applied), ", ".join("%s %s %.2fs" % (r["slot"], r["applied"], r.get("dur", 0))
                                    for r in applied) or "-",
            len(tr_report) - len(applied))]
    open(os.path.join(job, "beatsheet.md"), "w").write("\n".join(lines) + "\n")
    print("plan -> edl.json, plan.json, beatsheet.md (%d slots, %.1fs)" % (len(slots), total))


# ---------------------------------------------------------------- captions
def cmd_captions(a):
    job = a.job
    out = os.path.join(job, "captions.ass")
    sh([PY, os.path.join(LIB, "captions.py"), "--words", os.path.join(job, "vo_words.json"),
        "--preset", a.preset, "--platform", a.platform, "--out", out])
    v = os.path.join(job, "caption_check.json")
    sh([PY, os.path.join(LIB, "caption_check.py"), "--ass", out, "--platform", a.platform,
        "--fontsdir", os.path.join(HERE, "assets", "fonts"), "--json", v])
    verdict = jload(v)
    if not verdict.get("pass", verdict.get("passed", False)):
        raise SystemExit("caption check FAILED: %s" % json.dumps(verdict)[:800])
    edl_p = os.path.join(job, "edl.json"); edl = jload(edl_p)
    edl["subtitles"] = out
    jdump(edl, edl_p)
    print("captions ok ->", out)


# ---------------------------------------------------------------- render / qc
def cmd_render(a):
    job = a.job
    cmd = [PY, os.path.join(LIB, "render_edit.py"), os.path.join(job, "edl.json"),
           "--build", os.path.join(job, "build"),
           "--out", os.path.join(job, "out", "master.mp4")]
    if a.calls:
        cmd.append("--calls")
    if a.seek_exact:
        cmd += ["--seek-mode", "exact"]
    os.makedirs(os.path.join(job, "out"), exist_ok=True)
    os.environ.setdefault("ENGINE_FONTSDIR", os.path.join(HERE, "assets", "fonts"))
    r = sh(cmd)
    print(r.stdout[-1200:] if not a.calls else r.stdout)
    if not a.calls:
        sh(cmd + ["--verify"])
        print("verify: ok")


def cmd_qc(a):
    job = a.job
    master = os.path.join(job, "out", "master.mp4")
    r = subprocess.run([PY, os.path.join(LIB, "qc.py"), "--file", master,
                        "--plan", os.path.join(job, "plan.json"),
                        "--json", os.path.join(job, "qc.json")],
                       capture_output=True, text=True)
    sh(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", master,
        "-vf", "fps=1.6,scale=200:-2,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:"
        "text='%{pts\\:hms}':x=3:y=3:fontsize=15:fontcolor=yellow:box=1:boxcolor=black@0.7,"
        "tile=6x5:padding=2:color=black", "-frames:v", "1", "-q:v", "3",
        os.path.join(job, "out", "master_sheet.jpg")])
    print(r.stdout[-1500:] or r.stderr[-600:])
    print("sheet ->", os.path.join(job, "out", "master_sheet.jpg"))
    if r.returncode:
        raise SystemExit("QC FAILED")


# ---------------------------------------------------------------- encode
def cmd_encode(a):
    job = a.job
    sh([PY, os.path.join(LIB, "encode.py"), "--profile", a.profile,
        "--vin", os.path.join(job, "out", "master.mp4"),
        "--out", os.path.join(job, "out", "final_%s.mp4" % a.profile), "--no-norm"])
    print("encode ->", os.path.join(job, "out", "final_%s.mp4" % a.profile))


# ---------------------------------------------------------------- batch
SHARED = ["catalog.json", "catalog_debug.json", "vo.wav", "vo_layout.json",
          "vo_words.json", "vo_selects.json", "vo_sources.json"]


def cmd_batch(a):
    """Render every cut of one voiceover in a single pass, then emit ONE digest.

    The point is token economy: a model reads one combined beat sheet, one QC
    table and one strip image for the whole batch instead of paying full
    conversation overhead per video. Per-cut artifacts stay on disk for
    drill-down when the digest flags something.
    """
    profiles = [p.strip() for p in a.profiles.split(",") if p.strip()]
    cuts_root = os.path.join(a.job, "cuts")
    os.makedirs(cuts_root, exist_ok=True)
    results = []

    for prof in profiles:
        cj = os.path.join(cuts_root, prof)
        os.makedirs(cj, exist_ok=True)
        for f in SHARED:
            src = os.path.join(a.job, f)
            if os.path.exists(src) and not os.path.exists(os.path.join(cj, f)):
                shutil.copy2(src, os.path.join(cj, f))
        cutcfg = cfg("profiles.json")["profiles"].get(prof, {})
        rec = {"profile": prof, "job": cj, "stage": None, "error": None}
        try:
            pa = argparse.Namespace(job=cj, blueprint=a.blueprint, profile=prof,
                                    tail=0.7, music=a.music, src_map=a.src_map,
                                    final=a.final)
            rec["stage"] = "plan"; cmd_plan(pa)
            # text-on-black cuts ARE the words on screen — burning captions too
            # renders every line twice. Caught in a batch strip; skip them.
            if not cutcfg.get("text_on_black"):
                rec["stage"] = "captions"
                cmd_captions(argparse.Namespace(
                    job=cj, preset=cutcfg.get("caption_preset", "clean"),
                    platform=cutcfg.get("platform", "engine_default")))
            rec["stage"] = "render"
            cmd_render(argparse.Namespace(job=cj, calls=False, seek_exact=False))
            rec["stage"] = "qc"
            try:
                cmd_qc(argparse.Namespace(job=cj))
            except SystemExit:
                pass                      # verdict is captured from qc.json below
            q = jload(os.path.join(cj, "qc.json"))
            rec.update(qc_pass=q.get("pass"), qc_failed=q.get("failed", []))
            if q.get("pass") and a.encode != "none":
                rec["stage"] = "encode"
                cmd_encode(argparse.Namespace(job=cj, profile=a.encode))
                out = os.path.join(cj, "out", "final_%s.mp4" % a.encode)
                rec["deliverable"] = out
                rec["mib"] = round(os.path.getsize(out) / 1048576.0, 2)
            rec["stage"] = "done"
        except SystemExit as e:
            rec["error"] = str(e)[:200]
            rec["qc_pass"] = False
        results.append(rec)

    # ---- one combined beat sheet
    lines = ["# Batch digest — %s" % os.path.basename(a.job),
             "", "%d cuts · reference `%s`" % (len(results), os.path.basename(a.blueprint)), ""]
    for r in results:
        bs = os.path.join(r["job"], "beatsheet.md")
        status = "OK" if r.get("qc_pass") else ("FAILED@%s %s" % (r["stage"], r.get("error") or r.get("qc_failed")))
        lines.append("## %s — %s%s" % (r["profile"], status,
                                       (" · %.2f MiB" % r["mib"]) if r.get("mib") else ""))
        if os.path.exists(bs):
            body = open(bs).read().splitlines()
            tbl = [l for l in body if l.startswith("| S")]
            head = next((l for l in body if l.startswith("Total")), "")
            lines.append(head)
            # compress: slot | dur | source clip only — the VO and rationale are
            # identical across cuts of one voiceover, so repeating them is waste
            for l in tbl:
                c = [x.strip() for x in l.strip("|").split("|")]
                if len(c) >= 5:
                    lines.append("- %s %ss  %s" % (c[0], c[2], c[4].split(" [")[0]))
        lines.append("")
    digest = os.path.join(a.job, "batch_digest.md")
    open(digest, "w").write("\n".join(lines) + "\n")

    # ---- one strip image: 6 frames per cut, one row per cut
    rows = []
    strip_dir = os.path.join(a.job, "_strip"); os.makedirs(strip_dir, exist_ok=True)
    for r in results:
        m = os.path.join(r["job"], "out", "master.mp4")
        if not os.path.exists(m):
            continue
        row = os.path.join(strip_dir, "%s.jpg" % r["profile"])
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                    "format=duration", "-of", "csv=p=0", m],
                                   capture_output=True, text=True).stdout.strip() or 1)
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", m,
                        "-vf", "fps=%.4f,scale=150:-2,drawtext=fontfile=/usr/share/fonts/"
                        "truetype/dejavu/DejaVuSans.ttf:text='%s':x=3:y=3:fontsize=13:"
                        "fontcolor=yellow:box=1:boxcolor=black@0.7,tile=6x1:padding=2:color=black"
                        % (6.0 / dur, r["profile"].replace(":", "")),
                        "-frames:v", "1", "-q:v", "4", row], capture_output=True)
        if os.path.exists(row):
            rows.append(row)
    strip = os.path.join(a.job, "batch_strip.jpg")
    if rows:
        # stack rows with PIL rather than an ffmpeg concat of stills: the concat
        # demuxer needs per-image durations and silently produced nothing here
        from PIL import Image
        ims = [Image.open(r).convert("RGB") for r in rows]
        w = max(i.width for i in ims)
        pad = 3
        canvas = Image.new("RGB", (w, sum(i.height for i in ims) + pad * (len(ims) - 1)), (0, 0, 0))
        y = 0
        for i in ims:
            canvas.paste(i, (0, y)); y += i.height + pad
        canvas.save(strip, quality=82)

    jdump(results, os.path.join(a.job, "batch_results.json"))
    ok = sum(1 for r in results if r.get("qc_pass"))
    print("batch: %d/%d passed QC" % (ok, len(results)))
    for r in results:
        print("  %-16s %s%s" % (r["profile"], "OK" if r.get("qc_pass") else
                                "FAIL@%s" % r["stage"], (" %.1fMiB" % r["mib"]) if r.get("mib") else ""))
    print("digest ->", digest)
    print("strip  ->", strip if rows else "(none)")


# ---------------------------------------------------------------- shoot
AUDIO_ARTIFACTS = ["vo_meas.json", "vo_norm.wav", "mix.wav", "mix_meas.json", "mix_final.wav"]


def _share_audio(src_build, dst_build):
    """Copy the finished audio artifacts (+ their cache sidecars) between cuts.

    All cuts of one voiceover share the same VO and the same total duration, so
    the renderer's audio cache key is identical for each — but each cut has its
    own build dir, so it would re-measure and re-normalise five times. Seeding
    the cache turns those into instant hits. Segments are NOT shared: same slot
    ids, different sources."""
    if not os.path.isdir(src_build):
        return 0
    os.makedirs(dst_build, exist_ok=True)
    n = 0
    for f in AUDIO_ARTIFACTS:
        s, d = os.path.join(src_build, f), os.path.join(dst_build, f)
        sk = os.path.join(src_build, "." + os.path.splitext(f)[0] + ".ok")
        if os.path.exists(s) and not os.path.exists(d):
            shutil.copy2(s, d)
            if os.path.exists(sk):
                shutil.copy2(sk, os.path.join(dst_build, os.path.basename(sk)))
            n += 1
    return n


def _discover_vos(a):
    """Manifest if given, else pair each audio file with a same-named .txt."""
    if a.manifest:
        return jload(a.manifest)
    out = []
    for f in sorted(os.listdir(a.vo_dir)):
        if not f.lower().endswith((".wav", ".m4a", ".mp3", ".mov", ".mp4")):
            continue
        stem = os.path.splitext(f)[0]
        script = os.path.join(a.scripts, stem + ".txt") if a.scripts else None
        if script and os.path.exists(script):
            out.append({"id": stem, "vo": [os.path.join(a.vo_dir, f)], "script": script})
    return out


def cmd_shoot(a):
    """Plan (and optionally render) every cut of every voiceover in one pass.

    Two phases on purpose. `--plan-only` stops at the beat sheets so the
    approval gate still exists at 50 videos — but you review one structure per
    VOICEOVER (the 5 cuts share it) instead of 50 near-identical sheets. Then
    `--render` builds everything and reports one table plus one grid image.
    """
    profiles = [p.strip() for p in a.profiles.split(",") if p.strip()]
    vos = _discover_vos(a)
    if not vos:
        raise SystemExit("no voiceovers found — pass --manifest, or --vo-dir with matching .txt scripts")
    os.makedirs(a.job_root, exist_ok=True)
    shared = os.path.join(a.job_root, "_shared")

    # ---- catalog ONCE for the whole shoot
    if not os.path.exists(os.path.join(shared, "catalog.json")) or a.force:
        cmd_catalog(argparse.Namespace(job=shared, footage=a.footage))
    else:
        print("catalog: cached")

    index, t_start = [], __import__("time").time()
    for vi, v in enumerate(vos):
        vjob = os.path.join(a.job_root, v["id"])
        os.makedirs(vjob, exist_ok=True)
        for f in ["catalog.json", "catalog_debug.json"]:
            if not os.path.exists(os.path.join(vjob, f)):
                shutil.copy2(os.path.join(shared, f), os.path.join(vjob, f))
        try:
            if not os.path.exists(os.path.join(vjob, "vo.wav")) or a.force:
                cmd_takes(argparse.Namespace(job=vjob, vo=v["vo"], script=v["script"]))
                cmd_vo(argparse.Namespace(job=vjob, gap=a.gap, tail=0.7,
                                          gap_after=v.get("gap_after")))
        except SystemExit as e:
            print("  !! %s VO failed: %s" % (v["id"], str(e)[:160]))
            index.append({"vo": v["id"], "error": str(e)[:200], "cuts": []})
            continue

        cuts_root = os.path.join(vjob, "cuts")
        os.makedirs(cuts_root, exist_ok=True)
        rec_cuts, first_build = [], None
        for prof in profiles:
            cj = os.path.join(cuts_root, prof)
            os.makedirs(cj, exist_ok=True)
            for f in SHARED:
                s = os.path.join(vjob, f)
                if os.path.exists(s) and not os.path.exists(os.path.join(cj, f)):
                    shutil.copy2(s, os.path.join(cj, f))
            cutcfg = cfg("profiles.json")["profiles"].get(prof, {})
            rec = {"profile": prof, "job": cj, "stage": None}
            try:
                cmd_plan(argparse.Namespace(job=cj, blueprint=a.blueprint, profile=prof,
                                            tail=0.7, music=a.music, src_map=a.src_map,
                                            final=a.final))
                rec["stage"] = "planned"
                if not cutcfg.get("text_on_black"):
                    cmd_captions(argparse.Namespace(
                        job=cj, preset=cutcfg.get("caption_preset", "clean"),
                        platform=cutcfg.get("platform", "engine_default")))
                if a.render:
                    if first_build:
                        _share_audio(first_build, os.path.join(cj, "build"))
                    cmd_render(argparse.Namespace(job=cj, calls=False, seek_exact=False))
                    first_build = first_build or os.path.join(cj, "build")
                    try:
                        cmd_qc(argparse.Namespace(job=cj))
                    except SystemExit:
                        pass
                    q = jload(os.path.join(cj, "qc.json"))
                    rec.update(qc_pass=q.get("pass"), qc_failed=q.get("failed", []),
                               stage="rendered")
                    if q.get("pass") and a.encode != "none":
                        cmd_encode(argparse.Namespace(job=cj, profile=a.encode))
                        out = os.path.join(cj, "out", "final_%s.mp4" % a.encode)
                        rec["deliverable"] = out
                        rec["mib"] = round(os.path.getsize(out) / 1048576.0, 2)
            except SystemExit as e:
                rec["error"] = str(e)[:200]; rec["qc_pass"] = False
            rec_cuts.append(rec)
        index.append({"vo": v["id"], "job": vjob, "cuts": rec_cuts})
        done = sum(len(x.get("cuts", [])) for x in index)
        print("[%d/%d VOs] %s -> %d cuts (%d total, %.1f min elapsed)" %
              (vi + 1, len(vos), v["id"], len(rec_cuts), done,
               (__import__("time").time() - t_start) / 60))

    _shoot_digest(a, index, profiles)
    jdump(index, os.path.join(a.job_root, "shoot_index.json"))


def _shoot_digest(a, index, profiles):
    total = sum(len(v.get("cuts", [])) for v in index)
    ok = sum(1 for v in index for c in v.get("cuts", []) if c.get("qc_pass"))
    lines = ["# Shoot digest — %s" % os.path.basename(a.job_root), "",
             "%d voiceovers × %d profiles = %d cuts%s" %
             (len(index), len(profiles), total,
              ("  ·  %d/%d passed QC" % (ok, total)) if a.render else "  ·  PLANNED, not rendered"),
             "", "Reference `%s`" % os.path.basename(a.blueprint), ""]

    # One structure per VOICEOVER: the cuts share slot timing and VO, differing
    # only in which clip fills each slot. Repeating it 5x would be pure waste.
    lines.append("## Structure (one per voiceover — cuts share it)")
    for v in index:
        if v.get("error"):
            lines.append("- **%s** — FAILED: %s" % (v["vo"], v["error"])); continue
        rep = next((c for c in v["cuts"] if os.path.exists(os.path.join(c["job"], "beatsheet.md"))), None)
        if not rep:
            lines.append("- **%s** — no plan" % v["vo"]); continue
        body = open(os.path.join(rep["job"], "beatsheet.md")).read().splitlines()
        head = next((l for l in body if l.startswith("Total")), "")
        rows = [l for l in body if l.startswith("| S")]
        lines.append("- **%s** — %s" % (v["vo"], head))
        for l in rows:
            c = [x.strip() for x in l.strip("|").split("|")]
            if len(c) >= 5:
                lines.append("  - %s %ss  “%s”" % (c[0], c[2], c[3][:48]))
    lines.append("")

    if a.render:
        lines += ["## Results", "", "| voiceover | " + " | ".join(profiles) + " |",
                  "|---" * (len(profiles) + 1) + "|"]
        for v in index:
            cells = []
            for p in profiles:
                c = next((x for x in v.get("cuts", []) if x["profile"] == p), None)
                cells.append("—" if not c else
                             ("%.1fMiB" % c["mib"] if c.get("qc_pass") and c.get("mib")
                              else ("OK" if c.get("qc_pass") else
                                    "FAIL %s" % (c.get("qc_failed") or c.get("error", ""))[:24])))
            lines.append("| %s | %s |" % (v["vo"], " | ".join(cells)))
        lines.append("")
        bad = [(v["vo"], c["profile"], c.get("qc_failed") or c.get("error"))
               for v in index for c in v.get("cuts", []) if not c.get("qc_pass")]
        if bad:
            lines.append("## Needs attention")
            for b in bad:
                lines.append("- %s / %s — %s" % b)
            lines.append("")

    digest = os.path.join(a.job_root, "shoot_digest.md")
    open(digest, "w").write("\n".join(lines) + "\n")

    grid = os.path.join(a.job_root, "shoot_grid.jpg")
    if a.render:
        _shoot_grid(a, index, profiles, grid)
    print("\n%d/%d cuts passed QC" % (ok, total) if a.render else "\n%d cuts planned" % total)
    print("digest ->", digest)
    if a.render and os.path.exists(grid):
        print("grid   ->", grid)


def _shoot_grid(a, index, profiles, out):
    """One representative frame per finished cut, labelled VO·profile.

    50 full contact sheets is unreadable; the 14-check QC already proves the
    measurable properties. This grid is the cheap eyeball pass that catches the
    gross failures numbers miss — black frames, wrong footage, dead cards."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        fnt = ImageFont.load_default()
    TW, TH, LH = 132, 234, 17
    tiles = []
    for v in index:
        for p in profiles:
            c = next((x for x in v.get("cuts", []) if x["profile"] == p), None)
            m = os.path.join(c["job"], "out", "master.mp4") if c else None
            tiles.append((v["vo"], p, m if m and os.path.exists(m) else None))
    cols = len(profiles)
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * TW, rows * (TH + LH)), (12, 12, 14))
    dr = ImageDraw.Draw(canvas)
    for i, (vo, prof, m) in enumerate(tiles):
        x, y = (i % cols) * TW, (i // cols) * (TH + LH)
        dr.text((x + 3, y + 2), "%s·%s" % (vo, prof[:9]), font=fnt, fill=(255, 220, 60))
        if not m:
            dr.text((x + 6, y + LH + 40), "—", font=fnt, fill=(160, 60, 60)); continue
        tmp = os.path.join(a.job_root, "_tile.jpg")
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", "1.0",
                        "-i", m, "-vf", "scale=%d:-2" % (TW - 4), "-frames:v", "1", tmp],
                       capture_output=True)
        if os.path.exists(tmp):
            im = Image.open(tmp).convert("RGB")
            canvas.paste(im.crop((0, 0, im.width, min(im.height, TH))), (x + 2, y + LH))
    canvas.save(out, quality=80)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze"); p.add_argument("reference")
    p.add_argument("--blueprints", default=os.path.join(HERE, "blueprints"))
    p.add_argument("--name", required=True); p.add_argument("--force", action="store_true")
    p.add_argument("--clean-footage", action="store_true")
    p.add_argument("--no-transitions", action="store_true")
    p.add_argument("--merge-dissolves", action="store_true",
                   help="promote DISCOVERED dissolves (scdet never reports them) to cuts")

    p = sub.add_parser("catalog"); p.add_argument("footage", nargs="+")
    p.add_argument("--job", required=True)

    p = sub.add_parser("stock", help="fetch Pexels texture/atmosphere stock + catalog it, one step")
    p.add_argument("--job", required=True)
    p.add_argument("--category", nargs="+", default=["sky"],
                   help="one or more of: sky water rain city concrete fabric light nature "
                        "(see lib/fetch_assets.py CATEGORIES)")
    p.add_argument("--n", type=int, default=4, help="max clips per category")
    p.add_argument("--per-page", type=int, default=None)
    p.add_argument("--root", default="~/engine/stock")
    p.add_argument("--api-base", default=None, help="override Pexels API base (fixture/testing)")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("takes"); p.add_argument("--vo", nargs="+", required=True)
    p.add_argument("--script", required=True); p.add_argument("--job", required=True)

    p = sub.add_parser("vo"); p.add_argument("--job", required=True)
    p.add_argument("--gap", type=float, default=0.35); p.add_argument("--tail", type=float, default=0.7)
    p.add_argument("--gap-after", action="append")

    p = sub.add_parser("plan"); p.add_argument("--job", required=True)
    p.add_argument("--blueprint", required=True); p.add_argument("--profile", required=True)
    p.add_argument("--tail", type=float, default=0.7); p.add_argument("--music")
    p.add_argument("--src-map"); p.add_argument("--final", action="store_true")
    p.add_argument("--no-transitions", action="store_true",
                   help="force hard cuts everywhere (also drops punch/speed hints)")
    p.add_argument("--speed-hints", action="store_true",
                   help="apply detected slow-motion to slots (OFF by default: only "
                        "duplicate-frame slow-mo is detectable and it is rare)")
    p.add_argument("--stock", action="store_true",
                   help="allow fetched Pexels STOCK-family clips into the candidate "
                        "pool (DEFAULT DISABLED). Never the hook or conclusion slot "
                        "regardless of this flag -- see config/profiles.json "
                        "forbid_hook_families / forbid_conclusion_families.")
    p.add_argument("--sfx", action="store_true",
                   help="place DIEGETIC sfx keyed to shot content (config/sfx_map.json), "
                        "one per shot at its start (DEFAULT DISABLED).")
    p.add_argument("--sfx-offset", type=float, default=None,
                   help="seconds after a shot's slot-start to place its sfx (default 0.08)")
    p.add_argument("--stock-root", default=None,
                   help="stock library root for sfx lookup (default ~/engine/stock)")

    p = sub.add_parser("captions"); p.add_argument("--job", required=True)
    p.add_argument("--preset", default="clean"); p.add_argument("--platform", default="engine_default")

    p = sub.add_parser("render"); p.add_argument("--job", required=True)
    p.add_argument("--calls", action="store_true"); p.add_argument("--seek-exact", action="store_true")

    p = sub.add_parser("qc"); p.add_argument("--job", required=True)

    p = sub.add_parser("encode"); p.add_argument("--job", required=True)
    p.add_argument("--profile", default="cdn", choices=["master", "cdn", "preview"])

    p = sub.add_parser("batch"); p.add_argument("--job", required=True)
    p.add_argument("--blueprint", required=True)
    p.add_argument("--profiles", required=True, help="comma-separated cut profiles")
    p.add_argument("--music"); p.add_argument("--src-map")
    p.add_argument("--final", action="store_true")
    p.add_argument("--encode", default="cdn", choices=["none", "master", "cdn", "preview"])

    p = sub.add_parser("shoot"); p.add_argument("--job-root", required=True)
    p.add_argument("--footage", nargs="+", required=True)
    p.add_argument("--blueprint", required=True); p.add_argument("--profiles", required=True)
    p.add_argument("--manifest"); p.add_argument("--vo-dir"); p.add_argument("--scripts")
    p.add_argument("--gap", type=float, default=0.35)
    p.add_argument("--plan-only", dest="render", action="store_false", default=False)
    p.add_argument("--render", dest="render", action="store_true")
    p.add_argument("--encode", default="cdn", choices=["none", "master", "cdn", "preview"])
    p.add_argument("--music"); p.add_argument("--src-map")
    p.add_argument("--final", action="store_true"); p.add_argument("--force", action="store_true")

    a = ap.parse_args()
    {"analyze": cmd_analyze, "catalog": cmd_catalog, "stock": cmd_stock,
     "takes": cmd_takes, "vo": cmd_vo,
     "batch": cmd_batch, "shoot": cmd_shoot,
     "plan": cmd_plan, "captions": cmd_captions, "render": cmd_render,
     "qc": cmd_qc, "encode": cmd_encode}[a.cmd](a)


if __name__ == "__main__":
    main()
