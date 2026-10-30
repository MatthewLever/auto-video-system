#!/usr/bin/env python3
"""fetch_assets.py — licensed STOCK ASSET FETCHER for the edit engine.

Two sources, two very different jobs:

  video  Pexels video API  -> portrait TEXTURE/ATMOSPHERE b-roll for filler slots
  sfx    Openverse audio   -> CC0-only one-shots / ambiences (DEFAULT OFF)

The primary output of this module is NOT the media. It is `manifest.json`:
the per-asset licence record that says where every frame and every sample in a
published ad came from, under what licence, on what date, and what attribution
obligation rides with it. Media without a manifest record is not usable output.

------------------------------------------------------------------ WHY NO PEOPLE
Hard rule, enforced in `person_block()`: any candidate whose title / alt text /
tags / URL slug contains a person-word is rejected outright, and the rejection
is recorded.

Rationale (preserve this — it is the whole reason the filter exists):
every free stock platform DISCLAIMS model releases and pushes the warranty down
to the publisher. Pexels' own licence tells you to get permission for
"commercial use of identifiable people"; Openverse/CC0 waives COPYRIGHT only —
it says nothing about personality or publicity rights, which are a separate
right held by the person, not the photographer. So a stranger's face in a paid
ad for a product Matthew sells is HIS legal exposure, not the platform's, and
it is exposure he gets nothing for: the clip is filler b-roll. The cheapest
correct answer is to never let a human being into the stock pool at all.

Allowed subject matter (see CATEGORIES): sky/sunset/clouds, water/ocean/rain,
city-at-night/street lights, concrete/asphalt/wall texture, fabric/material
macro, abstract light/bokeh, nature texture.

--------------------------------------------------------------------- KEYS / ToS
Pexels REQUIRES an API key. Key is read from ~/publishing/media/pexels_key.json
(mode 600) or $PEXELS_KEY. The value is NEVER printed, never logged, never put
in the manifest. If there is no key this module FAILS — it must not fall back to
scraping or to the website's internal JSON, because the Pexels Terms explicitly
prohibit scraping and data-mining. No key, no video. That is the correct answer.

Openverse needs no key. Anonymous quota measured at ~20 requests/min burst and
~200/day sustained, so requests are rate-limited here and the daily count is
persisted next to the library.

CLI
  fetch_assets.py video --category sky --n 6 [--query "..."] [--root DIR]
  fetch_assets.py sfx   --query "rain" --n 3 [--root DIR]
  fetch_assets.py credits [--root DIR]        # caption-ready attribution block
  fetch_assets.py verify  [--root DIR]        # re-hash library vs manifest

Everything network-facing goes through _http_json/_http_bytes so tests can
monkeypatch one seam, and --api-base lets the whole Pexels path be driven
against a local fixture server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.expanduser("~/engine/stock")
PEXELS_KEY_PATH = os.path.expanduser("~/publishing/media/pexels_key.json")
PEXELS_API = "https://api.pexels.com"
OPENVERSE_API = "https://api.openverse.org/v1"
UA = "edit-engine/1.0 (stock fetcher; contact via project owner)"

# Openverse anonymous quota, measured: 20/min burst, 200/day sustained.
OV_MIN_INTERVAL = 3.2      # seconds between requests -> <=18/min, inside burst
OV_DAILY_CAP = 180         # leave headroom under the measured 200/day

TARGET_W, TARGET_H = 1080, 1920

# ------------------------------------------------------------------ categories
# query text -> library subdir. Subdir names are prefixed "stock_" because
# config/families.json maps ^stock[_-] to family STOCK before any other pattern
# (a category like "concrete floor" would otherwise hit the FEET pattern).
CATEGORIES = {
    "sky":      "sunset sky clouds timelapse",
    "water":    "ocean water waves surface",
    "rain":     "rain drops window closeup",
    "city":     "city night street lights bokeh",
    "concrete": "concrete wall texture asphalt",
    "fabric":   "fabric textile macro texture",
    "light":    "abstract light bokeh blur",
    "nature":   "leaves nature texture macro",
}

# --------------------------------------------------------------- person filter
# EXACT lemmas. Plural/irregular forms are generated below, so this list stays
# readable and additions are cheap.
PERSON_LEMMAS = [
    "person", "people", "man", "men", "woman", "women", "girl", "boy",
    "child", "children", "kid", "human", "face", "portrait", "hand", "hands",
    "team", "business", "office", "model", "crowd", "family", "couple",
    "worker", "staff",
    # beyond the required minimum — same risk, same answer
    "guy", "lady", "gentleman", "adult", "teen", "teenager", "baby", "toddler",
    "student", "employee", "colleague", "friend", "friends", "group",
    "selfie", "headshot", "body", "skin", "lips", "eye", "eyes", "hair",
    "finger", "fingers", "arm", "arms", "leg", "legs", "feet", "foot",
    "smile", "smiling", "posing", "pose", "sitting", "standing", "walking",
    "talking", "working", "meeting", "crew", "audience", "customer", "client",
    "doctor", "nurse", "chef", "athlete", "dancer", "musician", "artist",
    "wedding", "birthday", "party",
]

IRREGULAR = {
    "man": ["men"], "woman": ["women"], "child": ["children"],
    "person": ["persons", "people"], "foot": ["feet"],
}

# Compound ROOTS matched as a SUFFIX only (see person_block for the rule and
# the justification for suffix-only).
COMPOUND_SUFFIX_ROOTS = [
    "man", "men", "woman", "women", "girl", "boy", "kid", "child", "children",
    "person", "people", "hand", "hands", "worker", "workers",
    "model", "models", "crowd", "family", "couple", "staff", "team", "human",
    # NOT "face": measured false positive on "surface", which is a core allowed
    # subject ("water surface"). "face" stays an EXACT-match block only; the
    # compounds it would have caught ("babyface") are already blocked by their
    # other half.
]
# The ONE root also matched as a prefix: "hand". "handheld", "handmade",
# "handbag", "handshake" all overwhelmingly show hands. Over-blocking a door
# handle costs nothing; a hand in a paid ad costs a release we do not have.
COMPOUND_PREFIX_ROOTS = ["hand"]


def _plurals(w: str) -> set:
    out = {w}
    out |= set(IRREGULAR.get(w, []))
    if w.endswith("y") and len(w) > 3 and w[-2] not in "aeiou":
        out.add(w[:-1] + "ies")
    elif w.endswith(("s", "x", "z", "ch", "sh")):
        out.add(w + "es")
    else:
        out.add(w + "s")
    return out


PERSON_EXACT = set()
for _w in PERSON_LEMMAS:
    PERSON_EXACT |= _plurals(_w)


def tokenize(text: str) -> list:
    """Lowercase, split on anything that is not a letter or digit.

    URL slugs ("a-woman-walking-on-the-beach-12345"), hyphenated tags
    ("field-recording") and CamelCase-free titles all reduce to the same token
    stream, so one rule covers title, alt text, tags and the source URL."""
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]


def person_block(fields: dict) -> list:
    """Return the list of (field, token, rule) hits. Empty list == allowed.

    MATCHING RULE, and why:

      A. EXACT TOKEN match against PERSON_EXACT (lemmas + generated plurals +
         irregulars). Token-based, never substring — substring matching turns
         "manhattan" into a person and makes the filter useless noise.

      B. SUFFIX compound: token ENDS WITH a root in COMPOUND_SUFFIX_ROOTS.
         Catches the productive English pattern where the person-word is the
         HEAD of the compound and therefore what is depicted: businessman,
         sportswoman, schoolgirl, cowboy, coworker, supermodel, superhuman.
         Suffix-only is the point: "humanity", "management", "manual",
         "manufacture", "mandala", "manhattan" all have the person-word as a
         PREFIX or a coincidental substring and denote no depicted person, so
         they pass. This is the "humanity" vs "human" decision: 'humanity' is
         an abstract noun that never names a subject in frame; 'superhuman'
         names one. It costs us a handful of harmless over-blocks
         ("roman", "specimen", "omen") which is the safe direction.

      C. PREFIX compound for "hand" only (COMPOUND_PREFIX_ROOTS): handheld,
         handmade, handbag. Declared explicitly rather than generalised,
         because a general prefix rule is what breaks "humanity"/"manhattan".

    Every field is scanned, including the source URL slug — for Pexels VIDEO
    results the slug is often the ONLY descriptive text the API returns
    (video items frequently ship an empty tags array), so it is load-bearing,
    not a nicety."""
    hits = []
    for fname, text in fields.items():
        if not text:
            continue
        for tok in tokenize(text):
            if tok in PERSON_EXACT:
                hits.append((fname, tok, "exact"))
                continue
            if len(tok) > 3 and any(tok.endswith(r) and tok != r
                                    for r in COMPOUND_SUFFIX_ROOTS):
                hits.append((fname, tok, "suffix"))
                continue
            if len(tok) > 4 and any(tok.startswith(r) and tok != r
                                    for r in COMPOUND_PREFIX_ROOTS):
                hits.append((fname, tok, "prefix"))
    # deterministic order
    return sorted(set(hits))


# ------------------------------------------------------------------ http seam
class HttpError(Exception):
    def __init__(self, code, body, url):
        super().__init__("HTTP %s for %s: %s" % (code, url, (body or "")[:200]))
        self.code = code


def _http_json(url: str, headers: dict = None, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers=dict({"User-Agent": UA}, **(headers or {})))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, e.read().decode("utf-8", "replace"), url)


def _http_bytes(url: str, headers: dict = None, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=dict({"User-Agent": UA}, **(headers or {})))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise HttpError(e.code, e.read().decode("utf-8", "replace"), url)


# ------------------------------------------------------------------- manifest
MANIFEST_VERSION = 1


def manifest_path(root: str) -> str:
    return os.path.join(root, "manifest.json")


def load_manifest(root: str) -> dict:
    p = manifest_path(root)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {"version": MANIFEST_VERSION, "assets": {}, "rejected": {}}


def save_manifest(m: dict, root: str) -> None:
    os.makedirs(root, exist_ok=True)
    tmp = manifest_path(root) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(m, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, manifest_path(root))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_date() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_provenance(root: str, key: str, raw: dict) -> str:
    d = os.path.join(root, "_provenance")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, re.sub(r"\W", "_", key) + ".json")
    with open(p, "w") as f:
        json.dump(raw, f, indent=1, sort_keys=True)
    return os.path.relpath(p, root)


# ---------------------------------------------------------------- Pexels side
PEXELS_LICENCE = {
    "id": "pexels",
    "name": "Pexels Licence",
    "url": "https://www.pexels.com/license/",
    "text": ("Free to use, no attribution required for the photo/video itself. "
             "Identifiable people and brands are NOT released: the Pexels "
             "Licence forbids using identifiable persons in a way that implies "
             "endorsement, and puts responsibility for any needed permission on "
             "the user. This engine therefore rejects all person content."),
}
# Pexels API Guidelines require API users to show a PROMINENT link back to
# Pexels and to credit the photographer where possible. That is an obligation
# per downloaded asset, so it is recorded per asset and surfaced by `credits`.
PEXELS_ATTRIBUTION_REQUIRED = True


def load_pexels_key() -> str:
    """Key from file (preferred, mode 600) or $PEXELS_KEY. Value never printed."""
    env = os.environ.get("PEXELS_KEY")
    if env:
        return env.strip()
    if os.path.exists(PEXELS_KEY_PATH):
        st = os.stat(PEXELS_KEY_PATH)
        if st.st_mode & 0o077:
            sys.stderr.write(
                "WARNING: %s is group/world readable (mode %o); chmod 600 it.\n"
                % (PEXELS_KEY_PATH, st.st_mode & 0o777))
        with open(PEXELS_KEY_PATH) as f:
            j = json.load(f)
        k = (j.get("key") or j.get("api_key") or "").strip()
        if k:
            return k
    raise SystemExit(
        "NO PEXELS KEY.\n"
        "  Put one at %s  ->  {\"key\": \"<your key>\"}  (chmod 600)\n"
        "  or export PEXELS_KEY=...\n"
        "  Free keys: https://www.pexels.com/api/  (sign in, 'Your API Key')\n"
        "  There is no fallback: the Pexels Terms of Service explicitly ban "
        "scraping and data-mining, so a keyless path would be a ToS breach, not "
        "a workaround." % PEXELS_KEY_PATH)


def pick_rendition(item: dict) -> dict:
    """Nearest rendition to 1080x1920, portrait-only, mp4 preferred.

    Deterministic: sorts on a pure function of the metadata, ties broken by
    file id, so the same API item always yields the same file."""
    files = [f for f in (item.get("video_files") or [])
             if f.get("width") and f.get("height") and f.get("link")]
    portrait = [f for f in files if int(f["height"]) > int(f["width"])]
    pool = portrait or files
    if not pool:
        return {}

    def cost(f):
        w, h = int(f["width"]), int(f["height"])
        # distance in log-resolution space so 2160x3840 is not "twice as wrong"
        # as 540x960; plus a small penalty for anything that is not mp4.
        d = abs(w - TARGET_W) / float(TARGET_W) + abs(h - TARGET_H) / float(TARGET_H)
        return (round(d, 6), 0 if (f.get("file_type") or "").endswith("mp4") else 1,
                int(f.get("id") or 0))

    return sorted(pool, key=cost)[0]


def pexels_search(query: str, n: int, api_base: str, key: str,
                  per_page: int = 40, page: int = 1) -> dict:
    url = "%s/videos/search?%s" % (api_base.rstrip("/"), urllib.parse.urlencode({
        "query": query,
        "orientation": "portrait",   # decision: portrait only, no crop roulette
        "size": "medium",            # >= HD; keeps 4K monsters out of the cache
        "per_page": per_page,
        "page": page,
    }))
    return _http_json(url, headers={"Authorization": key})


def fetch_video(root: str, query: str, category: str, n: int, api_base: str,
                dry_run: bool = False, face_check: bool = True,
                per_page: int = 40, retry_rejected: bool = False) -> dict:
    key = load_pexels_key()
    man = load_manifest(root)
    resp = pexels_search(query, n, api_base, key, per_page=per_page)
    items = resp.get("videos") or []

    # ---- filter + deterministic rank
    cands, rejected = [], []
    for it in items:
        iid = str(it.get("id"))
        fields = {
            "url": it.get("url") or "",
            "alt": it.get("alt") or "",
            "tags": " ".join(str(t.get("title") if isinstance(t, dict) else t)
                             for t in (it.get("tags") or [])),
            "user": (it.get("user") or {}).get("name") or "",
            "query": query,
        }
        # the uploader's name is a person's name by construction; scanning it
        # would block everything. It is recorded, not filtered.
        fields.pop("user")
        hits = person_block(fields)
        if hits:
            rejected.append((iid, hits, it))
            continue
        f = pick_rendition(it)
        if not f:
            rejected.append((iid, [("rendition", "none", "no usable file")], it))
            continue
        if int(f["height"]) <= int(f["width"]):
            rejected.append((iid, [("orientation", "landscape", "not portrait")], it))
            continue
        w, h = int(f["width"]), int(f["height"])
        score = abs(w - TARGET_W) / float(TARGET_W) + abs(h - TARGET_H) / float(TARGET_H)
        cands.append((round(score, 6), int(iid), it, f))
    cands.sort(key=lambda x: (x[0], x[1]))          # deterministic pick order

    outdir = os.path.join(root, "video", "stock_%s" % category)
    got, cached, skipped = [], [], []
    for _score, iid, it, f in cands:
        if len(got) + len(cached) >= n:
            break
        mkey = "pexels:%d" % iid
        if mkey in man["assets"] and os.path.exists(
                os.path.join(root, man["assets"][mkey]["path"])):
            cached.append(mkey)
            continue
        if mkey in man["rejected"] and not retry_rejected:
            # a clip rejected by the FACE check is only discoverable by
            # downloading it, so remember the verdict or every run pays for it
            # again (measured: without this the 2nd run re-fetched the face clip).
            continue
        dest = os.path.join(outdir, "pexels_%d.mp4" % iid)
        if dry_run:
            skipped.append((mkey, f["link"], dest))
            continue
        os.makedirs(outdir, exist_ok=True)
        data = _http_bytes(f["link"])
        with open(dest, "wb") as fh:
            fh.write(data)
        if face_check and has_face(dest):
            # belt and braces: the metadata filter is only as good as the
            # metadata. A silent face in an untagged "abstract" clip is exactly
            # the failure the licence exposure comes from, so look at pixels too.
            qdir = os.path.join(root, "_rejected", "faces")
            os.makedirs(qdir, exist_ok=True)
            shutil.move(dest, os.path.join(qdir, os.path.basename(dest)))
            man["rejected"][mkey] = {"reason": "face_detected", "url": it.get("url"),
                                     "query": query, "retrieved_utc": _utc_date()}
            continue
        rec = {
            "source": "pexels",
            "source_url": it.get("url"),
            "asset_id": str(iid),
            "asset_kind": "video",
            "licence_id": PEXELS_LICENCE["id"],
            "licence_name": PEXELS_LICENCE["name"],
            "licence_url": PEXELS_LICENCE["url"],
            "licence_text": PEXELS_LICENCE["text"],
            "retrieved_utc": _utc_date(),
            "query": query,
            "category": category,
            "path": os.path.relpath(dest, root),
            "sha256": sha256_file(dest),
            "bytes": os.path.getsize(dest),
            "width": int(f["width"]), "height": int(f["height"]),
            "duration_s": it.get("duration"),
            "creator": (it.get("user") or {}).get("name"),
            "creator_url": (it.get("user") or {}).get("url"),
            "attribution": {
                "required": PEXELS_ATTRIBUTION_REQUIRED,
                "reason": "Pexels API Guidelines: API users must show a "
                          "prominent link back to Pexels and credit the author.",
                "text": "Video by %s on Pexels" % ((it.get("user") or {}).get("name") or "Pexels"),
                "link": "https://www.pexels.com",
            },
            "person_filter": "passed",
            "provenance": None,
        }
        rec["provenance"] = write_provenance(root, mkey, it)
        man["assets"][mkey] = rec
        got.append(mkey)

    for iid, hits, it in rejected:
        man["rejected"]["pexels:%s" % iid] = {
            "reason": "person_words" if hits and hits[0][2] in ("exact", "suffix", "prefix")
                      else hits[0][2],
            "hits": [list(h) for h in hits], "url": it.get("url"),
            "query": query, "retrieved_utc": _utc_date()}
    if not dry_run:
        save_manifest(man, root)
        write_credits(root)
    return {"query": query, "category": category, "returned": len(items),
            "person_rejected": len([r for r in rejected if r[1] and r[1][0][2] in
                                    ("exact", "suffix", "prefix")]),
            "other_rejected": len([r for r in rejected if r[1] and r[1][0][2] not in
                                   ("exact", "suffix", "prefix")]),
            "downloaded": got, "cached": cached, "dry_run": skipped}


def has_face(path: str, samples: int = 12) -> bool:
    """Sample frames and run the same YuNet detector the cataloger uses.

    Analyzer-side only (cv2) — never called from render_edit.py."""
    try:
        import cv2
    except ImportError:
        return False
    mp = os.path.join(HERE, "assets", "yunet.onnx")
    if not os.path.exists(mp):
        mp = os.path.join(os.path.dirname(HERE), "assets", "yunet.onnx")
    if not os.path.exists(mp):
        return False
    det = cv2.FaceDetectorYN.create(mp, "", (320, 320), 0.6)
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = [int(i * max(1, n - 1) / max(1, samples - 1)) for i in range(samples)] if n else []
    found = False
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        h, w = fr.shape[:2]
        sw = 320
        sh_ = max(1, int(sw * h / w))
        small = cv2.resize(fr, (sw, sh_))
        det.setInputSize((sw, sh_))
        _, faces = det.detect(small)
        if faces is not None and len(faces):
            found = True
            break
    cap.release()
    return found


# ------------------------------------------------------------- Openverse side
CC0 = {
    "id": "cc0",
    "name": "CC0 1.0 Universal (Public Domain Dedication)",
    "url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "text": ("The rights holder waived all copyright and related rights "
             "worldwide. No attribution is legally required. CC0 waives "
             "COPYRIGHT ONLY: it does not clear personality, publicity, "
             "trademark or privacy rights, which is a second reason this "
             "engine keeps identifiable people out of fetched assets."),
}


def _quota_path(root: str) -> str:
    return os.path.join(root, "_openverse_quota.json")


class OpenverseQuota:
    """Politeness + the measured anonymous limits, persisted across runs."""

    def __init__(self, root: str):
        self.root = root
        self.state = {"date": time.strftime("%Y-%m-%d", time.gmtime()),
                      "count": 0, "last": 0.0}
        p = _quota_path(root)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    s = json.load(f)
                if s.get("date") == self.state["date"]:
                    self.state = s
            except Exception:
                pass

    def wait(self):
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if today != self.state["date"]:
            self.state = {"date": today, "count": 0, "last": 0.0}
        if self.state["count"] >= OV_DAILY_CAP:
            raise SystemExit(
                "OPENVERSE DAILY QUOTA REACHED (%d requests today; anonymous "
                "sustained limit measured at ~200/day).\n"
                "  Wait for UTC midnight, or register a free Openverse API key "
                "at https://api.openverse.org/v1/auth_tokens/register/ and raise "
                "the cap. No key is registered by this tool on your behalf."
                % self.state["count"])
        dt = time.time() - float(self.state["last"])
        if dt < OV_MIN_INTERVAL:
            time.sleep(OV_MIN_INTERVAL - dt)

    def note(self):
        self.state["count"] += 1
        self.state["last"] = time.time()
        os.makedirs(self.root, exist_ok=True)
        with open(_quota_path(self.root), "w") as f:
            json.dump(self.state, f)


def openverse_search(query: str, page_size: int, api_base: str, quota=None,
                     page: int = 1) -> dict:
    url = "%s/audio/?%s" % (api_base.rstrip("/"), urllib.parse.urlencode({
        "q": query,
        "license": "cc0",          # server-side filter ...
        "page_size": page_size,
        "page": page,
    }))
    if quota:
        quota.wait()
    try:
        r = _http_json(url)
    except HttpError as e:
        if e.code == 429:
            raise SystemExit(
                "OPENVERSE RATE LIMIT (HTTP 429). Anonymous burst is ~20/min, "
                "sustained ~200/day. Slow down or come back tomorrow; this tool "
                "will not create an account or key for you.")
        raise
    finally:
        if quota:
            quota.note()
    return r


def _tags_of(item: dict) -> list:
    out = []
    for t in item.get("tags") or []:
        out.append(t.get("name") if isinstance(t, dict) else str(t))
    return [t for t in out if t]


SFX_MAX_BYTES = 8 * 1024 * 1024   # a one-shot / short bed, not a 10-minute field
                                  # recording: measured 76 KB/s through the proxy,
                                  # and long beds are the wrong shape for a 10s reel.


def fetch_sfx(root: str, query: str, n: int, api_base: str, dry_run: bool = False,
              normalize: bool = True, page_size: int = 20,
              allow_person_words: bool = False, max_bytes: int = SFX_MAX_BYTES,
              retry_rejected: bool = False) -> dict:
    man = load_manifest(root)
    quota = OpenverseQuota(root)
    resp = openverse_search(query, page_size, api_base, quota)
    items = resp.get("results") or []

    cands, rejected = [], []
    for it in items:
        iid = str(it.get("id"))
        # RE-VERIFY the licence on the item itself. The server-side filter is a
        # query parameter, i.e. a request; this is the check on the answer.
        lic = (it.get("license") or "").strip().lower()
        if lic != "cc0":
            rejected.append((iid, [("license", lic or "none", "not cc0")], it))
            continue
        if not it.get("url"):
            rejected.append((iid, [("url", "none", "no media url")], it))
            continue
        if not allow_person_words:
            hits = person_block({"title": it.get("title"),
                                 "tags": " ".join(_tags_of(it)),
                                 "query": query})
            if hits:
                rejected.append((iid, hits, it))
                continue
        fsz = int(it.get("filesize") or 0)
        if max_bytes and fsz > max_bytes:
            rejected.append((iid, [("filesize", str(fsz), "over max_bytes")], it))
            continue
        cands.append((fsz, iid, it))
    # deterministic AND useful: smallest first (short sounds beat long beds),
    # ties broken by the stable Openverse uuid.
    cands.sort(key=lambda x: (x[0], x[1]))

    got, cached, skipped, measures = [], [], [], []
    for _fsz, iid, it in cands:
        if len(got) + len(cached) >= n:
            break
        mkey = "openverse:%s" % iid
        if mkey in man["assets"] and os.path.exists(
                os.path.join(root, man["assets"][mkey]["path"])):
            cached.append(mkey)
            continue
        if mkey in man["rejected"] and not retry_rejected:
            continue
        ext = (it.get("filetype") or "mp3").lower()
        rawdir = os.path.join(root, "_raw", "sfx")
        outdir = os.path.join(root, "sfx")
        raw = os.path.join(rawdir, "%s.%s" % (iid, ext))
        dest = os.path.join(outdir, "%s.wav" % iid)
        if dry_run:
            skipped.append((mkey, it.get("url"), dest))
            continue
        os.makedirs(rawdir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)
        with open(raw, "wb") as fh:
            fh.write(_http_bytes(it["url"], timeout=300))
        before = measure_audio(raw)
        if normalize:
            normalize_sfx(raw, dest)
        else:
            shutil.copy2(raw, dest)
        after = measure_audio(dest)
        measures.append({"id": iid, "before": before, "after": after})
        rec = {
            "source": "openverse",
            "source_url": it.get("foreign_landing_url") or it.get("url"),
            "media_url": it.get("url"),
            "asset_id": iid,
            "asset_kind": "sfx",
            "licence_id": "cc0",
            "licence_name": CC0["name"],
            "licence_url": it.get("license_url") or CC0["url"],
            "licence_text": CC0["text"],
            "licence_verified_on_item": True,
            "retrieved_utc": _utc_date(),
            "query": query,
            "category": "sfx",
            "path": os.path.relpath(dest, root),
            "raw_path": os.path.relpath(raw, root),
            "sha256": sha256_file(dest),
            "sha256_raw": sha256_file(raw),
            "bytes": os.path.getsize(dest),
            "title": it.get("title"),
            "creator": it.get("creator"),
            "creator_url": it.get("creator_url"),
            "provider": it.get("provider"),
            "attribution": {
                "required": False,
                "reason": "CC0 1.0 waives all rights; no attribution legally "
                          "required. Credit kept voluntarily and for audit.",
                "text": "%s by %s (%s, CC0)" % (it.get("title"), it.get("creator"),
                                                it.get("provider")),
                "link": it.get("foreign_landing_url"),
            },
            "audio": {"before": before, "after": after,
                      "normalized": bool(normalize)},
            "provenance": None,
        }
        rec["provenance"] = write_provenance(root, mkey, it)
        man["assets"][mkey] = rec
        got.append(mkey)

    for iid, hits, it in rejected:
        man["rejected"]["openverse:%s" % iid] = {
            "reason": hits[0][2], "hits": [list(h) for h in hits],
            "url": it.get("foreign_landing_url"), "query": query,
            "retrieved_utc": _utc_date()}
    if not dry_run:
        save_manifest(man, root)
        write_credits(root)
    return {"query": query, "returned": len(items), "downloaded": got,
            "cached": cached, "rejected": len(rejected), "measures": measures,
            "dry_run": skipped}


# ------------------------------------------------------------ audio treatment
# Measured reality of CC0 Freesound previews (10-file sample):
#   integrated loudness spread 26.6 LU;  lead silence 2 ms .. 1573 ms;
#   2/10 files peaked above 0 dBTP;  mixed sample rates and channel counts;
#   all of them lossy CDN previews.
# So every fetched SFX is rewritten once, at fetch time, into one house format.
# The renderer then only has to place and gain it — which matters, because
# render_edit.py is stdlib+ffmpeg only and must stay dumb.
SFX_TARGET_LUFS = -20.0     # sits under a -16 LUFS VO without ducking
SFX_TARGET_TP = -1.5        # dBTP ceiling before the limiter
SFX_LIMIT = 0.85            # alimiter ceiling, linear (~-1.4 dBTP)
SFX_LEAD_KEEP = 0.02        # seconds of pre-roll kept after trimming


def _ff(args, timeout=300):
    return subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-y"] + args,
                          capture_output=True, text=True, timeout=timeout)


def measure_audio(path: str) -> dict:
    """LUFS-I, true peak, lead silence, sr/ch/duration — one ffmpeg pass each."""
    out = {}
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                        "-of", "json", path], capture_output=True, text=True)
    try:
        j = json.loads(r.stdout)
        a = next(s for s in j["streams"] if s["codec_type"] == "audio")
        out.update(sr=int(a.get("sample_rate", 0)), ch=int(a.get("channels", 0)),
                   codec=a.get("codec_name"),
                   dur=round(float(j["format"].get("duration") or 0), 3))
    except Exception:
        out.update(sr=0, ch=0, codec=None, dur=0.0)
    r = _ff(["-i", path, "-af", "ebur128=peak=true", "-f", "null", "-"])
    tail = r.stderr[r.stderr.rfind("Summary:"):]
    lines = tail.splitlines()
    lufs = tp = None
    for i, L in enumerate(lines):
        if L.strip().startswith("I:") and lufs is None:
            try:
                lufs = float(L.split()[1])
            except Exception:
                pass
        if "True peak" in L:
            for L2 in lines[i:]:
                if L2.strip().startswith("Peak:"):
                    try:
                        tp = float(L2.split()[1])
                    except Exception:
                        pass
                    break
    out["lufs"] = lufs
    out["tp_dbtp"] = tp
    out["lead_silence_ms"] = lead_silence_ms(path)
    return out


def lead_silence_ms(path: str, thresh_db: float = -50.0) -> float:
    """Milliseconds before the first sample above thresh (silencedetect)."""
    r = _ff(["-i", path, "-af", "silencedetect=n=%gdB:d=0.001" % thresh_db,
             "-f", "null", "-"])
    m = re.search(r"silence_start:\s*(-?[\d.]+)", r.stderr)
    if not m or abs(float(m.group(1))) > 1e-6:
        return 0.0                     # no silence detected at t=0
    e = re.search(r"silence_end:\s*([\d.]+)", r.stderr)
    return round(float(e.group(1)) * 1000.0, 1) if e else 0.0


def normalize_sfx(src: str, dst: str) -> dict:
    """Trim lead silence -> two-pass loudnorm -> limiter -> 48k/stereo/pcm_s16.

    Two-pass because single-pass loudnorm is a dynamic-mode guess; the second
    pass is linear and predictable, which is what determinism needs."""
    trim = ("silenceremove=start_periods=1:start_duration=%g:start_threshold=-50dB:"
            "detection=peak" % SFX_LEAD_KEEP)
    base = "%s,aresample=48000" % trim
    r = _ff(["-i", src, "-af", "%s,loudnorm=I=%g:TP=%g:LRA=11:print_format=json"
             % (base, SFX_TARGET_LUFS, SFX_TARGET_TP), "-f", "null", "-"])
    m = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    meas = json.loads(m[-1]) if m else None
    ln = "loudnorm=I=%g:TP=%g:LRA=11" % (SFX_TARGET_LUFS, SFX_TARGET_TP)
    if meas:
        ln += (":measured_I=%s:measured_TP=%s:measured_LRA=%s:measured_thresh=%s"
               ":offset=%s:linear=true" % (meas["input_i"], meas["input_tp"],
                                           meas["input_lra"], meas["input_thresh"],
                                           meas["target_offset"]))
    chain = "%s,%s,alimiter=limit=%g:level=disabled,aresample=48000" % (base, ln, SFX_LIMIT)
    rr = _ff(["-i", src, "-af", chain, "-ar", "48000", "-ac", "2",
              "-c:a", "pcm_s16le", dst])
    if rr.returncode != 0:
        raise RuntimeError("sfx normalize failed: %s" % rr.stderr[-800:])
    return meas or {}


# ---------------------------------------------------------------- credits/verify
def write_credits(root: str) -> str:
    """Caption-ready attribution block. Pexels API use REQUIRES the link."""
    man = load_manifest(root)
    req = [a for a in man["assets"].values() if a.get("attribution", {}).get("required")]
    opt = [a for a in man["assets"].values() if not a.get("attribution", {}).get("required")]
    lines = ["# Stock attribution", "",
             "Generated %s from manifest.json — do not edit by hand." % _utc_date(), ""]
    if req:
        lines += ["## REQUIRED (must appear in the caption / description)", ""]
        srcs = sorted(set(a["source"] for a in req))
        for s in srcs:
            names = sorted(set(a["attribution"]["text"] for a in req if a["source"] == s))
            lines.append("Stock footage from %s (https://www.pexels.com)" % s.title()
                         if s == "pexels" else "Stock from %s" % s)
            for nm in names:
                lines.append("  - %s" % nm)
        lines.append("")
        lines += ["Caption one-liner: `Stock: Pexels (pexels.com)`", ""]
    if opt:
        lines += ["## Voluntary (CC0 — no legal obligation)", ""]
        for a in sorted(opt, key=lambda x: x["asset_id"]):
            lines.append("  - %s" % a["attribution"]["text"])
        lines.append("")
    p = os.path.join(root, "ATTRIBUTION.md")
    os.makedirs(root, exist_ok=True)
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    return p


def verify(root: str) -> dict:
    man = load_manifest(root)
    bad, ok, missing = [], 0, []
    for k, a in sorted(man["assets"].items()):
        p = os.path.join(root, a["path"])
        if not os.path.exists(p):
            missing.append(k)
            continue
        if sha256_file(p) != a["sha256"]:
            bad.append(k)
        else:
            ok += 1
    return {"ok": ok, "hash_mismatch": bad, "missing": missing,
            "assets": len(man["assets"]), "rejected": len(man.get("rejected", {}))}


# ---------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("video", help="Pexels portrait texture/atmosphere b-roll")
    v.add_argument("--category", default="sky", choices=sorted(CATEGORIES))
    v.add_argument("--query", help="override the category's canned query")
    v.add_argument("--n", type=int, default=4)
    v.add_argument("--per-page", type=int, default=40)
    v.add_argument("--root", default=DEFAULT_ROOT)
    v.add_argument("--api-base", default=PEXELS_API)
    v.add_argument("--dry-run", action="store_true")
    v.add_argument("--no-face-check", dest="face_check", action="store_false")
    v.add_argument("--retry-rejected", action="store_true",
                   help="re-consider assets a previous run rejected (e.g. after "
                        "widening the allow-list); off by default so a known-bad "
                        "clip is never downloaded twice")

    s = sub.add_parser("sfx", help="Openverse CC0 audio (keyless)")
    s.add_argument("--query", required=True)
    s.add_argument("--n", type=int, default=3)
    s.add_argument("--page-size", type=int, default=20)
    s.add_argument("--root", default=DEFAULT_ROOT)
    s.add_argument("--api-base", default=OPENVERSE_API)
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--no-normalize", dest="normalize", action="store_false")
    s.add_argument("--allow-person-words", action="store_true")
    s.add_argument("--max-bytes", type=int, default=SFX_MAX_BYTES)
    s.add_argument("--retry-rejected", action="store_true")

    c = sub.add_parser("credits"); c.add_argument("--root", default=DEFAULT_ROOT)
    y = sub.add_parser("verify"); y.add_argument("--root", default=DEFAULT_ROOT)
    b = sub.add_parser("blocktest", help="run the person filter over stdin lines")
    b.add_argument("--text", action="append", default=[])

    a = ap.parse_args(argv)
    root = os.path.expanduser(getattr(a, "root", DEFAULT_ROOT))

    if a.cmd == "video":
        q = a.query or CATEGORIES[a.category]
        r = fetch_video(root, q, a.category, a.n, a.api_base, a.dry_run,
                        a.face_check, a.per_page, a.retry_rejected)
        print(json.dumps(r, indent=1, sort_keys=True))
    elif a.cmd == "sfx":
        r = fetch_sfx(root, a.query, a.n, a.api_base, a.dry_run, a.normalize,
                      a.page_size, a.allow_person_words, a.max_bytes,
                      a.retry_rejected)
        print(json.dumps(r, indent=1, sort_keys=True))
    elif a.cmd == "credits":
        print(write_credits(root))
    elif a.cmd == "verify":
        r = verify(root)
        print(json.dumps(r, indent=1, sort_keys=True))
        return 0 if not (r["hash_mismatch"] or r["missing"]) else 1
    elif a.cmd == "blocktest":
        for t in a.text:
            h = person_block({"text": t})
            print("%-46s %s" % (t, "BLOCK %s" % h if h else "allow"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
