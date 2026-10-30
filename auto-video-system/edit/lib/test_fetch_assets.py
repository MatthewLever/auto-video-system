#!/usr/bin/env python3
"""test_fetch_assets.py — offline tests for the stock fetcher.

Pexels cannot be tested live (no key, and the ToS bans any keyless path), so the
ENTIRE Pexels code path is driven here against a local fixture HTTP server:
same code, same urllib calls, same header, just a different --api-base. What
this proves: request shape (path + query params + Authorization header),
orientation filtering, person-word blocking on real-shaped metadata, rendition
selection, manifest content, caching, and determinism of the pick order.
What it does NOT prove: that api.pexels.com answers the way the fixture does.

Run:  python3 test_fetch_assets.py
"""
import http.server
import json
import os
import shutil
import socketserver
import sys
import tempfile
import threading
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_assets as FA  # noqa: E402

REQUESTS = []           # every request the fixture server saw
FIXTURE_MP4 = os.path.join("/root/engine/test/footage/Portrait", "c01_sharp_static.mp4")
FACE_MP4 = os.path.join("/root/engine/test/footage/Movement", "c10_face.mp4")

FAILURES = []


def check(name, cond, detail=""):
    print("%-58s %s %s" % (name, "PASS" if cond else "FAIL", detail))
    if not cond:
        FAILURES.append(name)


# ------------------------------------------------------------------ fixtures
def vfile(fid, w, h, name):
    return {"id": fid, "quality": "hd", "file_type": "video/mp4",
            "width": w, "height": h, "link": "http://%s/file/%s" % (ADDR, name)}


def build_fixture():
    return {"page": 1, "per_page": 40, "total_results": 4, "videos": [
        # 0 — clean texture, three renditions: 720x1280 / 1080x1920 / 2160x3840.
        {"id": 1001, "width": 2160, "height": 3840, "duration": 14,
         "url": "https://www.pexels.com/video/time-lapse-of-clouds-at-sunset-1001/",
         "alt": "time lapse of clouds at sunset", "tags": ["sky", "clouds", "sunset"],
         "user": {"name": "Ada Texture", "url": "https://www.pexels.com/@ada"},
         "video_files": [vfile(9, 720, 1280, "ok.mp4"),
                         vfile(7, 1080, 1920, "ok.mp4"),
                         vfile(8, 2160, 3840, "ok.mp4")]},
        # 1 — ADVERSARIAL: innocuous title, person-word only in the tags.
        {"id": 1002, "width": 1080, "height": 1920, "duration": 9,
         "url": "https://www.pexels.com/video/abstract-texture-1002/",
         "alt": "abstract texture", "tags": ["abstract", "woman hands", "macro"],
         "user": {"name": "Bo", "url": "https://www.pexels.com/@bo"},
         "video_files": [vfile(11, 1080, 1920, "person.mp4")]},
        # 2 — ADVERSARIAL: person-word ONLY in the URL slug (Pexels video items
        #     routinely return an empty tags array; the slug is all there is).
        {"id": 1003, "width": 1080, "height": 1920, "duration": 6,
         "url": "https://www.pexels.com/video/a-man-walking-through-fog-1003/",
         "alt": "", "tags": [],
         "user": {"name": "Cy", "url": "https://www.pexels.com/@cy"},
         "video_files": [vfile(12, 1080, 1920, "person.mp4")]},
        # 3 — landscape-only renditions: must be dropped even though the API
        #     was asked for orientation=portrait (never trust the server).
        {"id": 1004, "width": 1920, "height": 1080, "duration": 8,
         "url": "https://www.pexels.com/video/asphalt-road-texture-1004/",
         "alt": "asphalt road texture", "tags": ["asphalt", "road"],
         "user": {"name": "Di", "url": "https://www.pexels.com/@di"},
         "video_files": [vfile(13, 1920, 1080, "ok.mp4")]},
        # 4 — clean, 1440x2560; second-best rendition distance -> ranked after 1001.
        {"id": 1005, "width": 1440, "height": 2560, "duration": 11,
         "url": "https://www.pexels.com/video/rain-on-a-window-1005/",
         "alt": "rain on a window", "tags": ["rain", "window", "water"],
         "user": {"name": "Ez", "url": "https://www.pexels.com/@ez"},
         "video_files": [vfile(14, 1440, 2560, "ok.mp4")]},
        # 5 — clean metadata but a FACE in the pixels (the metadata filter's blind spot)
        {"id": 1006, "width": 1080, "height": 1920, "duration": 5,
         "url": "https://www.pexels.com/video/soft-light-study-1006/",
         "alt": "soft light study", "tags": ["light", "bokeh"],
         "user": {"name": "Fi", "url": "https://www.pexels.com/@fi"},
         "video_files": [vfile(15, 1080, 1920, "face.mp4")]},
    ]}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        REQUESTS.append({"path": u.path, "qs": urllib.parse.parse_qs(u.query),
                         "auth": self.headers.get("Authorization"),
                         "ua": self.headers.get("User-Agent")})
        if u.path == "/videos/search":
            if not self.headers.get("Authorization"):
                self.send_error(401, "no key")
                return
            body = json.dumps(build_fixture()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path.startswith("/file/"):
            name = u.path.split("/")[-1]
            src = FACE_MP4 if name == "face.mp4" else FIXTURE_MP4
            data = open(src, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)


def start_server():
    srv = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, "127.0.0.1:%d" % srv.server_address[1]


# ------------------------------------------------------------------- tests
def test_person_block():
    print("\n== person-word blocker ==")
    cases = [
        # (text, should_block, why)
        ("abstract texture", False, "clean"),
        ("sunset clouds timelapse", False, "clean"),
        ("concrete wall texture asphalt", False, "clean"),
        ("ocean water waves surface", False, "'surface' must NOT trip 'face'"),
        ("woman hands", True, "required word, plural"),
        ("a clip tagged: abstract texture / woman hands", True, "adversarial mixed"),
        ("man", True, "required"), ("men", True, "irregular plural"),
        ("women", True, "irregular plural"), ("children", True, "irregular plural"),
        ("kids", True, "generated plural"), ("crowds", True, "generated plural"),
        ("families", True, "y->ies plural"), ("businesses", True, "es plural"),
        ("human", True, "required"),
        ("humanity", False, "abstract noun: prefix-only match must not fire"),
        ("superhuman", True, "person-word is the HEAD of the compound"),
        ("manhattan", False, "coincidental substring"),
        ("manhattan skyline at night", False, "real query we would run"),
        ("management", False, "prefix, not head"),
        ("manual", False, "prefix, not head"),
        ("manufacturing", False, "prefix, not head"),
        ("mango", False, "prefix, not head"),
        ("germany", False, "coincidental"),
        ("businessman", True, "compound head"),
        ("schoolgirl", True, "compound head"),
        ("coworker", True, "compound head"),
        ("supermodel", True, "compound head"),
        ("salesperson", True, "compound head"),
        ("handheld", True, "hand-prefix rule (declared, deliberate over-block)"),
        ("handmade", True, "hand-prefix rule"),
        ("https://www.pexels.com/video/a-woman-walking-on-the-beach-1234567/", True,
         "URL slug is scanned"),
        ("https://www.pexels.com/video/time-lapse-of-clouds-1001/", False, "clean slug"),
        ("office", True, "required"), ("team", True, "required"),
        ("business", True, "required"), ("model", True, "required"),
        ("staff", True, "required"), ("couple", True, "required"),
        ("worker", True, "required"), ("family", True, "required"),
        ("portrait", True, "required"), ("face", True, "required"),
        ("boy", True, "required"), ("girl", True, "required"),
        ("child", True, "required"), ("kid", True, "required"),
        ("people", True, "required"), ("person", True, "required"),
    ]
    bad = []
    for text, want, why in cases:
        got = bool(FA.person_block({"t": text}))
        if got != want:
            bad.append((text, want, got, why))
    for b in bad:
        print("   MISMATCH %r want_block=%s got=%s (%s)" % b)
    check("person_block: %d/%d adversarial cases" % (len(cases) - len(bad), len(cases)),
          not bad)
    # every word the brief REQUIRES must be covered
    required = ["person", "people", "man", "men", "woman", "women", "girl", "boy",
                "child", "kid", "human", "face", "portrait", "hands", "hand",
                "team", "business", "office", "model", "crowd", "family",
                "couple", "worker", "staff"]
    miss = [w for w in required if not FA.person_block({"t": w})]
    check("person_block: all 24 required words blocked", not miss, str(miss))


def test_pexels(root):
    print("\n== Pexels path (fixture server) ==")
    os.environ["PEXELS_KEY"] = "TEST-KEY-DO-NOT-LOG"
    api = "http://%s" % ADDR
    r1 = FA.fetch_video(root, "sunset sky clouds timelapse", "sky", 2, api)

    # --- request shape
    req = [q for q in REQUESTS if q["path"] == "/videos/search"][0]
    check("request path /videos/search", req["path"] == "/videos/search")
    check("orientation=portrait sent", req["qs"].get("orientation") == ["portrait"])
    check("size sent", req["qs"].get("size") == ["medium"])
    check("per_page sent", req["qs"].get("per_page") == ["40"])
    check("query sent verbatim", req["qs"].get("query") == ["sunset sky clouds timelapse"])
    check("Authorization header carries the key",
          req["auth"] == "TEST-KEY-DO-NOT-LOG")

    man = FA.load_manifest(root)
    ids = sorted(man["assets"])
    check("2 assets downloaded", len(man["assets"]) == 2, str(ids))
    check("person-tagged 1002 rejected", "pexels:1002" in man["rejected"],
          str(man["rejected"].get("pexels:1002", {}).get("hits")))
    check("slug-only 1003 rejected", "pexels:1003" in man["rejected"],
          str(man["rejected"].get("pexels:1003", {}).get("hits")))
    check("landscape 1004 rejected", "pexels:1004" in man["rejected"])
    check("1002/1003 never written to disk",
          not any(os.path.exists(os.path.join(root, "video", "stock_sky", "pexels_%d.mp4" % i))
                  for i in (1002, 1003)))

    # --- rendition: 1080x1920 chosen over 720x1280 and 2160x3840
    a = man["assets"]["pexels:1001"]
    check("rendition nearest 1080x1920", (a["width"], a["height"]) == (1080, 1920),
          "%sx%s" % (a["width"], a["height"]))
    check("rank: 1001 before 1005", ids == ["pexels:1001", "pexels:1005"], str(ids))

    # --- manifest record completeness
    need = ["source", "source_url", "asset_id", "licence_id", "licence_url",
            "licence_text", "retrieved_utc", "query", "path", "sha256", "attribution"]
    miss = [k for k in need if not a.get(k)]
    check("manifest has every mandatory field", not miss, str(miss))
    check("attribution obligation recorded for Pexels",
          a["attribution"]["required"] is True and "pexels.com" in a["attribution"]["link"])
    check("sha256 matches file on disk",
          FA.sha256_file(os.path.join(root, a["path"])) == a["sha256"])
    check("key value NEVER in manifest",
          "TEST-KEY-DO-NOT-LOG" not in json.dumps(man))
    check("provenance snapshot written",
          os.path.exists(os.path.join(root, a["provenance"])))
    check("ATTRIBUTION.md written with required credit",
          "pexels.com" in open(os.path.join(root, "ATTRIBUTION.md")).read().lower())

    # --- caching: second identical call downloads nothing
    n_before = len([q for q in REQUESTS if q["path"].startswith("/file/")])
    r2 = FA.fetch_video(root, "sunset sky clouds timelapse", "sky", 2, api)
    n_after = len([q for q in REQUESTS if q["path"].startswith("/file/")])
    check("cache: no re-download (incl. the face-rejected clip)",
          n_after == n_before and r2["downloaded"] == [],
          "%d file GETs before / %d after, cached=%s" % (n_before, n_after, r2["cached"]))
    check("cache: keyed by source+id", sorted(r2["cached"]) == ["pexels:1001", "pexels:1005"])

    # --- determinism: same query, same library -> same picks, same manifest bytes
    m1 = open(FA.manifest_path(root), "rb").read()
    FA.fetch_video(root, "sunset sky clouds timelapse", "sky", 2, api)
    m2 = open(FA.manifest_path(root), "rb").read()
    # retrieval timestamps of REJECTED items move; assets must not
    a1 = json.loads(m1)["assets"]; a2 = json.loads(m2)["assets"]
    check("determinism: asset records byte-identical across runs",
          json.dumps(a1, sort_keys=True) == json.dumps(a2, sort_keys=True))

    # --- face check on clean metadata
    print("\n== face-check quarantine (metadata says 'soft light study') ==")
    FA.fetch_video(root, "abstract light bokeh blur", "light", 6, api)
    man = FA.load_manifest(root)
    check("face clip 1006 quarantined, not catalogued",
          "pexels:1006" not in man["assets"] and
          man["rejected"].get("pexels:1006", {}).get("reason") == "face_detected",
          str(man["rejected"].get("pexels:1006")))
    check("face clip moved to _rejected/faces",
          os.path.exists(os.path.join(root, "_rejected", "faces", "pexels_1006.mp4")))

    # --- no key -> hard failure, no fallback
    print("\n== no-key behaviour ==")
    del os.environ["PEXELS_KEY"]
    old = FA.PEXELS_KEY_PATH
    FA.PEXELS_KEY_PATH = "/nonexistent/pexels_key.json"
    try:
        FA.fetch_video(root, "x", "sky", 1, api)
        check("no key -> SystemExit", False, "did not raise")
    except SystemExit as e:
        msg = str(e)
        check("no key -> SystemExit with signup URL",
              "pexels.com/api" in msg and "scraping" in msg)
    finally:
        FA.PEXELS_KEY_PATH = old
        os.environ["PEXELS_KEY"] = "TEST-KEY-DO-NOT-LOG"

    # --- verify subcommand
    v = FA.verify(root)
    check("verify: all hashes good", not v["hash_mismatch"] and not v["missing"], str(v))


def main():
    global ADDR
    srv, ADDR = start_server()
    root = tempfile.mkdtemp(prefix="stocktest_")
    try:
        test_person_block()
        test_pexels(root)
    finally:
        srv.shutdown()
        shutil.rmtree(root, ignore_errors=True)
    print("\n%s  (%d failures)" % ("ALL PASS" if not FAILURES else "FAILURES: %s" % FAILURES,
                                   len(FAILURES)))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
