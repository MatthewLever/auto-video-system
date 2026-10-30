#!/usr/bin/env python3
"""contact_sheet.py — one-time semantic-tagging sheet: 1 representative frame per clip
(best-window midpoint), labeled with id/dur/metrics/flags + timeline bar. <=12 clips per 1280px sheet."""
import cv2, json, sys, os
from PIL import Image, ImageDraw, ImageFont

B = "/root/engine/research/b"
CAT, CLIPDIR, OUT = sys.argv[1:4] if len(sys.argv) > 3 else (f"{B}/out/catalog.json", f"{B}/corpus", f"{B}/out/sheet")
cat = json.load(open(CAT))
COLS, TILE_W, LABEL_H = 4, 320, 52
F1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
F2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf", 14)

def grab(path, t):
    cap = cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_MSEC, max(0, t * 1000))
    ok, fr = cap.read()
    if not ok: cap.set(cv2.CAP_PROP_POS_MSEC, 0); ok, fr = cap.read()
    cap.release(); return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

pages = [cat[i:i + 12] for i in range(0, len(cat), 12)]
for pi, page in enumerate(pages):
    rows = (len(page) + COLS - 1) // COLS
    fh = 554                                    # fixed portrait frame box (312x554 ~ 9:16)
    tile_h = fh + LABEL_H + 14
    sheet = Image.new("RGB", (COLS * TILE_W, rows * tile_h), (16, 16, 20))
    dr = ImageDraw.Draw(sheet)
    for i, e in enumerate(page):
        cx, cy = (i % COLS) * TILE_W, (i // COLS) * tile_h
        mid = (e["hero"][0] + e["hero"][1]) / 2 if e["hero"] else e["d"] / 2
        fr = grab(e["path"] if e.get("path") else os.path.join(CLIPDIR, e["f"]), mid)
        img = Image.fromarray(fr)
        sc = min((TILE_W - 8) / img.width, fh / img.height)
        img = img.resize((int(img.width * sc), int(img.height * sc)), Image.LANCZOS)
        sheet.paste(img, (cx + 4 + (TILE_W - 8 - img.width) // 2, cy + 4 + (fh - img.height) // 2))
        # timeline bar: grey=clip, green=hero, red=dead, yellow=hold, white tick=cut
        ty = cy + 4 + fh - 10
        dr.rectangle([cx + 8, ty, cx + TILE_W - 12, ty + 6], fill=(70, 70, 70))
        px = lambda t: cx + 8 + (TILE_W - 20) * min(1, t / e["d"])
        if e.get("hold"): dr.rectangle([px(e["hold"][0]), ty, px(e["hold"][1]), ty + 6], fill=(220, 180, 40))
        if e.get("dead"): dr.rectangle([px(e["dead"][0]), ty, px(e["dead"][1]), ty + 6], fill=(200, 60, 50))
        if e["hero"]: dr.rectangle([px(e["hero"][0]), ty, px(e["hero"][1]), ty + 6], fill=(80, 220, 100))
        for c in e["cuts"]: dr.rectangle([px(c) - 1, ty - 2, px(c) + 1, ty + 8], fill=(255, 255, 255))
        flags = []
        if e["junk"]: flags.append("JUNK")
        if e["face"]: flags.append("FACE")
        if e.get("static"): flags.append("STATIC")
        if e.get("rot"): flags.append(f"ROT{e['rot']}")
        if e["cuts"]: flags.append("CUT@" + ",".join(str(c) for c in e["cuts"]))
        if e.get("hold"): flags.append(f"HOLD{e['hold'][0]}-{e['hold'][1]}")
        if e.get("dead"): flags.append(f"DEAD{e['dead'][0]}-{e['dead'][1]}")
        l1 = f"{e['id']}  {e['d']}s  " + (f"hero {e['hero'][0]}-{e['hero'][1]} q={e['hero'][2]:.2f}" if e["hero"] else "no usable window")
        l2 = f"sh{e['q']['sharp']} jit{e['q']['jit']} mot{e['q']['mot']} clip{int(e['q']['clip']*100)}% " + " ".join(flags)
        dr.text((cx + 8, cy + fh + 10), l1, font=F1, fill=(240, 240, 240))
        dr.text((cx + 8, cy + fh + 32), l2, font=F2, fill=(180, 200, 255))
    out = f"{OUT}_{pi + 1}.jpg"
    sheet.save(out, quality=88)
    print("wrote", out, sheet.size)
