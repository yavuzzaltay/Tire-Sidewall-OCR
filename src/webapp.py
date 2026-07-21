"""Cok basit yerel web arayuzu: Lastigin_ustu/Lastigin_alti fotolarini goster,
istenenleri sec (+ hepsini sec), pipeline'i adim adim calistir, her asamanin
gorselini + performans/sure bilgisini ekranda goster.

Calistirma: python src/webapp.py  ->  http://127.0.0.1:5000
"""
from __future__ import annotations

import glob
import os
import re
import sys
import time
import uuid

import cv2
import numpy as np
from flask import Flask, render_template, request, send_file, url_for

sys.path.insert(0, os.path.dirname(__file__))
import extract
import preprocess
import tiling
from engines import get_engine
from pipeline import PipelineConfig, process_tire

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO_DIR = os.path.join(BASE_DIR, "Lastik_fotolari")
UST_DIR = os.path.join(PHOTO_DIR, "Lastigin_ustu")
ALT_DIR = os.path.join(PHOTO_DIR, "Lastigin_alti")
OUT_DIR = os.path.join(BASE_DIR, "results", "webapp_output")
THUMB_DIR = os.path.join(OUT_DIR, "thumbs")
RUNS_DIR = os.path.join(OUT_DIR, "runs")
os.makedirs(THUMB_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# Arayuz icin varsayilan konfig: hizli olmasi icin tek motor + tek varyant
# (uretimdeki tam ensemble degil -- burasi gorsel kesif/inceleme araci).
WEBAPP_CONFIG = PipelineConfig(engines=["rapidocr"], variants=["v1_clahe"], tile_width=2400)

FONT = cv2.FONT_HERSHEY_SIMPLEX

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))

_ENGINE = None


def get_orientation_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = get_engine("rapidocr")
    return _ENGINE


# ---------------------------------------------------------------------------
# Fotograf listeleme + kucuk resimler
# ---------------------------------------------------------------------------


def _tire_id_from_filename(name: str) -> str:
    m = re.match(r"(?:Ust|Alt)(?:_(\d+))?\.jpg$", name)
    if not m:
        return None
    return m.group(1) if m.group(1) else "base"


def discover_tires():
    """tire_id -> {"ust": path|None, "alt": path|None} sozlugu, sayisal sirali."""
    tires: dict[str, dict] = {}
    for folder, side in [(UST_DIR, "ust"), (ALT_DIR, "alt")]:
        for path in glob.glob(os.path.join(folder, "*.jpg")):
            name = os.path.basename(path)
            tid = _tire_id_from_filename(name)
            if tid is None:
                continue
            tires.setdefault(tid, {"ust": None, "alt": None})
            tires[tid][side] = path
    return dict(sorted(tires.items(), key=lambda kv: (-1 if kv[0] == "base" else int(kv[0]))))


def _thumb_path(side: str, tire_id: str) -> str:
    return os.path.join(THUMB_DIR, f"{side}_{tire_id}.jpg")


def ensure_thumb(side: str, tire_id: str, src_path: str) -> str:
    out_path = _thumb_path(side, tire_id)
    if os.path.exists(out_path):
        return out_path
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return out_path
    h, w = img.shape
    new_w = 140
    new_h = max(1, int(h * new_w / w))
    thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_path, thumb)
    return out_path


# ---------------------------------------------------------------------------
# Adim adim gorsellestirme (bir yanak icin)
# ---------------------------------------------------------------------------


def _to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def process_side_with_steps(path: str, side_label: str, out_dir: str, prefix: str, engine) -> dict:
    """Bir yanak (ust/alt) icin adim adim gorselleri uretir + sure olcer.
    Donus: {"steps": [{"title","img","desc"}], "elapsed": float, "status": str}"""
    steps = []
    t0 = time.time()

    raw = preprocess.load_gray(path)
    if raw is None:
        return {"steps": [], "elapsed": time.time() - t0, "status": "dosya_yok"}

    h, w = raw.shape
    disp_h = min(h, 900)
    scale = 220 / w
    thumb = cv2.resize(raw[:disp_h], (220, int(disp_h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((thumb.shape[0] + 30, 260, 3), 24, np.uint8)
    canvas[25:25 + thumb.shape[0], 20:20 + 220] = _to_bgr(thumb)
    cv2.putText(canvas, f"{w}x{h}px (kesit)", (8, 18), FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    p0 = os.path.join(out_dir, f"{prefix}_0_raw.png")
    cv2.imwrite(p0, canvas)
    steps.append({"title": "0. Ham görüntü", "img": os.path.basename(p0),
                  "desc": f"Dosyadan okunan ham görüntü, {w}x{h}px."})

    engine_score = engine.score
    oriented = preprocess.resolve_orientation(raw, engine_score)
    if oriented.status != "ok":
        return {"steps": steps, "elapsed": time.time() - t0, "status": "okunamadi"}

    cand_a, cand_b = preprocess._landscape_candidates(raw)
    land = cand_a if oriented.rotation == "A" else cand_b
    band = oriented.band
    lh, lw = land.shape
    dscale = min(1.0, 1600 / lw)
    disp = cv2.resize(land, (int(lw * dscale), int(lh * dscale)), interpolation=cv2.INTER_AREA)
    disp_bgr = _to_bgr(disp)
    y0, y1 = int(band.top * dscale), int(band.bottom * dscale)
    overlay = disp_bgr.copy()
    cv2.rectangle(overlay, (0, y0), (disp_bgr.shape[1], y1), (60, 180, 255), -1)
    disp_bgr = cv2.addWeighted(overlay, 0.28, disp_bgr, 0.72, 0)
    cv2.rectangle(disp_bgr, (0, y0), (disp_bgr.shape[1], y1), (60, 180, 255), 2)
    p1 = os.path.join(out_dir, f"{prefix}_1_band.png")
    cv2.imwrite(p1, disp_bgr)
    steps.append({"title": "1. Orientasyon + bant tespiti", "img": os.path.basename(p1),
                  "desc": f"Aday '{oriented.rotation}' seçildi; turuncu kutu metin bandı."})

    cropped = oriented.image
    ch, cw = cropped.shape
    seg_w = min(2400, cw)
    x0 = max(0, cw // 2 - seg_w // 2)
    sample = preprocess.v1_clahe(cropped[:, x0:x0 + seg_w])
    p2 = os.path.join(out_dir, f"{prefix}_2_cropped.png")
    cv2.imwrite(p2, sample)
    steps.append({"title": "2. Bant kırpma + kontrast (CLAHE)", "img": os.path.basename(p2),
                  "desc": f"Bant boyutu {cropped.shape[1]}x{cropped.shape[0]}px, orta kesit gösteriliyor."})

    tiles = tiling.tiles_full_res(cropped, 2400, 200)
    show_w = min(cw, 3 * 2200 + 2400)
    band_disp = preprocess.v1_clahe(cropped[:, :show_w])
    tscale = min(1.0, 1600 / show_w)
    band_disp = cv2.resize(band_disp, (int(show_w * tscale), int(ch * tscale)), interpolation=cv2.INTER_AREA)
    band_bgr = _to_bgr(band_disp)
    colors = [(255, 140, 60), (60, 220, 255), (140, 255, 90), (255, 90, 200), (255, 210, 60)]
    for i, t in enumerate(tiles):
        if t.x0 >= show_w:
            break
        tx0 = int(t.x0 * tscale)
        cv2.rectangle(band_bgr, (tx0, 0), (tx0 + 2, band_bgr.shape[0]), colors[i % len(colors)], -1)
    p3 = os.path.join(out_dir, f"{prefix}_3_tiling.png")
    cv2.imwrite(p3, band_bgr)
    steps.append({"title": "3. Döşeme (tiling)", "img": os.path.basename(p3),
                  "desc": f"{len(tiles)} parçaya bölündü (renkli çizgiler sınırlar)."})

    best_boxes, best_tile_img = [], None
    for t in tiles[:6]:
        vimg = preprocess.v1_clahe(t.image)
        boxes = engine.read(vimg)
        if len(boxes) > len(best_boxes):
            best_boxes, best_tile_img = boxes, vimg
    if best_tile_img is not None:
        tile_bgr = _to_bgr(best_tile_img)
        for b in best_boxes:
            if not b.bbox:
                continue
            pts = np.array(b.bbox, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(tile_bgr, [pts], True, (60, 220, 60), 2)
            x, y = int(b.bbox[0][0]), int(b.bbox[0][1])
            label = f"{b.text} ({b.confidence:.2f})"
            (tw_, th_), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            cv2.rectangle(tile_bgr, (x, max(0, y - th_ - 6)), (x + tw_ + 4, y), (60, 220, 60), -1)
            cv2.putText(tile_bgr, label, (x + 2, max(10, y - 4)), FONT, 0.5, (10, 30, 10), 1, cv2.LINE_AA)
        th2, tw2 = tile_bgr.shape[:2]
        ds = min(1.0, 1600 / tw2)
        tile_bgr = cv2.resize(tile_bgr, (int(tw2 * ds), int(th2 * ds)), interpolation=cv2.INTER_AREA)
        p4 = os.path.join(out_dir, f"{prefix}_4_ocr.png")
        cv2.imwrite(p4, tile_bgr)
        steps.append({"title": "4. OCR kutuları (en dolu tile)", "img": os.path.basename(p4),
                      "desc": f"{len(best_boxes)} metin kutusu bulundu."})

    return {"steps": steps, "elapsed": time.time() - t0, "status": "ok"}


FIELD_LABELS = {
    "ebat": "Ebat", "hiz_grubu": "Hız Grubu", "marka": "Marka", "desen": "Desen",
    "mevsim": "Mevsim", "dot": "DOT", "ekstra_lot": "Ekstra Lot",
    "uretim_yeri": "Üretildiği Yer", "tup": "Tüplü/Tüpsüz",
}


def process_tire_full(tire_id: str, ust_path, alt_path, run_dir: str) -> dict:
    engine = get_orientation_engine()
    tire_dir = os.path.join(run_dir, tire_id)
    os.makedirs(tire_dir, exist_ok=True)

    t0 = time.time()
    side_results = {}
    if ust_path:
        side_results["ust"] = process_side_with_steps(ust_path, "Üst", tire_dir, "ust", engine)
    if alt_path:
        side_results["alt"] = process_side_with_steps(alt_path, "Alt", tire_dir, "alt", engine)

    result = process_tire(ust_path, alt_path, WEBAPP_CONFIG, engine)
    elapsed = time.time() - t0

    fields = []
    for key, label in FIELD_LABELS.items():
        fd = result["fields"].get(key, {})
        if key == "ekstra_lot":
            val = ", ".join(fd.get("values", [])[:4]) or None
            conf = None
        else:
            val = fd.get("value")
            conf = fd.get("confidence")
        fields.append({"field": label, "value": val, "conf": conf})

    return {
        "tire_id": tire_id,
        "sides": side_results,
        "fields": fields,
        "elapsed": round(elapsed, 1),
        "n_sources": result.get("n_sources", 0),
        "status": result.get("status", {}),
    }


# ---------------------------------------------------------------------------
# Rotalar
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    tires = discover_tires()
    rows = []
    for tid, sides in tires.items():
        ust_thumb = alt_thumb = None
        if sides["ust"]:
            ensure_thumb("ust", tid, sides["ust"])
            ust_thumb = url_for("thumb", side="ust", tire_id=tid)
        if sides["alt"]:
            ensure_thumb("alt", tid, sides["alt"])
            alt_thumb = url_for("thumb", side="alt", tire_id=tid)
        rows.append({"tire_id": tid, "ust_thumb": ust_thumb, "alt_thumb": alt_thumb})
    return render_template("index.html", rows=rows, config=WEBAPP_CONFIG)


@app.route("/thumb/<side>/<tire_id>")
def thumb(side, tire_id):
    path = _thumb_path(side, tire_id)
    if not os.path.exists(path):
        return "", 404
    return send_file(path, mimetype="image/jpeg")


@app.route("/step/<run_id>/<tire_id>/<filename>")
def step_image(run_id, tire_id, filename):
    path = os.path.join(RUNS_DIR, run_id, tire_id, filename)
    if not os.path.exists(path):
        return "", 404
    return send_file(path, mimetype="image/png")


@app.route("/process", methods=["POST"])
def process():
    selected = request.form.getlist("tire_id")
    if not selected:
        return render_template("results.html", results=[], run_id=None, total_elapsed=0)

    tires = discover_tires()
    run_id = uuid.uuid4().hex[:8]
    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    results = []
    t_total0 = time.time()
    for tid in selected:
        sides = tires.get(tid)
        if not sides:
            continue
        r = process_tire_full(tid, sides["ust"], sides["alt"], run_dir)
        for side_name, side_data in r["sides"].items():
            for step in side_data["steps"]:
                step["url"] = url_for("step_image", run_id=run_id, tire_id=tid, filename=step["img"])
        results.append(r)
    total_elapsed = round(time.time() - t_total0, 1)

    return render_template("results.html", results=results, run_id=run_id,
                            total_elapsed=total_elapsed, n=len(results))


if __name__ == "__main__":
    print("Web arayüzü: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
