"""Rapor için pipeline'ın adım adım görselleştirmesini üretir: ham görüntüden
nihai alan çıkarımına kadar her aşamanın gerçek bir lastik üzerindeki hali.

Çıktı: results/debug/pipeline/*.png + results/debug/pipeline/walkthrough.json
(HTML rapor bu JSON'daki metin/tablo verilerini kullanır).
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import extract
import fuse
import preprocess
import tiling
from engines import get_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results", "debug", "pipeline")
os.makedirs(OUT_DIR, exist_ok=True)

TIRE_PATH = os.path.join(BASE_DIR, "Lastik_fotolari", "Lastigin_ustu", "Ust_5.jpg")

FONT = cv2.FONT_HERSHEY_SIMPLEX


def save(name, img):
    cv2.imwrite(os.path.join(OUT_DIR, name), img)
    print("saved", name, img.shape)


def to_bgr(gray):
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def main():
    engine = get_engine("rapidocr")
    walkthrough = {}

    # ---------- ADIM 0: ham görüntü (tamamı çok uzun oldugu icin temsili bir kesit) ----------
    raw = preprocess.load_gray(TIRE_PATH)
    walkthrough["raw_shape"] = list(raw.shape)
    h, w = raw.shape
    disp_h = 900
    raw_crop = raw[:disp_h]
    scale = 220 / w
    thumb = cv2.resize(raw_crop, (220, int(disp_h * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.full((thumb.shape[0] + 40, 360, 3), 24, np.uint8)
    canvas[30:30 + thumb.shape[0], 60:60 + 220] = to_bgr(thumb)
    cv2.putText(canvas, f"tam boyut: {w} x {h} px (kesit gosteriliyor)", (10, 20), FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    save("step0_raw.png", canvas)

    # ---------- ADIM 1: orientasyon çözümü ----------
    cand_a, cand_b = preprocess._landscape_candidates(raw)
    band_a = preprocess.detect_band(cand_a)
    band_b = preprocess.detect_band(cand_b) or band_a
    crop_a = cand_a[band_a.top:band_a.bottom]
    crop_b = cand_b[band_b.top:band_b.bottom]
    score_a = engine.score(preprocess._sample_for_scoring(crop_a))
    score_b = engine.score(preprocess._sample_for_scoring(crop_b))
    winner = "A" if score_a >= score_b else "B"
    walkthrough["orientation"] = {"score_a": round(score_a, 1), "score_b": round(score_b, 1), "winner": winner}

    land = cand_a if winner == "A" else cand_b
    band = band_a if winner == "A" else band_b
    lh, lw = land.shape
    dscale = 1900 / lw
    disp = cv2.resize(land, (1900, max(1, int(lh * dscale))), interpolation=cv2.INTER_AREA)
    disp_bgr = to_bgr(disp)
    y0, y1 = int(band.top * dscale), int(band.bottom * dscale)
    overlay = disp_bgr.copy()
    cv2.rectangle(overlay, (0, y0), (1900, y1), (60, 180, 255), -1)
    disp_bgr = cv2.addWeighted(overlay, 0.28, disp_bgr, 0.72, 0)
    cv2.rectangle(disp_bgr, (0, y0), (1900, y1), (60, 180, 255), 2)
    cv2.putText(disp_bgr, f"tespit edilen metin bandi (y={band.top}-{band.bottom})",
                (10, max(20, y0 - 10)), FONT, 0.6, (60, 180, 255), 2, cv2.LINE_AA)
    save("step1_oriented_band.png", disp_bgr)

    # ---------- ADIM 2: bant kırpma ----------
    cropped = land[band.top:band.bottom]
    ch, cw = cropped.shape
    seg_w = min(2600, cw)
    x0 = max(0, cw // 2 - seg_w // 2)
    sample_crop = cropped[:, x0:x0 + seg_w]
    save("step2_cropped_band.png", preprocess.v1_clahe(sample_crop))
    walkthrough["band_shape"] = list(cropped.shape)

    # ---------- ADIM 3: 5 varyant (aynı örnek kesit) ----------
    variants = preprocess.make_variants(sample_crop)
    rows = []
    for vname, vimg in variants.items():
        labeled = to_bgr(vimg.copy())
        cv2.putText(labeled, vname, (8, 26), FONT, 0.75, (60, 180, 255), 2, cv2.LINE_AA)
        rows.append(labeled)
        rows.append(np.full((8, seg_w, 3), 24, np.uint8))
    save("step3_variants.png", np.vstack(rows))

    # ---------- ADIM 4: döşeme (tiling) görselleştirmesi ----------
    tile_w, overlap = 2400, 200
    tiles = tiling.tiles_full_res(cropped, tile_w, overlap)
    walkthrough["n_tiles"] = len(tiles)
    walkthrough["tile_width"] = tile_w
    walkthrough["tile_overlap"] = overlap

    show_w = min(cw, 4 * (tile_w - overlap) + tile_w)
    band_disp = preprocess.v1_clahe(cropped[:, :show_w])
    tscale = 1900 / show_w
    band_disp = cv2.resize(band_disp, (1900, max(1, int(ch * tscale))), interpolation=cv2.INTER_AREA)
    band_bgr = to_bgr(band_disp)

    ROW_H = 34
    n_rows = 2
    ruler = np.full((n_rows * ROW_H + 10, 1900, 3), 24, np.uint8)
    colors = [(255, 140, 60), (60, 220, 255), (140, 255, 90), (255, 90, 200), (255, 210, 60), (170, 140, 255)]
    shown_tiles = [t for t in tiles if t.x0 < show_w]
    for i, t in enumerate(shown_tiles):
        tx0, tx1 = int(t.x0 * tscale), min(1900, int(t.x1 * tscale))
        color = colors[i % len(colors)]
        row = i % n_rows
        y0, y1 = 5 + row * ROW_H, 5 + row * ROW_H + ROW_H - 6
        cv2.rectangle(ruler, (tx0, y0), (tx1, y1), color, -1)
        cv2.rectangle(band_bgr, (tx0, 0), (tx0 + 2, band_bgr.shape[0]), color, -1)
        label = f"tile {i}: {t.x0}-{t.x1}px"
        (tw_, th_), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        lx = min(tx0 + 8, 1900 - tw_ - 8)
        cv2.putText(ruler, label, (lx, y0 + ROW_H - 12), FONT, 0.5, (10, 10, 10), 1, cv2.LINE_AA)
    # bindirme (overlap) bölgelerini isaretle
    for i in range(len(shown_tiles) - 1):
        ox0 = int(shown_tiles[i + 1].x0 * tscale)
        ox1 = int(shown_tiles[i].x1 * tscale)
        if ox1 > ox0:
            cv2.rectangle(band_bgr, (ox0, 0), (ox1, band_bgr.shape[0]), (255, 255, 255), 1)
    combo = np.vstack([band_bgr, np.full((6, 1900, 3), 24, np.uint8), ruler])
    save("step4_tiling.png", combo)

    # ---------- ADIM 5: OCR kutuları (bir tile üzerinde) ----------
    best_tile_idx, best_boxes = 0, []
    for i, t in enumerate(tiles[:6]):
        vimg = preprocess.v1_clahe(t.image)
        boxes = engine.read(vimg)
        if len(boxes) > len(best_boxes):
            best_tile_idx, best_boxes = i, boxes
    tile_img = preprocess.v1_clahe(tiles[best_tile_idx].image)
    tile_bgr = to_bgr(tile_img)
    for b in best_boxes:
        if not b.bbox:
            continue
        pts = np.array(b.bbox, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(tile_bgr, [pts], True, (60, 220, 60), 2)
        x, y = int(b.bbox[0][0]), int(b.bbox[0][1])
        label = f"{b.text} ({b.confidence:.2f})"
        (tw_, th_), _ = cv2.getTextSize(label, FONT, 0.55, 1)
        cv2.rectangle(tile_bgr, (x, max(0, y - th_ - 8)), (x + tw_ + 6, y), (60, 220, 60), -1)
        cv2.putText(tile_bgr, label, (x + 3, max(12, y - 5)), FONT, 0.55, (10, 30, 10), 1, cv2.LINE_AA)
    th_, tw_ = tile_bgr.shape[:2]
    dscale2 = min(1.0, 1900 / tw_)
    tile_bgr = cv2.resize(tile_bgr, (int(tw_ * dscale2), int(th_ * dscale2)), interpolation=cv2.INTER_AREA)
    save("step5_ocr_boxes.png", tile_bgr)
    walkthrough["ocr_tile_index"] = best_tile_idx
    walkthrough["ocr_box_count"] = len(best_boxes)
    walkthrough["ocr_sample_texts"] = [{"text": b.text, "conf": round(b.confidence, 2)} for b in best_boxes]

    # ---------- ADIM 6+7: tüm tile/varyant üzerinde çıkarım + birleştirme ----------
    per_source = []
    for t in tiles:
        for vname, vfn in preprocess.VARIANTS.items():
            vimg = vfn(t.image)
            boxes = engine.read(vimg)
            fields = extract.extract_from_boxes(boxes)
            per_source.append((f"ust|{t.x0}|{vname}|rapidocr", fields))
    fused = fuse.fuse_all(per_source)

    field_labels = {
        "ebat": "Ebat", "hiz_grubu": "Hız Grubu", "marka": "Marka", "desen": "Desen",
        "mevsim": "Mevsim", "dot": "DOT", "ekstra_lot": "Ekstra Lot",
        "uretim_yeri": "Üretildiği Yer", "tup": "Tüplü/Tüpsüz",
    }
    fused_summary = []
    for key, label in field_labels.items():
        fd = fused.get(key, {})
        if key == "ekstra_lot":
            val = ", ".join(fd.get("values", [])[:3])
            conf = None
        else:
            val = fd.get("value")
            conf = fd.get("confidence")
        fused_summary.append({"field": label, "value": val, "conf": conf})
    walkthrough["fused_summary"] = fused_summary
    walkthrough["n_sources"] = len(per_source)

    with open(os.path.join(OUT_DIR, "walkthrough.json"), "w", encoding="utf-8") as f:
        json.dump(walkthrough, f, ensure_ascii=False, indent=2)
    print(json.dumps(walkthrough, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
