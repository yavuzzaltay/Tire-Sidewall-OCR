"""results/debug/ altına birkaç örnek lastik için bant kırpma + 5 varyant görselini kaydeder."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import preprocess
from engines import get_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO_DIR = os.path.join(BASE_DIR, "Lastik_fotolari")
DEBUG_DIR = os.path.join(BASE_DIR, "results", "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

SAMPLES = [
    ("5_ust", os.path.join(PHOTO_DIR, "Lastigin_ustu", "Ust_5.jpg")),
    ("30_ust", os.path.join(PHOTO_DIR, "Lastigin_ustu", "Ust_30.jpg")),
    ("82_alt", os.path.join(PHOTO_DIR, "Lastigin_alti", "Alt_82.jpg")),
    ("56_ust_bos", os.path.join(PHOTO_DIR, "Lastigin_ustu", "Ust_56.jpg")),
]


def main():
    engine = get_engine("rapidocr")
    for tag, path in SAMPLES:
        raw = preprocess.load_gray(path)
        if raw is None:
            print(f"{tag}: dosya okunamadi")
            continue
        oriented = preprocess.resolve_orientation(raw, engine.score)
        if oriented.status != "ok":
            print(f"{tag}: UNREADABLE (kalite kapisi dogru calisti)")
            continue
        img = oriented.image
        h, w = img.shape
        seg_w = min(2600, w)
        x0 = max(0, w // 2 - seg_w // 2)
        crop = img[:, x0:x0 + seg_w]

        variants = preprocess.make_variants(crop)
        rows = []
        for vname, vimg in variants.items():
            labeled = vimg.copy()
            cv2.putText(labeled, vname, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255, 2)
            rows.append(labeled)
            rows.append(np.full((6, seg_w), 128, np.uint8))
        combo = np.vstack(rows)
        outp = os.path.join(DEBUG_DIR, f"{tag}_variants.png")
        cv2.imwrite(outp, combo)
        print(f"{tag}: kaydedildi -> {outp}  (bant boyutu={img.shape}, yon={oriented.rotation})")


if __name__ == "__main__":
    main()
