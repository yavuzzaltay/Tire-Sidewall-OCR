"""Hızlı görsel doğrulama: orientasyon + bant tespiti + varyantlar.

Kullanım (repo kökünden):
    python tests/test_preprocess_manual.py

Çıktılar: results/debug/manual_check/ altına yazılır.
"""
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import preprocess
from engines import get_engine

OUT_DIR = REPO_ROOT / "results" / "debug" / "manual_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    ("ust5", REPO_ROOT / "Lastik_fotolari" / "Lastigin_ustu" / "Ust_5.jpg"),
    ("ust20", REPO_ROOT / "Lastik_fotolari" / "Lastigin_ustu" / "Ust_20.jpg"),
    ("alt50", REPO_ROOT / "Lastik_fotolari" / "Lastigin_alti" / "Alt_50.jpg"),
    ("ust56_empty", REPO_ROOT / "Lastik_fotolari" / "Lastigin_ustu" / "Ust_56.jpg"),  # bilinen boş çekim
]


def main():
    engine = get_engine("rapidocr")
    t0 = time.time()

    for tag, path in SAMPLES:
        raw = preprocess.load_gray(str(path))
        if raw is None:
            print(f"{tag}: DOSYA OKUNAMADI ({path})")
            continue
        print(f"{tag}: ham boyut {raw.shape}")
        oriented = preprocess.resolve_orientation(raw, engine.score)
        if oriented.status != "ok":
            print("  -> UNREADABLE (kalite kapısı doğru çalıştı)")
            continue
        img = oriented.image
        print(f"  -> yön={oriented.rotation}, bant kırpılmış boyut={img.shape}")
        cv2.imwrite(str(OUT_DIR / f"{tag}_cropped.png"), img)

        variants = preprocess.make_variants(img)
        for vname, vimg in variants.items():
            # önizleme için genişliği küçült
            h, w = vimg.shape
            scale = 2400 / w
            small = cv2.resize(vimg, (2400, int(h * scale)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUT_DIR / f"{tag}_{vname}.png"), small)

    print(f"\nToplam süre: {time.time() - t0:.1f}s")
    print(f"Çıktılar: {OUT_DIR}")


if __name__ == "__main__":
    main()
