"""Hızlı görsel doğrulama: orientasyon + bant tespiti + varyantlar."""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
import preprocess
from engines import get_engine

SCRATCH = Path(
    r"C:\Users\YAVUZA~1\AppData\Local\Temp\claude\C--Users-Yavuz-Altay-Desktop-LastikOCR"
    r"\244be65c-3030-4053-b800-6849a43e2716\scratchpad\pp_test"
)
SCRATCH.mkdir(parents=True, exist_ok=True)

samples = [
    ("ust5", r"Lastik_fotolari\Lastigin_ustu\Ust_5.jpg"),
    ("ust20", r"Lastik_fotolari\Lastigin_ustu\Ust_20.jpg"),
    ("alt50", r"Lastik_fotolari\Lastigin_alti\Alt_50.jpg"),
    ("ust56_empty", r"Lastik_fotolari\Lastigin_ustu\Ust_56.jpg"),  # bilinen boş çekim
]

engine = get_engine("rapidocr")
t0 = time.time()

for tag, relpath in samples:
    raw = preprocess.load_gray(relpath)
    if raw is None:
        print(f"{tag}: DOSYA OKUNAMADI")
        continue
    print(f"{tag}: ham boyut {raw.shape}")
    oriented = preprocess.resolve_orientation(raw, engine.score)
    if oriented.status == "unreadable":
        print(f"  -> UNREADABLE (kalite kapısı)")
        continue
    img = oriented.image
    print(f"  -> yön={oriented.rotation}, bant kırpılmış boyut={img.shape}")
    cv2.imwrite(str(SCRATCH / f"{tag}_cropped.png"), img)

    variants = preprocess.make_variants(img)
    for vname, vimg in variants.items():
        # önizleme için genişliği küçült
        h, w = vimg.shape
        scale = 2400 / w
        small = cv2.resize(vimg, (2400, int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(SCRATCH / f"{tag}_{vname}.png"), small)

print(f"\nToplam süre: {time.time()-t0:.1f}s")
print(f"Çıktılar: {SCRATCH}")
