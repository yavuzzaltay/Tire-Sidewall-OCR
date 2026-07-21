"""Lastik yanak görüntüleri için ön işleme: orientasyon düzeltme, kalite kapısı,
metin bandı kırpma ve OCR için zenginleştirme varyantları.

Line-scan kamera görüntüleri çok uzun/dar şeritler (~1116 x 12900 px) olarak gelir.
Bu modül ham şeridi -> doğru yönde, sadece metin bandını içeren, kontrastı
artırılmış birden çok varyanta çevirir.
"""
from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Yükleme ve orientasyon
# ---------------------------------------------------------------------------


def load_gray(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def _landscape_candidates(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ham görüntüyü yatay (metin satırları yatay okunacak şekilde) hale getirecek
    180 derece farklı iki aday döndürür. Hangisinin doğru yön olduğu ayrıca
    çözülmelidir (resolve_orientation)."""
    h, w = raw.shape
    if h > w:
        # dikey şerit -> 90 derece döndürerek yatay yap
        a = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        b = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
    else:
        # zaten yatay kaydedilmiş (bazı dosyalar bu şekilde)
        a = raw.copy()
        b = cv2.rotate(raw, cv2.ROTATE_180)
    return a, b


# ---------------------------------------------------------------------------
# Kalite kapısı + metin bandı tespiti
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BandResult:
    top: int
    bottom: int
    row_profile: np.ndarray


def detect_band(gray_landscape: np.ndarray, min_frac: float = 0.04) -> Optional[BandResult]:
    """Yatay görüntüde parlaklığın yüksek olduğu tek/en büyük satır aralığını bulur.
    Lastik sırt/etiket bandı budur; geri kalanı arka plan (siyah) olur.
    Bant bulunamazsa (boş/başarısız çekim) None döner."""
    h, w = gray_landscape.shape
    profile = np.median(gray_landscape.astype(np.float32), axis=1)

    lo, hi = np.percentile(profile, [5, 95])
    if hi - lo < 8.0:
        return None  # neredeyse düz -> içerik yok

    thresh = lo + 0.35 * (hi - lo)
    mask = profile > thresh

    # en uzun ardışık True bloğunu bul
    best_start, best_len = -1, 0
    cur_start, cur_len = -1, 0
    for i, v in enumerate(mask):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    if best_len < max(8, int(min_frac * h)):
        return None

    pad = max(4, int(0.12 * best_len))
    top = max(0, best_start - pad)
    bottom = min(h, best_start + best_len + pad)
    return BandResult(top=top, bottom=bottom, row_profile=profile)


# ---------------------------------------------------------------------------
# Orientasyon çözümü (OCR güven skoruyla)
# ---------------------------------------------------------------------------

ScoreFn = Callable[[np.ndarray], float]


@dataclasses.dataclass
class OrientedImage:
    status: str  # "ok" | "unreadable"
    image: Optional[np.ndarray] = None  # bant kırpılmış, doğru yönde gri görüntü
    band: Optional[BandResult] = None
    rotation: Optional[str] = None  # hangi aday seçildi: "A" | "B"


def _sample_for_scoring(cropped: np.ndarray, width: int = 2200) -> np.ndarray:
    h, w = cropped.shape
    x0 = max(0, w // 2 - width // 2)
    x1 = min(w, x0 + width)
    sample = cropped[:, x0:x1]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(sample)


def resolve_orientation(raw: np.ndarray, score_fn: ScoreFn) -> OrientedImage:
    cand_a, cand_b = _landscape_candidates(raw)

    band_a = detect_band(cand_a)
    if band_a is None:
        # b, a'nın 180 derece çevrilmişi -> aynı sebepten bantsız olacaktır
        return OrientedImage(status="unreadable")

    band_b = detect_band(cand_b)
    if band_b is None:
        band_b = band_a  # olmamalı ama güvenlik için

    crop_a = cand_a[band_a.top : band_a.bottom]
    crop_b = cand_b[band_b.top : band_b.bottom]

    score_a = score_fn(_sample_for_scoring(crop_a))
    score_b = score_fn(_sample_for_scoring(crop_b))

    if score_a >= score_b:
        return OrientedImage(status="ok", image=crop_a, band=band_a, rotation="A")
    return OrientedImage(status="ok", image=crop_b, band=band_b, rotation="B")


# ---------------------------------------------------------------------------
# Zenginleştirme varyantları
# ---------------------------------------------------------------------------


def v1_clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def v2_shading_clahe(img: np.ndarray) -> np.ndarray:
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=31)
    bg = np.clip(bg.astype(np.float32), 1, 255)
    norm = (img.astype(np.float32) / bg) * 128.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(norm)


def v3_gradient(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mmax = float(mag.max()) if mag.max() > 0 else 1.0
    mag = np.clip(mag / mmax * 255.0, 0, 255).astype(np.uint8)
    return 255 - mag  # açık zemin üstüne koyu kenarlar


def v4_morph_contrast(img: np.ndarray) -> np.ndarray:
    ksize = 25
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.add(cv2.subtract(img, blackhat), tophat)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(enhanced)


def v5_negative(img: np.ndarray) -> np.ndarray:
    return 255 - v2_shading_clahe(img)


VARIANTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "v1_clahe": v1_clahe,
    "v2_shading": v2_shading_clahe,
    "v3_gradient": v3_gradient,
    "v4_morph": v4_morph_contrast,
    "v5_negative": v5_negative,
}


def make_variants(img: np.ndarray) -> dict[str, np.ndarray]:
    return {name: fn(img) for name, fn in VARIANTS.items()}
