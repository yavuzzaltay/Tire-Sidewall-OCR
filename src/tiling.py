"""Bant kırpılmış şeridi OCR'a uygun parçalara böler.

İki tamamlayıcı görünüm üretilir:
- küçültülmüş tam şerit: büyük yazılar (marka, desen, ebat) için, tek parça,
  motorun tüm satırı bağlamıyla görmesini sağlar.
- tam çözünürlük döşeme: küçük yazılar (DOT, lot kodları, üretim yeri) için,
  bindirmeli dikdörtgenler; küçültme sırasında kaybolacak detayları korur.
"""
from __future__ import annotations

import dataclasses
from typing import List

import cv2
import numpy as np


def full_strip_small(img: np.ndarray, target_width: int = 3000) -> np.ndarray:
    h, w = img.shape
    if w <= target_width:
        return img.copy()
    scale = target_width / w
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_AREA)


@dataclasses.dataclass
class Tile:
    x0: int
    x1: int
    image: np.ndarray


def tiles_full_res(img: np.ndarray, tile_width: int = 1600, overlap: int = 200) -> List[Tile]:
    h, w = img.shape
    if w <= tile_width:
        return [Tile(0, w, img.copy())]

    step = max(1, tile_width - overlap)
    tiles: List[Tile] = []
    x = 0
    while True:
        x1 = min(x + tile_width, w)
        x0 = max(0, x1 - tile_width) if x1 == w else x
        tiles.append(Tile(x0, x1, img[:, x0:x1].copy()))
        if x1 >= w:
            break
        x += step
    return tiles


def offset_bbox(bbox, x_offset: int):
    """Tile-yerel bbox koordinatlarını tam şerit koordinatlarına taşır."""
    if bbox is None:
        return None
    return [[float(px) + x_offset, float(py)] for px, py in bbox]
