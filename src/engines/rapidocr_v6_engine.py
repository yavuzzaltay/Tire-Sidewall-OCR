"""Yeni nesil `rapidocr` paketi (3.9.1, PP-OCRv6 det/rec + PP-OCRv4 cls) sarmalayıcısı.
Eski `rapidocr_onnxruntime` (1.2.3, PP-OCRv3) tabanlı RapidOcrEngine ile yan yana
durur — Faz 2 karşılaştırması için ayrı bir motor adıyla (rapidocr_v6) kayıtlı."""
from __future__ import annotations

from typing import List

import numpy as np

from .base import OcrBox, gpu_available, score_boxes
from .rapidocr_engine import _register_cuda_dll_dirs


class RapidOcrV6Engine:
    name = "rapidocr_v6"

    def __init__(self, use_cuda: bool | None = None):
        if use_cuda is None:
            use_cuda = gpu_available()  # GPU yoksa (ör. CPU-only Docker) otomatik CPU'ya duser
        if use_cuda:
            _register_cuda_dll_dirs()
        from rapidocr import RapidOCR

        params = {"EngineConfig.onnxruntime.use_cuda": use_cuda} if use_cuda else {}
        self._engine = RapidOCR(params=params or None)

    def read(self, img: np.ndarray) -> List[OcrBox]:
        img_in = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img
        out = self._engine(img_in)
        boxes: List[OcrBox] = []
        if out is not None and out.boxes is not None:
            for bbox, text, score in zip(out.boxes, out.txts, out.scores):
                boxes.append(OcrBox(text=text, confidence=float(score), bbox=bbox.tolist()))
        return boxes

    def score(self, img: np.ndarray) -> float:
        return score_boxes(self.read(img))
