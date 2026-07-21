from __future__ import annotations

from typing import List

import numpy as np

from .base import OcrBox, gpu_available, score_boxes


class EasyOcrEngine:
    name = "easyocr"

    def __init__(self, gpu: bool | None = None):
        import easyocr

        if gpu is None:
            gpu = gpu_available()  # GPU yoksa (ör. CPU-only Docker) otomatik CPU'ya duser
        self._reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, img: np.ndarray) -> List[OcrBox]:
        result = self._reader.readtext(img)
        boxes: List[OcrBox] = []
        for bbox, text, conf in result:
            boxes.append(OcrBox(text=text, confidence=float(conf), bbox=bbox))
        return boxes

    def score(self, img: np.ndarray) -> float:
        return score_boxes(self.read(img))
