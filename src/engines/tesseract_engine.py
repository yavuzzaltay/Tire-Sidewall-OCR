from __future__ import annotations

import shutil
from typing import List

import numpy as np

from .base import OcrBox, score_boxes

_DEFAULT_WIN_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class TesseractEngine:
    name = "tesseract"

    def __init__(self, config: str = "--psm 11"):
        import pytesseract

        if shutil.which("tesseract") is None:
            pytesseract.pytesseract.tesseract_cmd = _DEFAULT_WIN_PATH
        self._pytesseract = pytesseract
        self.config = config

    def read(self, img: np.ndarray) -> List[OcrBox]:
        from pytesseract import Output

        data = self._pytesseract.image_to_data(img, config=self.config, output_type=Output.DICT)
        boxes: List[OcrBox] = []
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])
            if text and conf > 0:
                x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                boxes.append(OcrBox(text=text, confidence=conf / 100.0, bbox=bbox))
        return boxes

    def score(self, img: np.ndarray) -> float:
        return score_boxes(self.read(img))
