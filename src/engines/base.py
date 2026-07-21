"""Tüm OCR motorları için ortak sonuç tipi."""
from __future__ import annotations

import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class OcrBox:
    text: str
    confidence: float  # 0..1
    bbox: Optional[list] = None  # [[x,y], [x,y], [x,y], [x,y]]


def score_boxes(boxes: List[OcrBox]) -> float:
    """Basit metin-varlığı skoru: uzun + yüksek güvenli metinler daha çok katkı verir.
    Orientasyon seçimi ve hızlı karşılaştırmalar için kullanılır."""
    return sum(len(b.text.strip()) * b.confidence for b in boxes)
