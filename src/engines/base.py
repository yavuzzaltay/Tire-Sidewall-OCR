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


_gpu_available_cache: Optional[bool] = None


def gpu_available() -> bool:
    """torch üzerinden CUDA'nın gerçekten kullanılabilir olup olmadığını kontrol eder.
    GPU'suz makinelerde (ör. paylaşılan Docker imajı) motorların otomatik olarak
    CPU'ya düşmesini sağlar — GPU'lu makinede elle bir şey değiştirmeye gerek kalmaz."""
    global _gpu_available_cache
    if _gpu_available_cache is None:
        try:
            import torch
            _gpu_available_cache = bool(torch.cuda.is_available())
        except Exception:
            _gpu_available_cache = False
    return _gpu_available_cache
