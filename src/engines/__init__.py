"""OCR motor kayıt defteri. get_engine(name) motoru tembel (lazı) yükler."""
from __future__ import annotations

# ÖNEMLİ (GPU): torch (EasyOCR) ve onnxruntime-gpu (RapidOCR) her ikisi de kendi
# cuDNN 9.x DLL kopyasını yüklemeye çalışıyor; aynı süreçte hangisi ÖNCE cudnn
# DLL'lerini "kazanırsa" diğeri "WinError 127 / DLL bulunamadı" ile çöküyor
# (Windows aynı isimli DLL'i tekrar yüklemiyor, ilk yüklenen sürüm kalıcı oluyor).
# torch kendi bütün cuDNN'ini paketiyle taşıdığından, torch'u en başta -hangi
# motor önce istenirse istensin- import ederek bu DLL adı çakışmasını torch
# lehine sabitliyoruz; onnxruntime'ın CUDA sağlayıcısı sonradan aynı yüklü
# DLL'leri sorunsuz kullanabiliyor (test edildi).
try:
    import torch  # noqa: F401
except ImportError:
    pass

_CACHE: dict = {}


def get_engine(name: str):
    if name in _CACHE:
        return _CACHE[name]

    if name == "rapidocr":
        from .rapidocr_engine import RapidOcrEngine

        engine = RapidOcrEngine()
    elif name == "rapidocr_v6":
        from .rapidocr_v6_engine import RapidOcrV6Engine

        engine = RapidOcrV6Engine()
    elif name == "easyocr":
        from .easyocr_engine import EasyOcrEngine

        engine = EasyOcrEngine()
    elif name == "tesseract":
        from .tesseract_engine import TesseractEngine

        engine = TesseractEngine()
    else:
        raise ValueError(f"Bilinmeyen motor: {name}")

    _CACHE[name] = engine
    return engine


ENGINE_NAMES = ["rapidocr", "rapidocr_v6", "easyocr", "tesseract"]
