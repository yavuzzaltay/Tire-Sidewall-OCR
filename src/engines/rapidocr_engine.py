from __future__ import annotations

import os
import site
from typing import List

import numpy as np

from .base import OcrBox, score_boxes


_CUDA_DLL_DIRS_REGISTERED = False


def _register_cuda_dll_dirs() -> None:
    """onnxruntime-gpu'nun native provider DLL'i, kendi bağımlılıklarını (cublasLt64_13.dll
    vb.) klasik Windows DLL arama sırasıyla (PATH dahil) arıyor; os.add_dll_directory()
    bu native yüklemeyi kapsamıyor (denendi, işe yaramadı). Bu yüzden pip ile kurulan
    nvidia-cublas/nvidia-cudnn paketlerinin klasörlerini doğrudan PATH'e ekliyoruz."""
    global _CUDA_DLL_DIRS_REGISTERED
    if _CUDA_DLL_DIRS_REGISTERED:
        return
    dirs = []
    for sp in site.getsitepackages():
        for sub in [("nvidia", "cu13", "bin", "x86_64"), ("nvidia", "cudnn", "bin")]:
            d = os.path.join(sp, *sub)
            if os.path.isdir(d):
                dirs.append(d)
    if dirs:
        # ÖNEMLİ: sona ekliyoruz (öne değil). Öne eklemek torch'un kendi
        # paketlediği cuDNN DLL'leriyle çakışıp EasyOCR'ı bozuyordu (WinError 127) —
        # torch kendi torch\\lib içindeki DLL'leri önce bulmalı, bizim nvidia-*
        # paketlerimiz sadece onnxruntime'ın bulamadığı durumda yedek olmalı.
        # add_dll_directory() de denendi ama onnxruntime'ın native provider
        # yüklemesini etkilemiyor; yalnızca PATH işe yarıyor.
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + os.pathsep.join(dirs)
    _CUDA_DLL_DIRS_REGISTERED = True


class RapidOcrEngine:
    name = "rapidocr"

    def __init__(self, use_cuda: bool = True):
        if use_cuda:
            _register_cuda_dll_dirs()
        from rapidocr_onnxruntime import RapidOCR

        # NOT: rapidocr_onnxruntime 1.2.3'te cls_use_cuda/rec_use_cuda kwargs'ları
        # kütüphanenin kendi parametre işleme kodundaki bir kusur yüzünden doğru
        # anahtara yazılmıyor (yalnızca det_use_cuda doğru çalışıyor). Bu yüzden
        # GPU/CPU seçimini paketin config.yaml dosyasındaki Det/Cls/Rec.use_cuda
        # varsayılanlarını değiştirerek yapıyoruz; buradaki kwarg sadece Det için
        # ek bir garanti katmanı.
        if use_cuda:
            self._engine = RapidOCR(det_use_cuda=True, det_model_path="")
        else:
            self._engine = RapidOCR(det_use_cuda=False, det_model_path="")

    def read(self, img: np.ndarray) -> List[OcrBox]:
        img_in = np.stack([img] * 3, axis=-1) if img.ndim == 2 else img
        result, _ = self._engine(img_in)
        boxes: List[OcrBox] = []
        if result:
            for bbox, text, conf in result:
                boxes.append(OcrBox(text=text, confidence=float(conf), bbox=bbox))
        return boxes

    def score(self, img: np.ndarray) -> float:
        return score_boxes(self.read(img))
