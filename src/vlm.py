"""Yerel VLM (Qwen2.5-VL, Ollama üzerinden) ile lastik yanak okuma denemesi.

Klasik pipeline'dan farklı olarak OCR kutuları üretmez; doğrudan yapılandırılmış
alan çıktısı (JSON) üretmesi istenir. Sonuç, halüsinasyonlara karşı lexicon
sözlüğüyle doğrulanır (bilinmeyen marka/desen adları elenir).
"""
from __future__ import annotations

import base64
import json
import re
from typing import Optional

import cv2
import numpy as np
import requests
from rapidfuzz import fuzz, process, utils as fuzz_utils

import lexicon

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5vl:3b"

PROMPT = """Bu bir araba lastiğinin yanak (sidewall) fotoğrafının bir kesiti. Kabartma
(siyah üstüne siyah, düşük kontrastlı) yazıları oku. Şu alanları JSON olarak döndür:

{
  "marka": "lastik markası (ör. Continental, Goodyear, Lassa) veya emin değilsen null",
  "desen": "lastik deseni/model adı (ör. EcoContact 6) veya emin değilsen null",
  "ebat": "lastik ebadı, ör. 205/55 R16 formatında, veya emin değilsen null",
  "hiz_grubu": "yük endeksi + hız harfi, ör. 91H, veya emin değilsen null",
  "dot": "DOT damgasındaki son 4 haneli üretim tarihi kodu (hafta+yıl, ör. 1526) veya emin değilsen null",
  "uretim_yeri": "MADE IN ... yazısındaki ülke adı veya emin değilsen null",
  "tup": "TUBELESS goruyorsan 'Tupsuz', TUBE TYPE goruyorsan 'Tuplu', emin degilsen null",
  "mevsim": "M+S veya kar tanesi ikonu goruyorsan '4 Mevsim' veya 'Kis', yoksa null"
}

Sadece JSON döndür, başka açıklama ekleme. Emin olmadığın alanlara kesinlikle null yaz,
tahmin uydurma."""


MAX_WIDTH = 1400  # daha genis goruntulerde model yaniti bos/yavas donuyordu (test edildi)


def _encode_image(img: np.ndarray) -> str:
    # Ollama/Qwen2.5-VL gri tonlamali (tek kanal) PNG'de bos yanit donuyor; RGB sart.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        scale = MAX_WIDTH / w
        img = cv2.resize(img, (MAX_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError("görüntü encode edilemedi")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _call_ollama(img: np.ndarray, timeout: float = 60.0) -> str:
    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [_encode_image(img)],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "")


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def _clean_null(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("null", "none", "bilinmiyor", "emin değilim", "n/a"):
        return None
    return s


_TUP_MAP = {"tupsuz": "Tüpsüz", "tüpsüz": "Tüpsüz", "tuplu": "Tüplü", "tüplü": "Tüplü",
            "tubeless": "Tüpsüz", "tube type": "Tüplü"}
_MEVSIM_MAP = {"4 mevsim": "4 Mevsim", "kis": "Kış", "kış": "Kış", "yaz": "Yaz",
               "all season": "4 Mevsim", "winter": "Kış", "summer": "Yaz"}


def _normalize_tup(v: Optional[str]) -> Optional[str]:
    v = _clean_null(v)
    return _TUP_MAP.get(v.lower(), v) if v else None


def _normalize_mevsim(v: Optional[str]) -> Optional[str]:
    v = _clean_null(v)
    return _MEVSIM_MAP.get(v.lower(), v) if v else None


def _validate_brand(value: Optional[str]) -> Optional[str]:
    value = _clean_null(value)
    if not value:
        return None
    match = process.extractOne(value, lexicon.BRANDS, scorer=fuzz.WRatio, score_cutoff=75,
                                processor=fuzz_utils.default_process)
    return match[0] if match else None


def _validate_pattern(value: Optional[str], brand: Optional[str]) -> Optional[str]:
    value = _clean_null(value)
    if not value:
        return None
    pool = lexicon.BRAND_PATTERNS.get(brand) if brand else [p for p, _ in lexicon.ALL_PATTERNS]
    if not pool:
        return None
    match = process.extractOne(value, pool, scorer=fuzz.WRatio, score_cutoff=70,
                                processor=fuzz_utils.default_process)
    return match[0] if match else None


def read_tire_fields(img: np.ndarray) -> dict:
    """Tek bir görüntü kesitinden (bant veya geniş tile) yapılandırılmış alanları okur.
    Dönüş: extract.py alan adlarıyla uyumlu dict (marka, desen, ebat, hiz_grubu, dot,
    uretim_yeri, tup, mevsim) — ham VLM çıktısı + lexicon ile doğrulanmış hali."""
    raw_text = _call_ollama(img)
    raw = _extract_json(raw_text)

    raw_brand = _clean_null(raw.get("marka"))
    raw_pattern = _clean_null(raw.get("desen"))
    brand = _validate_brand(raw_brand)
    pattern = _validate_pattern(raw_pattern, brand)

    return {
        "marka": brand,
        "desen": pattern,
        "ebat": _clean_null(raw.get("ebat")),
        "hiz_grubu": _clean_null(raw.get("hiz_grubu")),
        "dot": _clean_null(raw.get("dot")),
        "uretim_yeri": _clean_null(raw.get("uretim_yeri")),
        "tup": _normalize_tup(raw.get("tup")),
        "mevsim": _normalize_mevsim(raw.get("mevsim")),
        "_raw": raw,
        "_raw_text": raw_text,
    }
