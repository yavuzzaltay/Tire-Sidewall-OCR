"""Fused sonuçları ground truth ile karşılaştırma yardımcıları."""
from __future__ import annotations

import re
from typing import Optional

PRIORITY_FIELDS = ["marka", "desen", "ebat", "dot"]
ALL_SCORED_FIELDS = ["marka", "desen", "ebat", "hiz_grubu", "mevsim", "dot", "uretim_yeri", "tup"]


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_/]", "", s).upper()


def field_match(field: str, predicted: Optional[str], truth: Optional[str]) -> Optional[bool]:
    """None dönerse bu alan bu lastik için değerlendirme dışıdır (GT bilinmiyor: '?' veya boş)."""
    if truth is None or truth == "" or truth == "?":
        return None
    if predicted is None or predicted == "":
        return False

    p, t = _norm(predicted), _norm(truth)

    if field == "ebat":
        return p == t
    if field == "hiz_grubu":
        return p == t or p in t or t in p
    if field == "dot":
        return p == t
    if field in ("marka", "desen", "uretim_yeri", "mevsim", "tup"):
        return p == t or p in t or t in p
    return p == t


def score_tire(fused_fields: dict, gt_row: dict) -> dict:
    """Her alan için True/False/None döner."""
    out = {}
    for field in ALL_SCORED_FIELDS:
        predicted = fused_fields.get(field, {}).get("value")
        truth = gt_row.get(field)
        out[field] = field_match(field, predicted, truth)
    return out
