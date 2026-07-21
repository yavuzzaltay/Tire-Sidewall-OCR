"""Birden çok kaynaktan (yanak x tile x varyant x motor) gelen alan adaylarını
oylayarak tek bir nihai değere indirger.

source_id kuralı: "{side}|{tile_x0}|{variant}|{engine}" (izlenebilirlik için).
"""
from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import lexicon
from extract import FIELDS, Candidate, FieldCandidates, FOURDIGIT_RE

MULTI_VALUE_FIELDS = {"ekstra_lot"}

# nihai markanın sözlüğünde olmayan desen adayları bu çarpanla cezalandırılır
# (tamamen elenmiyor: marka oylaması yanlış çıkmış olabilir, desen güçlüyse hâlâ kazanabilir)
BRAND_MISMATCH_PENALTY = 0.15


@dataclasses.dataclass
class ScoredValue:
    value: str
    score: float
    n_sources: int
    n_engines: int


def _source_engine(source_id: str) -> str:
    parts = source_id.split("|")
    return parts[-1] if parts else source_id


def _dot_aux_candidates(value: str) -> List[str]:
    """DOT alanındaki karışık ham kodlardan olası 4 haneli tarih alt dizilerini
    ikincil (daha düşük ağırlıklı) aday olarak çıkarır."""
    out = []
    for m in FOURDIGIT_RE.finditer(value):
        token = m.group(1)
        wk = int(token[:2])
        if 1 <= wk <= 53:
            out.append(token)
    return out


def fuse_field(field: str, per_source: List[Tuple[str, List[Candidate]]],
               brand_hint: Optional[str] = None) -> dict:
    tally: Dict[str, float] = defaultdict(float)
    sources_per_value: Dict[str, set] = defaultdict(set)
    engines_per_value: Dict[str, set] = defaultdict(set)

    def _add(value: str, conf: float, source_id: str, engine: str):
        tally[value] += conf
        sources_per_value[value].add(source_id)
        engines_per_value[value].add(engine)

    for source_id, candidates in per_source:
        engine = _source_engine(source_id)
        for c in candidates:
            _add(c.value, c.confidence, source_id, engine)
            if field == "dot":
                for aux in _dot_aux_candidates(c.value):
                    _add(aux, c.confidence * 0.6, source_id, engine)

    if field == "desen" and brand_hint:
        allowed = set(lexicon.BRAND_PATTERNS.get(brand_hint, []))
        if allowed:
            for value in list(tally.keys()):
                if value not in allowed:
                    tally[value] *= BRAND_MISMATCH_PENALTY

    scored = []
    for value, base_score in tally.items():
        bonus = 0.15 * len(engines_per_value[value]) + 0.08 * len(sources_per_value[value])
        scored.append(ScoredValue(value=value, score=base_score + bonus,
                                   n_sources=len(sources_per_value[value]),
                                   n_engines=len(engines_per_value[value])))
    scored.sort(key=lambda s: -s.score)

    if not scored:
        return {"value": None, "confidence": 0.0, "candidates": []}

    if field in MULTI_VALUE_FIELDS:
        top = scored[:6]
        return {
            "values": [s.value for s in top],
            "candidates": [(s.value, round(s.score, 2)) for s in top],
        }

    best = scored[0]
    runner_up = scored[1] if len(scored) > 1 else None
    return {
        "value": best.value,
        "confidence": round(best.score, 2),
        "n_sources": best.n_sources,
        "n_engines": best.n_engines,
        "runner_up": (runner_up.value, round(runner_up.score, 2)) if runner_up else None,
        "candidates": [(s.value, round(s.score, 2)) for s in scored[:5]],
    }


def fuse_all(per_source: List[Tuple[str, FieldCandidates]]) -> Dict[str, dict]:
    result = {}
    marka_sources = [(sid, fc.get("marka", [])) for sid, fc in per_source]
    result["marka"] = fuse_field("marka", marka_sources)
    brand_hint = result["marka"].get("value")

    for field in FIELDS:
        if field == "marka":
            continue
        field_sources = [(sid, fc.get(field, [])) for sid, fc in per_source]
        if field == "desen":
            result[field] = fuse_field(field, field_sources, brand_hint=brand_hint)
        else:
            result[field] = fuse_field(field, field_sources)
    return result
