"""OCR kutularından (bir kaynak: tek tile x tek varyant x tek motor çıktısı)
alan adaylarını (Ebat, Marka, Desen, DOT, ...) çıkarır.

Her alan için birden çok Candidate dönebilir; nihai seçim fuse.py'de yapılır.
"""
from __future__ import annotations

import dataclasses
import re
from typing import Dict, List

from rapidfuzz import fuzz, process, utils as fuzz_utils

import lexicon
from engines.base import OcrBox

FIELDS = [
    "ebat", "hiz_grubu", "marka", "desen", "mevsim",
    "dot", "ekstra_lot", "uretim_yeri", "tup",
]


@dataclasses.dataclass
class Candidate:
    value: str
    confidence: float
    evidence: str = ""


FieldCandidates = Dict[str, List[Candidate]]

# ---------------------------------------------------------------------------
# Regex desenleri
# ---------------------------------------------------------------------------

SIZE_RE = re.compile(r"(\d{3})\s*[/\\]\s*(\d{2})\s*(ZR|Z\s?R|R)\s*[\s.]{0,3}(\d{2})\b", re.IGNORECASE)
LOAD_SPEED_RE = re.compile(r"\b(\d{2,3})\s*([A-Z]{1,2})\b")
SPEED_ONLY_RE = re.compile(r"\b([A-HJ-NP-TV-Z])\b")
VALID_SPEED_LETTERS = set("ABCDEFGHJKLMNPQRSTUVWYZ")
MADE_IN_RE = re.compile(r"MADE\s+IN\s+([A-Z]{3,})", re.IGNORECASE)
TUBELESS_RE = re.compile(r"TUBELESS", re.IGNORECASE)
TUBETYPE_RE = re.compile(r"TUBE\s*TYPE", re.IGNORECASE)
DOT_ANCHOR_RE = re.compile(r"D[O0]T", re.IGNORECASE)
FOURDIGIT_RE = re.compile(r"\b(\d{4})\b")
ALNUM_CODE_RE = re.compile(r"\b(?=[A-Z0-9]{5,10}\b)(?=[A-Z0-9]*[A-Z])(?=[A-Z0-9]*\d)[A-Z0-9]{5,10}\b")


def _clean(text: str) -> str:
    return text.strip()


def sorted_join(boxes: List[OcrBox]) -> str:
    """Kutuları x-merkezine göre soldan sağa sıralayıp birleştirir (tek satırlık
    tile'larda okuma sırasını yaklaşık olarak geri kazanır)."""

    def xc(b: OcrBox) -> float:
        if not b.bbox:
            return 0.0
        xs = [p[0] for p in b.bbox]
        return sum(xs) / len(xs)

    ordered = sorted(boxes, key=xc)
    return " ".join(_clean(b.text) for b in ordered if _clean(b.text))


# ---------------------------------------------------------------------------
# Alan çıkarıcılar
# ---------------------------------------------------------------------------


def _extract_size_and_speed(text: str, base_conf: float) -> tuple[List[Candidate], List[Candidate]]:
    size_out: List[Candidate] = []
    speed_out: List[Candidate] = []
    for m in SIZE_RE.finditer(text):
        section, aspect, constr, rim = m.groups()
        constr_norm = "ZR" if constr.upper().replace(" ", "") == "ZR" else "R"
        value = f"{section}/{aspect} {constr_norm}{rim}"
        size_out.append(Candidate(value=value, confidence=base_conf, evidence=m.group(0)))

        tail = text[m.end(): m.end() + 25]
        lm = LOAD_SPEED_RE.search(tail)
        matched_speed = False
        if lm:
            load, speed = lm.groups()
            speed = speed.upper()
            if speed and speed[-1] in VALID_SPEED_LETTERS and speed[-1] not in "R":
                speed_out.append(Candidate(value=f"{load}{speed}", confidence=base_conf * 0.9, evidence=lm.group(0)))
                matched_speed = True
        if not matched_speed:
            sm = SPEED_ONLY_RE.search(tail)
            if sm and sm.group(1) in VALID_SPEED_LETTERS:
                speed_out.append(Candidate(value=sm.group(1), confidence=base_conf * 0.6, evidence=sm.group(0)))
    return size_out, speed_out


def _extract_made_in(text: str, base_conf: float) -> List[Candidate]:
    out = []
    for m in MADE_IN_RE.finditer(text):
        out.append(Candidate(value=m.group(1).title(), confidence=base_conf, evidence=m.group(0)))
    return out


def _extract_tube(text: str, base_conf: float) -> List[Candidate]:
    out = []
    if TUBELESS_RE.search(text):
        out.append(Candidate(value="Tüpsüz", confidence=base_conf, evidence="TUBELESS"))
    if TUBETYPE_RE.search(text):
        out.append(Candidate(value="Tüplü", confidence=base_conf, evidence="TUBE TYPE"))
    return out


def _extract_dot(text: str, base_conf: float) -> List[Candidate]:
    out: List[Candidate] = []
    anchors = list(DOT_ANCHOR_RE.finditer(text))
    if anchors:
        for m in anchors:
            tail = text[m.end(): m.end() + 40]
            codes = re.findall(r"[A-Z0-9]{2,6}", tail.upper())
            if codes:
                out.append(Candidate(value=" ".join(codes), confidence=base_conf, evidence="DOT " + tail[:40]))
            for fm in FOURDIGIT_RE.finditer(tail):
                token = fm.group(1)
                wk, yr = int(token[:2]), int(token[2:])
                if 1 <= wk <= 53:
                    out.append(Candidate(value=token, confidence=min(0.97, base_conf + 0.1),
                                          evidence=f"DOT tarih adayı (hafta {wk:02d}, yıl 20{yr:02d})"))
    else:
        for fm in FOURDIGIT_RE.finditer(text):
            token = fm.group(1)
            wk, yr = int(token[:2]), int(token[2:])
            if 1 <= wk <= 53 and 15 <= yr <= 30:
                out.append(Candidate(value=token, confidence=base_conf * 0.55,
                                      evidence=f"bağımsız tarih adayı (hafta {wk:02d}, yıl 20{yr:02d})"))
    return out


def _extract_extra_codes(text: str, base_conf: float, exclude: set[str]) -> List[Candidate]:
    out = []
    for m in ALNUM_CODE_RE.finditer(text.upper()):
        token = m.group(0)
        if token in exclude:
            continue
        out.append(Candidate(value=token, confidence=base_conf * 0.4, evidence="ham alfasayısal kod"))
    return out


def _ngrams(tokens: List[str], n: int) -> List[str]:
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _extract_brand_pattern(text: str, base_conf: float) -> tuple[List[Candidate], List[Candidate]]:
    """Marka + desen adaylarını TÜM sözlük üstünde arar (tek-kaynak, bağlamsız geçiş).
    Not: bir tile'da lokal olarak bulunan marka, o tile'ın kendi desen aramasını
    kısıtlamak için KULLANILMIYOR — denendi ama lokal marka eşleşmeleri de gürültülü
    çıktığından net bir iyileşme sağlamadı (bkz. results/rapor.md). Bunun yerine
    pipeline.py iki geçişli bir yaklaşım kullanıyor: önce TÜM kaynaklardan marka
    oylanıyor, sonra extract_pattern_only() ile o küresel markaya kısıtlı ikinci
    bir desen araması yapılıp sonuçlar ek oy olarak ekleniyor."""
    tokens = [t for t in re.split(r"\s+", text) if t]
    brand_out: List[Candidate] = []
    pattern_out: List[Candidate] = []

    candidates_txt = tokens + _ngrams(tokens, 2)
    for tok in candidates_txt:
        if len(tok) < 4:
            continue
        match = process.extractOne(tok, lexicon.BRANDS, scorer=fuzz.WRatio, score_cutoff=82,
                                    processor=fuzz_utils.default_process)
        if match:
            name, score, _ = match
            brand_out.append(Candidate(value=name, confidence=base_conf * (score / 100.0), evidence=tok))

    pattern_names = [p for p, _ in lexicon.ALL_PATTERNS]
    windows = tokens + _ngrams(tokens, 2) + _ngrams(tokens, 3) + _ngrams(tokens, 4)
    seen_best: Dict[str, float] = {}
    for win in windows:
        if len(win) < 5:
            continue
        match = process.extractOne(win, pattern_names, scorer=fuzz.WRatio, score_cutoff=88,
                                    processor=fuzz_utils.default_process)
        if match:
            name, score, _ = match
            if score > seen_best.get(name, 0):
                seen_best[name] = score
    pattern_brand_map = dict(lexicon.ALL_PATTERNS)
    for name, score in seen_best.items():
        pattern_out.append(Candidate(value=name, confidence=base_conf * (score / 100.0), evidence="desen sözlük eşleşmesi"))
        brand_out.append(Candidate(value=pattern_brand_map[name], confidence=base_conf * (score / 100.0) * 0.8,
                                    evidence=f"desenden çıkarım: {name}"))
    return brand_out, pattern_out


def extract_pattern_only(text: str, base_conf: float, allowed_patterns: List[tuple[str, str]],
                          cutoff: float = 80) -> List[Candidate]:
    """İkinci geçiş: sadece verilen (genelde küresel oylamayla belirlenmiş marka)
    sözlüğüne karşı desen arar. Daha dar sözlük => daha düşük cutoff güvenle kullanılabilir."""
    if not allowed_patterns:
        return []
    tokens = [t for t in re.split(r"\s+", text) if t]
    windows = tokens + _ngrams(tokens, 2) + _ngrams(tokens, 3) + _ngrams(tokens, 4)
    pattern_names = [p for p, _ in allowed_patterns]
    seen_best: Dict[str, float] = {}
    for win in windows:
        if len(win) < 4:
            continue
        match = process.extractOne(win, pattern_names, scorer=fuzz.WRatio, score_cutoff=cutoff,
                                    processor=fuzz_utils.default_process)
        if match:
            name, score, _ = match
            if score > seen_best.get(name, 0):
                seen_best[name] = score
    return [Candidate(value=name, confidence=base_conf * (score / 100.0), evidence="desen (küresel marka kısıtlı 2. geçiş)")
            for name, score in seen_best.items()]


def _extract_season(text: str, pattern_candidates: List[Candidate], base_conf: float) -> List[Candidate]:
    out: List[Candidate] = []
    lower = text.lower()
    for marker, season in lexicon.SEASON_TEXT_MARKERS:
        if marker.lower() in lower:
            out.append(Candidate(value=season, confidence=base_conf, evidence=f"metin işareti: {marker}"))
    for cand in pattern_candidates:
        pl = cand.value.lower()
        for kw, season in lexicon.SEASON_KEYWORDS:
            if kw in pl:
                out.append(Candidate(value=season, confidence=cand.confidence * 0.85,
                                      evidence=f"desen adından: {cand.value}"))
                break
    return out


# ---------------------------------------------------------------------------
# Ana giriş noktası
# ---------------------------------------------------------------------------


def extract_from_boxes(boxes: List[OcrBox]) -> FieldCandidates:
    result: FieldCandidates = {f: [] for f in FIELDS}
    if not boxes:
        return result

    text = sorted_join(boxes)
    text_u = text.upper()
    avg_conf = sum(b.confidence for b in boxes) / len(boxes)

    size_c, speed_c = _extract_size_and_speed(text_u, avg_conf)
    result["ebat"].extend(size_c)
    result["hiz_grubu"].extend(speed_c)
    result["uretim_yeri"].extend(_extract_made_in(text_u, avg_conf))
    result["tup"].extend(_extract_tube(text_u, avg_conf))
    result["dot"].extend(_extract_dot(text_u, avg_conf))

    brand_c, pattern_c = _extract_brand_pattern(text, avg_conf)
    result["marka"].extend(brand_c)
    result["desen"].extend(pattern_c)
    result["mevsim"].extend(_extract_season(text_u, pattern_c, avg_conf))

    exclude = set()
    for f in ("ebat", "dot"):
        exclude.update(c.evidence.upper() for c in result[f])
    result["ekstra_lot"].extend(_extract_extra_codes(text_u, avg_conf, exclude))

    return result


def merge_field_candidates(a: FieldCandidates, b: FieldCandidates) -> FieldCandidates:
    out: FieldCandidates = {f: list(a.get(f, [])) for f in FIELDS}
    for f in FIELDS:
        out[f].extend(b.get(f, []))
    return out
