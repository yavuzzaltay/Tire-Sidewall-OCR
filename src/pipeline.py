"""Tek bir lastik (Ust+Alt) için uçtan uca işlem hattı:
ön işleme -> orientasyon -> döşeme -> varyant -> OCR motoru -> alan çıkarımı -> birleştirme.

benchmark.py ve run_all.py bu modülü paylaşır.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

import extract
import fuse
import lexicon
import preprocess
import tiling
from engines import get_engine


@dataclasses.dataclass
class PipelineConfig:
    engines: List[str]
    variants: List[str]
    tile_width: int = 1600
    tile_overlap: int = 200
    sides: Tuple[str, ...] = ("ust", "alt")


def process_side(path: str, side: str, config: PipelineConfig, orientation_engine
                  ) -> Tuple[str, List[Tuple[str, extract.FieldCandidates, str]]]:
    raw = preprocess.load_gray(path)
    if raw is None:
        return "dosya_yok", []
    oriented = preprocess.resolve_orientation(raw, orientation_engine.score)
    if oriented.status != "ok":
        return "okunamadi", []

    # RapidOCR, width/height oranı ~8'i aşan girdilerde tespiti tamamen atlıyor
    # (bkz. width_height_ratio config'i). İnce bantlarda tile genişliğini güvenli
    # payla sınırlayarak bu tuzağa düşmeyi önlüyoruz.
    band_h = oriented.image.shape[0]
    safe_tile_width = min(config.tile_width, max(400, band_h * 6))
    tiles = tiling.tiles_full_res(oriented.image, safe_tile_width, config.tile_overlap)
    per_source: List[Tuple[str, extract.FieldCandidates, str]] = []
    for tile in tiles:
        for vname in config.variants:
            vfn = preprocess.VARIANTS[vname]
            vimg = vfn(tile.image)
            for ename in config.engines:
                engine = get_engine(ename)
                boxes = engine.read(vimg)
                fields = extract.extract_from_boxes(boxes)
                text = extract.sorted_join(boxes)
                source_id = f"{side}|{tile.x0}|{vname}|{ename}"
                per_source.append((source_id, fields, text))
    return "ok", per_source


def process_tire(ust_path: Optional[str], alt_path: Optional[str], config: PipelineConfig,
                  orientation_engine) -> dict:
    all_sources: List[Tuple[str, extract.FieldCandidates, str]] = []
    statuses: Dict[str, str] = {}
    paths = {"ust": ust_path, "alt": alt_path}
    for side in config.sides:
        path = paths.get(side)
        if not path:
            continue
        status, sources = process_side(path, side, config, orientation_engine)
        statuses[side] = status
        all_sources.extend(sources)

    # 1. geçiş: TÜM kaynaklardan marka için küresel oylama (tek tile'ın gürültülü
    # yerel tahminine güvenmek yerine).
    brand_sources = [(sid, fc.get("marka", [])) for sid, fc, _ in all_sources]
    fused_brand = fuse.fuse_field("marka", brand_sources)
    global_brand = fused_brand.get("value")

    # 2. geçiş: küresel marka biliniyorsa, SADECE o markanın sözlüğüyle sınırlı
    # ikinci bir desen araması yap ve ek oy olarak ekle (extract.py'deki 1. geçiş
    # dokunulmadan kalır; bu sadece ek, daha güvenilir sinyal ekler).
    if global_brand and global_brand in lexicon.BRAND_PATTERNS:
        allowed = [(p, global_brand) for p in lexicon.BRAND_PATTERNS[global_brand]]
        enriched: List[Tuple[str, extract.FieldCandidates, str]] = []
        for source_id, fields, text in all_sources:
            extra = extract.extract_pattern_only(text, 0.65, allowed)
            if extra:
                fields = dict(fields)
                fields["desen"] = list(fields.get("desen", [])) + extra
            enriched.append((source_id, fields, text))
        all_sources = enriched

    fuse_input = [(sid, fc) for sid, fc, _ in all_sources]
    fused = fuse.fuse_all(fuse_input)
    return {"status": statuses, "fields": fused, "n_sources": len(all_sources)}
