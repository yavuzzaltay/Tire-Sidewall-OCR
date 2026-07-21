"""Etiketli 9 lastik üzerinde motor x varyant kombinasyonlarını karşılaştırır.

Aşamalı çalışır (tam çapraz çarpım yerine, süreyi makul tutmak için):
  A) Motor karşılaştırması (sabit varyant=v1_clahe, RapidOCR/EasyOCR/Tesseract tek tek)
  B) Varyant karşılaştırması (sabit motor=A'nın kazananı, 5 varyant tek tek)
  C) Birleşim/ensemble (en iyi motor + tüm varyantlar birlikte; en iyi 2 motor + en iyi varyant)

Sonuçlar results/benchmark_results.csv ve results/benchmark_log.json içine yazılır.
"""
from __future__ import annotations

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import preprocess
import scoring
from engines import ENGINE_NAMES, get_engine
from pipeline import PipelineConfig, process_tire

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO_DIR = os.path.join(BASE_DIR, "Lastik_fotolari")
GT_PATH = os.path.join(BASE_DIR, "data", "ground_truth.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

VARIANT_NAMES = list(preprocess.VARIANTS.keys())


def load_ground_truth() -> pd.DataFrame:
    df = pd.read_csv(GT_PATH, dtype=str)
    return df


def run_config(name: str, config: PipelineConfig, gt_df: pd.DataFrame, orientation_engine) -> dict:
    print(f"\n=== Konfig: {name} (motorlar={config.engines}, varyantlar={config.variants}) ===")
    t0 = time.time()
    per_field_correct = {f: 0 for f in scoring.ALL_SCORED_FIELDS}
    per_field_total = {f: 0 for f in scoring.ALL_SCORED_FIELDS}
    per_tire_details = []

    for _, row in gt_df.iterrows():
        ust_path = os.path.join(PHOTO_DIR, "Lastigin_ustu", row["ust_file"])
        alt_path = os.path.join(PHOTO_DIR, "Lastigin_alti", row["alt_file"])
        t_tire0 = time.time()
        result = process_tire(ust_path, alt_path, config, orientation_engine)
        elapsed = time.time() - t_tire0

        gt_row = row.to_dict()
        field_scores = scoring.score_tire(result["fields"], gt_row)
        for f, val in field_scores.items():
            if val is None:
                continue
            per_field_total[f] += 1
            if val:
                per_field_correct[f] += 1

        predicted_summary = {f: result["fields"].get(f, {}).get("value") for f in scoring.ALL_SCORED_FIELDS}
        per_tire_details.append({
            "tire_id": row["tire_id"], "elapsed_s": round(elapsed, 1),
            "status": result["status"], "predicted": predicted_summary,
            "scores": field_scores,
        })
        tags = "".join("✓" if field_scores[f] else ("·" if field_scores[f] is None else "✗")
                        for f in scoring.ALL_SCORED_FIELDS)
        print(f"  lastik {row['tire_id']:>4}  [{tags}]  {elapsed:5.1f}s  "
              f"marka={predicted_summary['marka']!r} desen={predicted_summary['desen']!r} "
              f"ebat={predicted_summary['ebat']!r} dot={predicted_summary['dot']!r}")

    total_elapsed = time.time() - t0
    accuracy = {f: (per_field_correct[f] / per_field_total[f] if per_field_total[f] else None)
                for f in scoring.ALL_SCORED_FIELDS}
    print(f"  -> toplam süre {total_elapsed:.0f}s | doğruluk: " +
          ", ".join(f"{f}={accuracy[f]*100:.0f}%" if accuracy[f] is not None else f"{f}=-"
                    for f in scoring.ALL_SCORED_FIELDS))

    return {
        "name": name, "engines": config.engines, "variants": config.variants,
        "accuracy": accuracy, "correct": per_field_correct, "total": per_field_total,
        "elapsed_s": round(total_elapsed, 1), "per_tire": per_tire_details,
    }


def main():
    gt_df = load_ground_truth()
    orientation_engine = get_engine("rapidocr")

    all_results = []

    # --- Aşama A: motor karşılaştırması (sabit varyant v1_clahe) ---
    for ename in ENGINE_NAMES:
        cfg = PipelineConfig(engines=[ename], variants=["v1_clahe"], tile_width=2400)
        all_results.append(run_config(f"motor={ename}", cfg, gt_df, orientation_engine))

    priority_avg = lambda r: sum(r["accuracy"][f] or 0 for f in scoring.PRIORITY_FIELDS) / len(scoring.PRIORITY_FIELDS)
    stage_a = [r for r in all_results if r["name"].startswith("motor=")]
    best_engine_result = max(stage_a, key=priority_avg)
    best_engine = best_engine_result["engines"][0]
    print(f"\n>>> Aşama A kazananı (öncelikli alanlarda): {best_engine}")

    # --- Aşama B: varyant karşılaştırması (sabit motor=kazanan) ---
    for vname in VARIANT_NAMES:
        if vname == "v1_clahe":
            continue
        cfg = PipelineConfig(engines=[best_engine], variants=[vname], tile_width=2400)
        all_results.append(run_config(f"varyant={vname} (motor={best_engine})", cfg, gt_df, orientation_engine))

    stage_b = [r for r in all_results if r["name"].startswith("varyant=")]
    stage_b.append(next(r for r in stage_a if r["engines"] == [best_engine]))  # v1_clahe sonucu da dahil
    best_variant_result = max(stage_b, key=priority_avg)
    best_variant = best_variant_result["variants"][0]
    print(f"\n>>> Aşama B kazananı: {best_variant}")

    # --- Aşama C: birleşim/ensemble (süreyi makul tutmak için sınırlı sayıda kombinasyon) ---
    cfg_all_variants = PipelineConfig(engines=[best_engine], variants=VARIANT_NAMES, tile_width=2400)
    all_results.append(run_config(f"ensemble: motor={best_engine} + tüm varyantlar", cfg_all_variants, gt_df, orientation_engine))

    other_engines = [e for e in ENGINE_NAMES if e != best_engine]
    if other_engines:
        second_engine_result = max([r for r in stage_a if r["engines"][0] != best_engine], key=priority_avg)
        second_engine = second_engine_result["engines"][0]
        cfg_two_engines = PipelineConfig(engines=[best_engine, second_engine], variants=[best_variant], tile_width=2400)
        all_results.append(run_config(f"ensemble: motor={best_engine}+{second_engine}, varyant={best_variant}",
                                       cfg_two_engines, gt_df, orientation_engine))

    # --- kaydet ---
    summary_rows = []
    for r in all_results:
        row = {"config": r["name"], "elapsed_s": r["elapsed_s"]}
        row.update({f"acc_{f}": r["accuracy"][f] for f in scoring.ALL_SCORED_FIELDS})
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "benchmark_results.csv"), index=False)

    with open(os.path.join(RESULTS_DIR, "benchmark_log.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n\n========== ÖZET TABLO ==========")
    print(summary_df.to_string(index=False))
    print(f"\nKaydedildi: {RESULTS_DIR}\\benchmark_results.csv, benchmark_log.json")

    return all_results, best_engine, best_variant


if __name__ == "__main__":
    main()
