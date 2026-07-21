"""benchmark.py'nin kazandığı konfigürasyonla 119 lastiğin tamamını işler.

Çıktı: results/sonuclar.csv (lastik başına 9 alan + güven + OKUNAMADI bayrakları)
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import extract
from engines import get_engine
from pipeline import PipelineConfig, process_tire

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO_DIR = os.path.join(BASE_DIR, "Lastik_fotolari")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def discover_tire_ids():
    ust_dir = os.path.join(PHOTO_DIR, "Lastigin_ustu")
    ids = []
    for path in glob.glob(os.path.join(ust_dir, "Ust*.jpg")):
        name = os.path.basename(path)
        m = re.match(r"Ust(?:_(\d+))?\.jpg$", name)
        if m:
            ids.append(m.group(1) if m.group(1) else "")
    return sorted(set(ids), key=lambda i: (-1 if i == "" else int(i)))


def _cost(r: dict) -> int:
    return len(r["engines"]) * len(r["variants"])


def _priority_avg(r: dict) -> float:
    vals = [r["accuracy"].get(f) for f in ["marka", "desen", "ebat", "dot"]]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def pick_best_config(n_tires_full: int, max_minutes: float = 90.0, gpu_speedup: float = 1.0) -> PipelineConfig:
    """gpu_speedup: benchmark_log.json CPU üzerinde ölçüldü. GPU aktifse (bkz. §6.1/§11 rapor),
    RapidOCR ~3.1x, EasyOCR ~6.4x hızlanıyor; bu, önceden "90 dk sınırını aşıyor" diye elenen
    daha isabetli ensemble konfigürasyonların artık bütçeye sığmasını sağlıyor."""
    log_path = os.path.join(RESULTS_DIR, "benchmark_log.json")
    if not os.path.exists(log_path):
        print("UYARI: benchmark_log.json bulunamadı, varsayılan konfig kullanılıyor (rapidocr + v1_clahe)")
        return PipelineConfig(engines=["rapidocr"], variants=["v1_clahe"], tile_width=2400)

    with open(log_path, encoding="utf-8") as f:
        results = json.load(f)

    def est_minutes(r: dict) -> float:
        # benchmark 9 lastik üzerinden ölçüldü (CPU); 119'un tamamı için oranla, GPU hızlanmasını uygula
        return (r["elapsed_s"] / 9.0) * n_tires_full / 60.0 / gpu_speedup

    simple = [r for r in results if _cost(r) == 1]
    best_simple = max(simple, key=_priority_avg) if simple else None
    best_overall = max(results, key=_priority_avg)

    margin = _priority_avg(best_overall) - (_priority_avg(best_simple) if best_simple else -1)
    overall_minutes = est_minutes(best_overall)

    if best_simple and margin < 0.05:
        chosen, reason = best_simple, f"basit (tek motor+tek varyant) konfig ensemble'a çok yakın (fark {margin*100:.0f} puan)"
    elif best_simple and overall_minutes > max_minutes:
        chosen, reason = best_simple, (
            f"ensemble +{margin*100:.0f} puan kazandırıyor ama 119 lastikte tahmini ~{overall_minutes:.0f} dk sürer "
            f"(sınır {max_minutes:.0f} dk); bu oturum için basit konfig tercih edildi"
        )
    else:
        chosen, reason = best_overall, f"ensemble +{margin*100:.0f} puan kazanç sağladı, tahmini süre ~{overall_minutes:.0f} dk kabul edilebilir"

    print(f"Benchmark'tan seçilen konfig: {chosen['name']}  (GPU hızlanma çarpanı: x{gpu_speedup:.1f})")
    print(f"  Gerekçe: {reason}")
    print(f"  motorlar={chosen['engines']}, varyantlar={chosen['variants']}, "
          f"öncelikli doğruluk={_priority_avg(chosen):.2f}, maliyet çarpanı=x{_cost(chosen)}, "
          f"tahmini süre=~{est_minutes(chosen):.0f} dk")
    return PipelineConfig(engines=chosen["engines"], variants=chosen["variants"], tile_width=2400)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tire_ids = discover_tire_ids()
    # GPU aktif (CUDA'lı torch + onnxruntime-gpu kuruldu, bkz. rapor §6.1/§11): ölçülen
    # hızlanma RapidOCR icin 3.1x, EasyOCR icin 6.4x idi; karma motor konfigleri icin
    # temkinli/ortak bir carpan kullaniyoruz.
    config = pick_best_config(len(tire_ids), gpu_speedup=3.0)
    orientation_engine = get_engine("rapidocr")

    resume = "--resume" in sys.argv
    rows = []
    sonuclar_path = os.path.join(RESULTS_DIR, "sonuclar.csv")
    done_ids = set()
    if resume and os.path.exists(sonuclar_path):
        prev = pd.read_csv(sonuclar_path, dtype=str)
        rows = prev.to_dict("records")
        done_ids = set(prev["tire_id"].astype(str))
        print(f"--resume: {len(rows)} lastik zaten işlenmiş (atlanacak), {len(tire_ids) - len(done_ids)} kaldı.\n")

    print(f"{len(tire_ids)} lastik bulundu. İşleniyor...\n")

    t0 = time.time()
    for i, tid in enumerate(tire_ids):
        tid_key = tid if tid else "base"
        if tid_key in done_ids:
            continue
        suffix = f"_{tid}" if tid else ""
        ust_name, alt_name = f"Ust{suffix}.jpg", f"Alt{suffix}.jpg"
        ust_full = os.path.join(PHOTO_DIR, "Lastigin_ustu", ust_name)
        alt_full = os.path.join(PHOTO_DIR, "Lastigin_alti", alt_name)
        ust_path = ust_full if os.path.exists(ust_full) else None
        alt_path = alt_full if os.path.exists(alt_full) else None

        t1 = time.time()
        result = process_tire(ust_path, alt_path, config, orientation_engine)
        elapsed = time.time() - t1

        row = {
            "tire_id": tid if tid else "base", "ust_file": ust_name, "alt_file": alt_name,
            "ust_status": result["status"].get("ust"), "alt_status": result["status"].get("alt"),
            "elapsed_s": round(elapsed, 1),
        }
        for field in extract.FIELDS:
            fd = result["fields"].get(field, {})
            if field == "ekstra_lot":
                row[field] = " | ".join(fd.get("values", []) or [])
            else:
                row[field] = fd.get("value")
                row[f"{field}_conf"] = fd.get("confidence")
        rows.append(row)
        n_done_total = len(rows)
        n_new = n_done_total - len(done_ids)

        avg_so_far = (time.time() - t0) / n_new
        remaining = avg_so_far * (len(tire_ids) - n_done_total)
        print(f"[{n_done_total}/{len(tire_ids)}] {tid or '(base)':>4}  {elapsed:5.1f}s  "
              f"marka={row.get('marka')!r} desen={row.get('desen')!r} ebat={row.get('ebat')!r} dot={row.get('dot')!r}"
              f"  (kalan ~{remaining/60:.0f} dk)")

        if n_new % 10 == 0:
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "sonuclar.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "sonuclar.csv"), index=False)
    total = time.time() - t0
    print(f"\nTamamlandı: {len(rows)} lastik, {total:.0f}s ({total/max(1,len(rows)):.1f}s/lastik ortalama)")
    print(f"Kaydedildi: {RESULTS_DIR}\\sonuclar.csv")

    n_unreadable_ust = (df["ust_status"] == "okunamadi").sum()
    n_unreadable_alt = (df["alt_status"] == "okunamadi").sum()
    print(f"\nOkunamayan (kalite kapısı): Ust={n_unreadable_ust}/{len(df)}, Alt={n_unreadable_alt}/{len(df)}")
    for field in ["marka", "desen", "ebat", "hiz_grubu", "mevsim", "dot", "uretim_yeri", "tup"]:
        n_found = df[field].notna().sum()
        print(f"  {field}: {n_found}/{len(df)} lastikte bir değer bulundu ({100*n_found/len(df):.0f}%)")


if __name__ == "__main__":
    main()
