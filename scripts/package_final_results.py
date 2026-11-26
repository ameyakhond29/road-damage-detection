#!/usr/bin/env python3
"""Copy verified Week 3–6 tables/figures into results/final, figures/final, tables/final."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RES = ROOT / "results"
FIG = ROOT / "figures"
FINAL_RES = RES / "final"
FINAL_FIG = FIG / "final"
FINAL_TAB = ROOT / "tables" / "final"


def copy_if(src: Path, dest: Path) -> bool:
    if not src.is_file():
        print(f"  MISSING: {src}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    print(f"  OK: {dest.relative_to(ROOT)}")
    return True


def write_table_from_csv(src: Path, dest: Path, keep_cols: list[str] | None = None) -> bool:
    if not src.is_file():
        print(f"  MISSING table source: {src}")
        return False
    with src.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("", encoding="utf-8")
        return True
    if keep_cols:
        keys = [k for k in keep_cols if k in rows[0]]
        rows = [{k: r.get(k) for k in keys} for r in rows]
    else:
        keys = list(rows[0].keys())
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  OK: {dest.relative_to(ROOT)}")
    return True


def main() -> int:
    FINAL_RES.mkdir(parents=True, exist_ok=True)
    FINAL_FIG.mkdir(parents=True, exist_ok=True)
    FINAL_TAB.mkdir(parents=True, exist_ok=True)

    print("Packaging final tables …")
    write_table_from_csv(RES / "week4" / "baseline_comparison.csv", FINAL_TAB / "baseline_comparison.csv")
    write_table_from_csv(RES / "week5" / "seed_summary.csv", FINAL_TAB / "ablation_comparison.csv")
    write_table_from_csv(RES / "week6" / "final_test_results.csv", FINAL_TAB / "final_test_results.csv")
    write_table_from_csv(RES / "week6" / "final_comparison_table.csv", FINAL_TAB / "final_comparison_table.csv")
    write_table_from_csv(RES / "week6" / "efficiency_benchmark.csv", FINAL_TAB / "efficiency_benchmark.csv")
    write_table_from_csv(RES / "week6" / "country_results.csv", FINAL_TAB / "country_level_results.csv")

    # Per-class / small-object extracted from comparison
    comp = RES / "week6" / "final_comparison_table.csv"
    if comp.is_file():
        with comp.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        per_class = []
        small = []
        for r in rows:
            per_class.append(
                {
                    "model_id": r.get("model_id"),
                    "display_name": r.get("display_name"),
                    "val_ap50_D00": r.get("val_ap50_D00"),
                    "val_ap50_D10": r.get("val_ap50_D10"),
                    "val_ap50_D20": r.get("val_ap50_D20"),
                    "val_ap50_D40": r.get("val_ap50_D40"),
                    "test_ap50_D00": r.get("test_ap50_D00"),
                    "test_ap50_D10": r.get("test_ap50_D10"),
                    "test_ap50_D20": r.get("test_ap50_D20"),
                    "test_ap50_D40": r.get("test_ap50_D40"),
                }
            )
            small.append(
                {
                    "model_id": r.get("model_id"),
                    "display_name": r.get("display_name"),
                    "val_small_object_ap": r.get("val_small_object_ap"),
                    "test_small_object_ap": r.get("test_small_object_ap"),
                }
            )
        for name, data in (
            ("per_class_results.csv", per_class),
            ("small_object_results.csv", small),
        ):
            dest = FINAL_TAB / name
            keys = list(data[0].keys()) if data else []
            with dest.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(data)
            print(f"  OK: {dest.relative_to(ROOT)}")

    print("Packaging final result JSON copies …")
    for src, name in (
        (RES / "week6" / "final_comparison_table.json", "final_comparison_table.json"),
        (RES / "week6" / "final_test_results.json", "final_test_results.json"),
        (RES / "week6" / "final_run_manifest.json", "final_run_manifest.json"),
        (RES / "week5" / "model_selection_decision.md", "model_selection_decision.md"),
        (RES / "week6" / "CONFIDENCE_THRESHOLD_LOCKED.json", "CONFIDENCE_THRESHOLD_LOCKED.json"),
        (RES / "week6" / "error_analysis_summary.json", "error_analysis_summary.json"),
    ):
        copy_if(src, FINAL_RES / name)

    print("Packaging final figures …")
    for src, name in (
        (FIG / "week6" / "confusion_matrix.png", "confusion_matrix.png"),
        (FIG / "week6" / "precision_recall_curves.png", "precision_recall_curves.png"),
        (FIG / "week6" / "accuracy_efficiency_tradeoff.png", "accuracy_efficiency_tradeoff.png"),
        (FIG / "week6" / "false_positives.png", "false_positives.png"),
        (FIG / "week6" / "false_negatives.png", "false_negatives.png"),
        (FIG / "week6" / "missed_small_objects.png", "missed_small_objects.png"),
        (FIG / "week6" / "localization_errors.png", "localization_errors.png"),
        (FIG / "week6" / "country_performance.png", "country_performance.png"),
        (FIG / "week5" / "map_comparison.png", "ablation_map_comparison.png"),
        (FIG / "week4" / "baseline_comparison.png", "baseline_comparison.png"),
        (FIG / "class_distribution_modeling_set.png", "class_distribution_modeling_set.png"),
    ):
        copy_if(src, FINAL_FIG / name)

    manifest = {
        "tables": sorted(str(p.relative_to(ROOT)) for p in FINAL_TAB.glob("*") if p.is_file()),
        "figures": sorted(str(p.relative_to(ROOT)) for p in FINAL_FIG.glob("*") if p.is_file()),
        "results": sorted(str(p.relative_to(ROOT)) for p in FINAL_RES.glob("*") if p.is_file()),
        "note": "Copied from Week 3–6 artifacts; values not recomputed or altered.",
    }
    (FINAL_RES / "package_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {FINAL_RES / 'package_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
