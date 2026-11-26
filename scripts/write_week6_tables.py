#!/usr/bin/env python3
"""Build final comparison tables + Week 6 report from saved result files only."""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CLASS_NAMES
from src.evaluation.week6_utils import load_protocol, pending, write_csv, write_json

WEEK6 = ROOT / "results" / "week6"


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v, d=4):
    if pending(v):
        return "Pending"
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    protocol = load_protocol()
    test_rows = read_csv(WEEK6 / "final_test_results.csv")
    eff_rows = read_csv(WEEK6 / "efficiency_benchmark.csv")
    country = read_csv(WEEK6 / "country_results.csv")
    err_sum = read_json(WEEK6 / "error_analysis_summary.json")
    amend = read_json(WEEK6 / "amendment_full_test_counts" / "full_test_counts.json")
    amend_counts = amend.get("full_test_counts", {}) if amend else {}
    amend_country = read_csv(WEEK6 / "amendment_full_test_counts" / "full_test_country_results.csv")
    lock = read_json(WEEK6 / "CONFIDENCE_THRESHOLD_LOCKED.json")
    manifest = read_json(WEEK6 / "final_run_manifest.json")

    by_model_split: dict[tuple[str, str], dict] = {}
    for r in test_rows:
        by_model_split[(r.get("model_id"), r.get("split_label"))] = r
    eff_by = {r.get("model_id"): r for r in eff_rows}

    model_order = ["yolov8n", "yolo11n_week4", "week5_best_ablation", "final_selected"]
    # ensure present
    for mid in list({r.get("model_id") for r in test_rows}):
        if mid not in model_order:
            model_order.append(mid)

    comparison = []
    for mid in model_order:
        val = by_model_split.get((mid, "Validation"), {})
        test = by_model_split.get((mid, "Test"), {})
        eff = eff_by.get(mid, {})
        display = (
            val.get("display_name")
            or test.get("display_name")
            or eff.get("display_name")
            or mid
        )
        row = {
            "model_id": mid,
            "display_name": display,
            "val_map50": val.get("map50", "Pending"),
            "val_map50_95": val.get("map50_95", "Pending"),
            "val_precision": val.get("precision", "Pending"),
            "val_recall": val.get("recall", "Pending"),
            "val_f1": val.get("f1", "Pending"),
            "val_small_object_ap": val.get("small_object_ap", "Pending"),
            "test_map50": test.get("map50", "Pending"),
            "test_map50_95": test.get("map50_95", "Pending"),
            "test_precision": test.get("precision", "Pending"),
            "test_recall": test.get("recall", "Pending"),
            "test_f1": test.get("f1", "Pending"),
            "test_small_object_ap": test.get("small_object_ap", "Pending"),
            "test_fp": test.get("false_positive_count", "Pending"),
            "test_fn": test.get("false_negative_count", "Pending"),
            "parameter_count": eff.get("parameter_count", "Pending"),
            "gflops": eff.get("gflops", "Pending"),
            "latency_ms_mean": eff.get("latency_ms_mean", "Pending"),
            "peak_gpu_memory_mb": eff.get("peak_gpu_memory_mb", "Pending"),
            "training_time_sec": eff.get("training_time_sec", "Pending"),
            "model_file_size_mb": eff.get("model_file_size_mb", "Pending"),
            "identical_to_final": test.get("identical_to_final") or eff.get("identical_to_final"),
        }
        for c in CLASS_NAMES:
            row[f"val_ap50_{c}"] = val.get(f"ap50_{c}", "Pending")
            row[f"test_ap50_{c}"] = test.get(f"ap50_{c}", "Pending")
        comparison.append(row)

    write_csv(WEEK6 / "final_comparison_table.csv", comparison)
    write_json(
        WEEK6 / "final_comparison_table.json",
        {
            "note": "Validation and Test metrics are separate columns — never mixed.",
            "confidence_threshold": lock.get("selected_conf", protocol.get("confidence_threshold")),
            "conf_selected_on": "Validation",
            "rows": comparison,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    # --- Report ---
    conf = lock.get("selected_conf", protocol.get("confidence_threshold"))
    final = next((r for r in comparison if r["model_id"] == "final_selected"), {})
    v8 = next((r for r in comparison if r["model_id"] == "yolov8n"), {})
    v11 = next((r for r in comparison if r["model_id"] == "yolo11n_week4"), {})

    country_lines = []
    country_source = amend_country if amend_country else country
    for r in country_source:
        if r.get("model_id") not in (None, "final_selected"):
            continue
        country_lines.append(
            f"| {r.get('split_label')} | {r.get('country')} | {r.get('n_images')} | "
            f"{r.get('n_objects')} | {fmt(r.get('box_recall_iou50'))} | "
            f"{r.get('country_held_out_protocol')} |"
        )
    if not country_lines:
        country_lines = ["| Pending | Pending | Pending | Pending | Pending | Pending |"]

    comp_lines = []
    for r in comparison:
        comp_lines.append(
            f"| {r['display_name']} | {fmt(r['val_map50'])} | {fmt(r['test_map50'])} | "
            f"{fmt(r['val_small_object_ap'])} | {fmt(r['test_small_object_ap'])} | "
            f"{fmt(r['parameter_count'], 0)} | {fmt(r['gflops'], 2)} | "
            f"{fmt(r['latency_ms_mean'], 2)} |"
        )

    per_class_lines = []
    for r in comparison:
        if r["model_id"] != "final_selected":
            continue
        for c in CLASS_NAMES:
            per_class_lines.append(
                f"| {c} | {fmt(r.get(f'val_ap50_{c}'))} | {fmt(r.get(f'test_ap50_{c}'))} |"
            )
    if not per_class_lines:
        per_class_lines = ["| Pending | Pending | Pending |"]

    run_manifest = read_json(WEEK6 / "final_run_manifest.json") or {}
    test_done = bool(read_json(WEEK6 / "TEST_EVALUATION_COMPLETED.json")) or protocol.get(
        "test_evaluation_completed"
    )

    report = f"""# Week 6 — Final results report

_Generated from saved result files only. No invented metrics._  
_Generated at: {datetime.now(timezone.utc).isoformat()}_

## 0. Data provenance and evaluation scope (read first)

- **Data actually used:** the synthetic fixture (`data/fixture/RDD2022/`), **not** the official RDD2022 download. Every metric in this report is fixture-scale. Official RDD2022 was not present under `data/raw/`.
- **Run mode:** `final_run_manifest.json` records `smoke = {bool(run_manifest.get("smoke"))}`.
  {"The original final test evaluation was run in **smoke mode**. Ultralytics `model.val` metrics (mAP, precision, recall, F1, per-class AP, small-object AP) cover the full test split. The original FP/FN counts and the per-country rows covered only the first 12 test images (sorted by filename) and at most 8 images per country. Those original values are kept in `results/week6/final_test_results.csv` for transparency." if run_manifest.get("smoke") else "The final test evaluation was run in full mode (no image cap)."}
- **Protocol amendment `week6_full_test_counts_v1`:** {"FP/FN counts, the error analysis, and the per-country rows were re-computed on the **full** test split (" + str(amend_counts.get("n_test_images", "Pending")) + " images) using the **same** checkpoint, operating confidence, NMS IoU, and matching IoU. The model, thresholds, and selection were not changed and no retraining was done. Record: `results/week6/amendment_full_test_counts/full_test_counts.json`. The error analysis (Section 10) was re-run on the full split; the original 12-image outputs are archived in `results/week6/archive_smoke_12img/`." if amend_counts else "**Not yet applied.** FP/FN counts and the error analysis cover only the 12-image subset from the smoke run. See `docs/known_issues_and_resolutions.md`, item B1."}

## 1. Final research question

Starting from a small detector, which change best improves **accuracy per added compute** on RDD2022
(P2 head, lightweight attention, or 800 px input), and how does the selected model perform on the
**held-out test set** and across countries present in the fixed split?

## 2. Dataset and split

- Dataset: RDD2022 (China_Drone excluded) — **but the files actually evaluated were built from the synthetic fixture** (see Section 0)
- Classes: D00, D10, D20, D40
- Split: `{protocol.get("split_id")}` (grouped leakage-safe 70/15/15)
- Data config: `{protocol.get("data")}`
- **No new split was created in Week 6.**

## 3. Models evaluated

| Role | Model | Checkpoint (protocol) |
|---|---|---|
| Final selected | {protocol["final_model"]["display_name"]} | `{protocol["final_model"]["checkpoint"]}` |
| Baseline | YOLOv8n Week 3 | `{protocol["baselines"]["yolov8n"]["checkpoint"]}` |
| Baseline | YOLO11n Week 4 | `{protocol["baselines"]["yolo11n_week4"]["checkpoint"]}` |
| Best Week 5 ablation | {protocol["week5_best_ablation"]["display_name"]} | identical to final |

Selection source: `results/week5/model_selection_decision.md` (validation only).

## 4. Frozen evaluation protocol

See `docs/final_evaluation_protocol.md` and `configs/final_evaluation.yaml`.

| Item | Value |
|---|---|
| imgsz | {protocol.get("imgsz")} |
| NMS IoU | {protocol.get("iou_threshold")} |
| Operating conf (val-selected) | **{conf}** |
| conf for mAP | {protocol.get("conf_for_map")} |
| Test evaluation completed | {test_done} |
| Test used for selection | **False** |

Confidence lock file: `results/week6/CONFIDENCE_THRESHOLD_LOCKED.json`.

## 5. Final test results

Final model (`final_selected`) — **Test** metrics from `final_test_results.csv`:

| Metric | Validation | Test |
|---|---|---|
| mAP@0.5 | {fmt(final.get("val_map50"))} | {fmt(final.get("test_map50"))} |
| mAP@0.5:0.95 | {fmt(final.get("val_map50_95"))} | {fmt(final.get("test_map50_95"))} |
| Precision | {fmt(final.get("val_precision"))} | {fmt(final.get("test_precision"))} |
| Recall | {fmt(final.get("val_recall"))} | {fmt(final.get("test_recall"))} |
| F1 | {fmt(final.get("val_f1"))} | {fmt(final.get("test_f1"))} |
| Small-object AP | {fmt(final.get("val_small_object_ap"))} | {fmt(final.get("test_small_object_ap"))} |
| FP count (op. conf), full split (amendment) | — | {fmt(amend_counts.get("false_positive_count", "Pending"), 0)} |
| FN count (op. conf), full split (amendment) | — | {fmt(amend_counts.get("false_negative_count", "Pending"), 0)} |
| TP count, full split (amendment) | — | {fmt(amend_counts.get("true_positive_count", "Pending"), 0)} |
| Ground-truth objects, full split | — | {fmt(amend_counts.get("n_gt_objects", "Pending"), 0)} |
| FP count, original smoke run (12 images, kept for transparency) | — | {fmt(final.get("test_fp"), 0)} |
| FN count, original smoke run (12 images, kept for transparency) | — | {fmt(final.get("test_fn"), 0)} |

Raw: `results/week6/final_test_results.csv`.

## 6. Comparison with baselines

Validation and Test are **separate columns**.

| Model | Val mAP50 | Test mAP50 | Val small-AP | Test small-AP | Params | GFLOPs | Latency ms |
|---|---|---|---|---|---|---|---|
{chr(10).join(comp_lines)}

## 7. Per-class performance (final model)

| Class | Val AP50 | Test AP50 |
|---|---|---|
{chr(10).join(per_class_lines)}

## 8. Small-object performance

Small-object AP uses GT boxes with normalized area &lt; {protocol.get("small_object_area_threshold", 0.02)}.

| Split | Small-object AP |
|---|---|
| Validation | {fmt(final.get("val_small_object_ap"))} |
| Test | {fmt(final.get("test_small_object_ap"))} |

## 9. Country-level robustness

**No new country-held-out split was created.** The existing grouped split’s test portion is
100% `United_States` (incidental composition). Validation contains `China_MotorBike` and `Norway`.
Country-held-out OOD protocol: **not available** without redesigning the split (explicitly avoided).

| Split label | Country | Images | Objects | Box recall@0.5 (proxy) | Country-held-out? |
|---|---|---|---|---|---|
{chr(10).join(country_lines)}

Figure: `figures/week6/country_performance.png`.  
CSV: `results/week6/country_results.csv`.

{"**Count source:** the test rows above come from the protocol amendment (`results/week6/amendment_full_test_counts/full_test_country_results.csv`), which covers the full test split (50 images, 55 boxes). The original smoke-run country rows (8-image cap) are archived in `results/week6/country_results.csv`. Baseline models are not shown in the country table." if amend_country else "**Count caveat:** the Image and Object counts above are from the smoke run and are capped (8 images per country). The test split has 50 images and 55 boxes. The amendment has not been applied yet."}

## 10. Error analysis

Summary from `error_analysis_summary.json`:

- Split analyzed: {err_sum.get("split_label", "Pending")}
- Images: {err_sum.get("n_images", "Pending")}
- Error rows: {err_sum.get("n_error_rows", "Pending")}
- Multi-damage images: {err_sum.get("n_multi_damage_images", "Pending")}
- Pothole FNs: {err_sum.get("n_pothole_fn", "Pending")}
- Lighting/weather tags: **Pending** (not in RDD2022 metadata)

Figures: `figures/week6/false_positives.png`, `false_negatives.png`, `missed_small_objects.png`,
`localization_errors.png`, `confusion_matrix.png`, `precision_recall_curves.png`.  
Cases: `results/week6/error_cases.csv`.

## 11. Efficiency analysis

From `efficiency_benchmark.csv` (actual device recorded; claim T4 only if `is_t4=true`):

| Model | Params | GFLOPs | File MB | Latency mean/p50/p95 | Img/s | Peak GPU MB | Train sec |
|---|---|---|---|---|---|---|---|
"""

    for r in eff_rows:
        if r.get("identical_to_final"):
            continue
        report += (
            f"| {r.get('display_name')} | {fmt(r.get('parameter_count'), 0)} | {fmt(r.get('gflops'), 2)} | "
            f"{fmt(r.get('model_file_size_mb'), 2)} | "
            f"{fmt(r.get('latency_ms_mean'), 2)} / {fmt(r.get('latency_ms_p50'), 2)} / {fmt(r.get('latency_ms_p95'), 2)} | "
            f"{fmt(r.get('images_per_second'), 2)} | {fmt(r.get('peak_gpu_memory_mb'), 1)} | "
            f"{fmt(r.get('training_time_sec'), 1)} |\n"
        )

    report += f"""
Figure: `figures/week6/accuracy_efficiency_tradeoff.png` (Test mAP vs GFLOPs).

## 12. Main findings

1. Week 5 selected **YOLO11n@640** as the final candidate using **validation only**; no ablation
   beat the control on available val mAP (smoke caveat documented in Week 5).
2. Operating confidence **{conf}** was locked from a **validation** sweep before test evaluation.
3. Held-out **Test** metrics for the final model: mAP50={fmt(final.get("test_map50"))},
   mAP50-95={fmt(final.get("test_map50_95"))} (from saved files).
4. vs YOLOv8n Test mAP50={fmt(v8.get("test_map50"))}; vs Week-4 YOLO11n Test mAP50={fmt(v11.get("test_map50"))}.
5. Test geography is United States–only under the fixed split — interpret country robustness cautiously.
6. Seeds 1/2 for the final model remain **Pending** where checkpoints were not trained.

Do not overstate improvements when smoke/near-zero metrics dominate.

## 13. Limitations

- All runs used the synthetic fixture, not the official RDD2022 download.
- Many Week 3–5 runs were smoke (short epochs); absolute mAP may be near zero.
- The original final test run was in smoke mode. Counts were re-computed on the full split by documented amendment (Section 0); model, thresholds, and selection were not changed.
- Multi-seed (0/1/2) final comparison incomplete if only seed 0 exists.
- Country-held-out OOD was not redesigned; test≠multi-country.
- Small-object AP is an area-filter approximation.
- T4 GPU may be unavailable locally — efficiency numbers are device-labeled.

## 14. Threats to validity

- Selection among near-tied / near-zero smoke runs can be unstable.
- Single test pass prevents peeking but also prevents correcting silent eval bugs without a documented breach.
- Proxy country box-recall ≠ official per-country mAP.
- Fixture data ≠ the RDD2022 distribution. Results do not transfer to RDD2022 claims.
- The one-shot test protocol assumed full test coverage. The original run did not meet that for FP/FN counts; the amendment re-read the full split for counts only. A reader should treat the amendment as a second look at the test split, made after the original results were observed.

## 15. Recommended future work

- Complete full-epoch seeds 0/1/2 for Week 5 ablations; re-freeze only if validation ranking changes
  **before** any new test peek (would be a new protocol version).
- Designed country-held-out OOD track (documented separately; do not silently replace the main split).
- Stronger small-object heads once real multi-seed val gains appear.
- Calibrated operating points per class (still val-only).

---

## Reproducibility

```bash
./scripts/run_week6.sh --smoke
./scripts/run_week6.sh
```

Manifest: `results/week6/final_run_manifest.json`.  
Protocol: `configs/final_evaluation.yaml`.
"""

    out = ROOT / "docs" / "week6_final_results_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"Wrote {WEEK6 / 'final_comparison_table.csv'}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
