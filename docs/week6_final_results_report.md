# Week 6 — Final results report

_Generated from saved result files only. No invented metrics._  
_Generated at: 2026-11-26T03:33:44.528279+00:00_

## 0. Data provenance and evaluation scope (read first)

- **Data actually used:** the synthetic fixture (`data/fixture/RDD2022/`), **not** the official RDD2022 download. Every metric in this report is fixture-scale. Official RDD2022 was not present under `data/raw/`.
- **Run mode:** `final_run_manifest.json` records `smoke = True`.
  The original final test evaluation was run in **smoke mode**. Ultralytics `model.val` metrics (mAP, precision, recall, F1, per-class AP, small-object AP) cover the full test split. The original FP/FN counts and the per-country rows covered only the first 12 test images (sorted by filename) and at most 8 images per country. Those original values are kept in `results/week6/final_test_results.csv` for transparency.
- **Protocol amendment `week6_full_test_counts_v1`:** FP/FN counts, the error analysis, and the per-country rows were re-computed on the **full** test split (50 images) using the **same** checkpoint, operating confidence, NMS IoU, and matching IoU. The model, thresholds, and selection were not changed and no retraining was done. Record: `results/week6/amendment_full_test_counts/full_test_counts.json`. The error analysis (Section 10) was re-run on the full split; the original 12-image outputs are archived in `results/week6/archive_smoke_12img/`.

## 1. Final research question

Starting from a small detector, which change best improves **accuracy per added compute** on RDD2022
(P2 head, lightweight attention, or 800 px input), and how does the selected model perform on the
**held-out test set** and across countries present in the fixed split?

## 2. Dataset and split

- Dataset: RDD2022 (China_Drone excluded) — **but the files actually evaluated were built from the synthetic fixture** (see Section 0)
- Classes: D00, D10, D20, D40
- Split: `grouped_70_15_15_seed6845` (grouped leakage-safe 70/15/15)
- Data config: `configs/rdd2022.yaml`
- **No new split was created in Week 6.**

## 3. Models evaluated

| Role | Model | Checkpoint (protocol) |
|---|---|---|
| Final selected | YOLO11n @ 640 (Week 5 selected / control) | `results/week5/yolo11n_640/seed_0/train/weights/best.pt` |
| Baseline | YOLOv8n Week 3 | `results/week3/yolov8n_baseline/train/weights/best.pt` |
| Baseline | YOLO11n Week 4 | `results/week4/yolo11n_baseline/train/weights/best.pt` |
| Best Week 5 ablation | Best Week 5 ablation (= YOLO11n@640 control) | identical to final |

Selection source: `results/week5/model_selection_decision.md` (validation only).

## 4. Frozen evaluation protocol

See `docs/final_evaluation_protocol.md` and `configs/final_evaluation.yaml`.

| Item | Value |
|---|---|
| imgsz | 640 |
| NMS IoU | 0.7 |
| Operating conf (val-selected) | **0.25** |
| conf for mAP | 0.001 |
| Test evaluation completed | True |
| Test used for selection | **False** |

Confidence lock file: `results/week6/CONFIDENCE_THRESHOLD_LOCKED.json`.

## 5. Final test results

Final model (`final_selected`) — **Test** metrics from `final_test_results.csv`:

| Metric | Validation | Test |
|---|---|---|
| mAP@0.5 | 0.0000 | 0.0000 |
| mAP@0.5:0.95 | 0.0000 | 0.0000 |
| Precision | 0.0001 | 0.0000 |
| Recall | 0.0384 | 0.0000 |
| F1 | 0.0002 | 0.0000 |
| Small-object AP | 0.0000 | 0.0000 |
| FP count (op. conf), full split (amendment) | — | 219 |
| FN count (op. conf), full split (amendment) | — | 42 |
| TP count, full split (amendment) | — | 0 |
| Ground-truth objects, full split | — | 55 |
| FP count, original smoke run (12 images, kept for transparency) | — | 72 |
| FN count, original smoke run (12 images, kept for transparency) | — | 11 |

Raw: `results/week6/final_test_results.csv`.

## 6. Comparison with baselines

Validation and Test are **separate columns**.

| Model | Val mAP50 | Test mAP50 | Val small-AP | Test small-AP | Params | GFLOPs | Latency ms |
|---|---|---|---|---|---|---|---|
| YOLOv8n Week 3 baseline | 0.0001 | 0.0000 | 0.0001 | 0.0000 | 3138732 | 8.75 | 21.46 |
| YOLO11n Week 4 strong baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2605612 | 6.57 | 20.29 |
| Best Week 5 ablation (= YOLO11n@640 control) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2605612 | 6.57 | 21.83 |
| YOLO11n @ 640 (Week 5 selected / control) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2605612 | 6.57 | 21.83 |

## 7. Per-class performance (final model)

| Class | Val AP50 | Test AP50 |
|---|---|---|
| D00 | 0.0000 | 0.0000 |
| D10 | 0.0000 | 0.0000 |
| D20 | 0.0000 | 0.0000 |
| D40 | 0.0000 | 0.0000 |

## 8. Small-object performance

Small-object AP uses GT boxes with normalized area &lt; 0.02.

| Split | Small-object AP |
|---|---|
| Validation | 0.0000 |
| Test | 0.0000 |

## 9. Country-level robustness

**No new country-held-out split was created.** The existing grouped split’s test portion is
100% `United_States` (incidental composition). Validation contains `China_MotorBike` and `Norway`.
Country-held-out OOD protocol: **not available** without redesigning the split (explicitly avoided).

| Split label | Country | Images | Objects | Box recall@0.5 (proxy) | Country-held-out? |
|---|---|---|---|---|---|
| Test | United_States | 50 | 55 | 0.0000 | False |

Figure: `figures/week6/country_performance.png`.  
CSV: `results/week6/country_results.csv`.

**Count source:** the test rows above come from the protocol amendment (`results/week6/amendment_full_test_counts/full_test_country_results.csv`), which covers the full test split (50 images, 55 boxes). The original smoke-run country rows (8-image cap) are archived in `results/week6/country_results.csv`. Baseline models are not shown in the country table.

## 10. Error analysis

Summary from `error_analysis_summary.json`:

- Split analyzed: Test
- Images: 50
- Error rows: 274
- Multi-damage images: 12
- Pothole FNs: 1
- Lighting/weather tags: **Pending** (not in RDD2022 metadata)

Figures: `figures/week6/false_positives.png`, `false_negatives.png`, `missed_small_objects.png`,
`localization_errors.png`, `confusion_matrix.png`, `precision_recall_curves.png`.  
Cases: `results/week6/error_cases.csv`.

## 11. Efficiency analysis

From `efficiency_benchmark.csv` (actual device recorded; claim T4 only if `is_t4=true`):

| Model | Params | GFLOPs | File MB | Latency mean/p50/p95 | Img/s | Peak GPU MB | Train sec |
|---|---|---|---|---|---|---|---|
| YOLO11n @ 640 (Week 5 selected / control) | 2605612 | 6.57 | 5.25 | 21.83 / 21.69 / 26.37 | 45.81 | Pending | 30.4 |
| YOLOv8n Week 3 baseline | 3138732 | 8.75 | 6.20 | 21.46 / 21.26 / 26.52 | 46.59 | Pending | 66.1 |
| YOLO11n Week 4 strong baseline | 2605612 | 6.57 | 5.25 | 20.29 / 20.38 / 21.95 | 49.28 | Pending | 83.7 |

Figure: `figures/week6/accuracy_efficiency_tradeoff.png` (Test mAP vs GFLOPs).

## 12. Main findings

1. Week 5 selected **YOLO11n@640** as the final candidate using **validation only**; no ablation
   beat the control on available val mAP (smoke caveat documented in Week 5).
2. Operating confidence **0.25** was locked from a **validation** sweep before test evaluation.
3. Held-out **Test** metrics for the final model: mAP50=0.0000,
   mAP50-95=0.0000 (from saved files).
4. vs YOLOv8n Test mAP50=0.0000; vs Week-4 YOLO11n Test mAP50=0.0000.
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
