# Final results summary

_Values loaded from packaged / Week 6 saved files only. No invented metrics._

> **Provenance warning:** all values are from the synthetic fixture, not the official RDD2022 download. The original final test run used smoke mode. Its FP/FN counts and error analysis were re-computed on the full 50-image test split under protocol amendment `week6_full_test_counts_v1` (counts only; model, thresholds, and selection unchanged). See `docs/known_issues_and_resolutions.md`.

## 1. Best model

**YOLO11n @ 640 (Week 5 selected / control)**  
Checkpoint: `results/week5/yolo11n_640/seed_0/train/weights/best.pt`  
Selected on **validation** (Week 5). Operating conf **0.25** locked on validation.

## 2. Baseline comparison (Test mAP@0.5)

| Model | Val mAP50 | Test mAP50 |
|---|---|---|
| YOLOv8n | 0.0001 | 0.0000 |
| YOLO11n Week 4 | 0.0000 | 0.0000 |
| Final selected | 0.0000 | 0.0000 |

## 3. Main performance metrics (final model)

| Metric | Validation | Test |
|---|---|---|
| mAP@0.5 | 0.0000 | 0.0000 |
| mAP@0.5:0.95 | 0.0000 | 0.0000 |
| Precision | 0.0001 | 0.0000 |
| Recall | 0.0384 | 0.0000 |
| F1 | 0.0002 | 0.0000 |

## 4. Per-class AP@0.5 (final)

| Class | Val | Test |
|---|---|---|
| D00 | 0.0000 | 0.0000 |
| D10 | 0.0000 | 0.0000 |
| D20 | 0.0000 | 0.0000 |
| D40 | 0.0000 | 0.0000 |

## 5. Small-object AP

| Split | Value |
|---|---|
| Validation | 0.0000 |
| Test | 0.0000 |

## 6. Computational trade-offs (final)

| Item | Value |
|---|---|
| Parameters | 2605612 |
| GFLOPs | 6.57 |
| Latency ms (mean) | 21.83 |
| Peak GPU MB | Pending |
| Training time sec | 30.4 |

Week 5 ablations increased GFLOPs (P2 ~10.5, combined ~14.6) without validated mAP gains under smoke budgets.

## 7. Country-level robustness

| Split | Country | Images | Objects | Box recall@0.5 |
|---|---|---|---|---|
| Validation | China_MotorBike | 8 | 8 | 0.0000 |
| Validation | Norway | 8 | 7 | 0.0000 |
| Test | United_States | 8 | 11 | 0.0000 |

Test set is United States–only under the fixed split (not a designed OOD protocol).

## 8. Main failure cases

From `error_analysis_summary.json`: images=50, error rows=274, pothole FNs=1, multi-damage images=12.  
Figures: `figures/final/false_positives.png`, `false_negatives.png`, `missed_small_objects.png`.

## 9. Project limitations

- Smoke training → near-zero absolute mAP; do not overclaim.  
- Multi-seed incomplete (1/2 Pending).  
- No T4 numbers in local efficiency run.  
- Thin cracks remain hard (small-object AP).

## 10. Final conclusion

Under the frozen Week 6 protocol, the team advances **YOLO11n@640** as the selected lightweight detector. Ablations did not beat the control on available **validation** metrics; held-out **test** metrics are reported separately and were not used for selection.
