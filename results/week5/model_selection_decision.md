# Week 5 — Model selection decision (validation only)

_Generated: 2026-11-26T03:03:58.644057+00:00_

## Selection rule

- Primary criterion: **highest mean validation mAP@0.5** across seeds 0/1/2 (or available seeds).
- Secondary (ties / near-ties): small-object AP@0.5, then per-class AP on D00/D10/D40, then lower GFLOPs / latency.
- Checkpoints were selected during training using **validation mAP@0.5 only**.
- **No test-set metrics were inspected or used for this decision.**

## Selected final candidate

**yolo11n_640** — YOLO11n baseline

## Why it was selected

- Highest available mean **validation mAP@0.5** among completed ablations: **0.0000**.
- Baseline (yolo11n_640) mean val mAP@0.5: **0.0000**.
- Improvement vs baseline: **-0.0000** (negative means the candidate is worse — reported honestly).
- Architecture / factors: `stock YOLO11n (P3/P4/P5)` / `['baseline']`.

- No ablation beat the YOLO11n@640 control on mean val mAP@0.5 (or only the control has completed runs). The baseline remains the candidate.

## Validation metrics supporting the decision

| Metric | Value |
|---|---|
| map50_mean | 0.0000 |
| map50_std | 0.0000 |
| map50_best / worst | 0.0000 / 0.0000 |
| map50_95_mean | 0.0000 |
| small_object_ap_mean | 0.0000 |
| ap_D00 / D10 / D20 / D40 mean | 0.0000 / 0.0000 / 0.0000 / 0.0000 |
| n_seeds_completed | 1 |

## Consistency across seeds

- only one seed completed — treat as provisional

- Per-seed val mAP@0.5: seed 0=0.0000

## Computational cost added

- params: 2.61e+06 (Δ +0, +0.0% vs baseline)

- GFLOPs: 6.57 (Δ +0, +0.0% vs baseline)

- latency_ms: 83.3 (Δ +0, +0.0% vs baseline)

## Why no test results were used

Per the project protocol and Week 5 objective, the test split is locked until Week 6. All checkpoint selection used validation mAP@0.5 only; this decision file was written from `seed_summary.csv` / `ablation_results.csv` validation aggregates exclusively. `evaluate_week5_ablation.py` refuses `--split test` without `--allow-test`.

## Ranking (mean val mAP@0.5)

| Rank | Experiment | map50_mean | Δ vs baseline |
|---|---|---|---|

| 1 | yolo11n_640 | 0.0000 | -0.0000 |

| 2 | yolo11n_p2_640 | 0.0000 | -0.0000 |

| 3 | yolo11n_attention_640 | 0.0000 | -0.0000 |

| 4 | yolo11n_800 | 0.0000 | -0.0000 |

| 5 | combined_800 | 0.0000 | -0.0000 |


## Caveat

Some or all runs are **smoke** (short epoch budget). Treat the selected candidate as **provisional** until full 100-epoch × seeds 0/1/2 results replace Pending / smoke rows.


## Honesty note (near-zero validation mAP)

Completed smoke runs currently show ~0 validation mAP@0.5 (fixture / 1–2 epoch budget). This does **not** support claiming that P2, attention, or 800 px improves accuracy. The control `yolo11n_640` is retained as the provisional Week 6 candidate until full multi-seed validation results are available. Seeds 1 and 2 remain **Pending**.


## Honesty note (near-zero validation mAP)

Completed smoke runs currently show ~0 validation mAP@0.5 (fixture / 1–2 epoch budget). This does **not** support claiming that P2, attention, or 800 px improves accuracy. The control `yolo11n_640` is retained as the provisional Week 6 candidate until full multi-seed validation results are available. Seeds 1 and 2 remain **Pending**.
