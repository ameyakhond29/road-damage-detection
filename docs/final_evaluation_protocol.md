# Final evaluation protocol (frozen)

**Protocol ID:** `week6_final_v1`  
**Config:** `configs/final_evaluation.yaml`  
**Status:** Frozen before held-out test use. Do not change the model, confidence threshold, IoU, or metrics definition after test results are written.

## Selected model checkpoint

| Field | Value |
|---|---|
| Experiment | `yolo11n_640` (Week 5 decision) |
| Display name | YOLO11n @ 640 (Week 5 selected / control) |
| Checkpoint | `results/week5/yolo11n_640/seed_0/train/weights/best.pt` |
| Architecture | Stock YOLO11n — Detect(P3, P4, P5) |
| Input resolution | **640** |
| Seed (available) | **0** (seeds 1/2 Pending) |
| Selection metric | Validation mAP@0.5 (Week 5); test **not** used |
| Decision file | `results/week5/model_selection_decision.md` |

**Caveat:** Week 5 runs were smoke / near-zero val mAP; the control is retained provisionally because no ablation improved validation metrics. This is documented honestly — not claimed as a strong accuracy win.

## Baselines (same frozen protocol)

| Model | Checkpoint |
|---|---|
| YOLOv8n (Week 3) | `results/week3/yolov8n_baseline/train/weights/best.pt` |
| YOLO11n (Week 4) | `results/week4/yolo11n_baseline/train/weights/best.pt` |
| Best Week 5 ablation | Same as final (`yolo11n_640`) — identical checkpoint |

## Dataset split

- Config: `configs/rdd2022.yaml`
- Split ID: `grouped_70_15_15_seed6845`
- Files: `data/splits/{train,val,test}.csv`
- Class mapping: D00=0, D10=1, D20=2, D40=3
- **No new split** in Week 6.

## Thresholds

| Parameter | Value | Selected how |
|---|---|---|
| NMS IoU | **0.7** | Fixed across Weeks 2–5 |
| conf for mAP | **0.001** | COCO-style / Ultralytics default for mAP |
| Operating confidence | Locked in `results/week6/CONFIDENCE_THRESHOLD_LOCKED.json` | **Validation sweep only** (F1, then precision; prefer 0.25 on ties) |

**Rule:** Do not change the operating confidence after any test metric is inspected.

## Evaluation metrics

mAP@0.5, mAP@0.5:0.95, precision, recall, F1, per-class AP / P / R, small-object AP (area &lt; 0.02), detection / FP / FN counts, latency, GFLOPs, parameters, peak GPU memory, training time.

Every published number must be labeled **Validation**, **Test**, or **Country analysis (not country-held-out OOD)**.

## Country analysis

The main test split is 100% `United_States` under the existing grouped split. That is **incidental**, not a designed country-held-out protocol. Validation includes `China_MotorBike` and `Norway`. We report per-country counts and proxy recall **without creating a new split**. Formal country-held-out OOD remains out of scope unless separately designed later.

## Hardware / software

- **Target:** NVIDIA T4  
- **Actual:** recorded at run time in `final_run_manifest.json` / efficiency CSV (Mac MPS/CPU if no T4)  
- Software versions: recorded in the run manifest (`torch`, `ultralytics`, Python)

## Test evaluation completion

Tracked by:

- `configs/final_evaluation.yaml` → `test_evaluation_completed`
- `results/week6/TEST_EVALUATION_COMPLETED.json`

Held-out test evaluation is run **exactly once** per selected model checkpoint after freeze. Re-runs require `--force-rerun` and must be documented as a protocol breach — prefer documenting bugs instead of silent retuning.

## Seeds

Principal comparison seeds: 0, 1, 2. Where checkpoints are missing, mark **Pending**. Week 3/4 baselines used seed 6845.
