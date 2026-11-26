# Week 5 — Ablation report

Project: Road Damage Detection Using Lightweight Object Detectors  
Dataset: RDD2022 (grouped leakage-safe 70/15/15, `split_id=grouped_70_15_15_seed6845`)  
Classes: D00, D10, D20, D40

## 1. Week 5 objective

Evaluate whether lightweight architectural and input-resolution changes improve road-damage
detection—especially for small cracks and potholes—without an unreasonable computational increase.
Checkpoint selection uses **validation mAP@0.5 only**. The test set is **not** used for
selection or tuning (deferred to Week 6).

## 2. Ablation research questions

1. Does a **P2 (stride-4) detection head** improve thin crack / small damage AP at 640 px?
2. Does a **lightweight neck attention** block (C2PSA on P3) help without large cost?
3. Does raising input resolution to **800 px** help more than architectural changes alone?
4. Does the **combined** model (P2 + attention + 800) justify its extra compute?
5. Which candidate should advance to Week 6 final evaluation **based on validation only**?

## 3. Description of each model variant

| Experiment | Variant | Design |
|---|---|---|
| `yolo11n_640` | YOLO11n @ 640 | Stock YOLO11n (P3/P4/P5). Control. |
| `yolo11n_p2_640` | + P2 head @ 640 | Same backbone; Detect on P2/P3/P4/P5 (`configs/models/yolo11n-p2.yaml`). |
| `yolo11n_attention_640` | + attention @ 640 | C2PSA on P3 neck feature (`configs/models/yolo11n-attn.yaml`). |
| `yolo11n_800` | YOLO11n @ 800 | Stock YOLO11n; imgsz=800; batch 8 (documented memory policy). |
| `combined_800` | P2 + attn @ 800 | `yolo11n-p2-attn.yaml` + imgsz=800; batch 8. |

Attention uses Ultralytics built-in **C2PSA** (no custom CUDA ops). P2 follows the YOLO11 +
yolov8-p2 head pattern.

## 4. Controlled variables

- Identical dataset split (`configs/rdd2022.yaml` → Week 3 materialized dirs)
- Identical class mapping (D00=0 … D40=3)
- Same optimizer / LR schedule family (`optimizer: auto`, `lr0=0.01`)
- Same augmentation recipe
- Same early-stopping patience family (reduced only under `--smoke`)
- AMP when CUDA is available
- Seeds **0, 1, 2** for principal comparisons (smoke may use seed 0 only)
- Best checkpoint = Ultralytics best on **validation** fitness / mAP@0.5

Documented adjustment: batch **16→8** at 800 px (`training_defaults.yaml` `batch_800`).

## 5. Experimental settings

| Item | Value |
|---|---|
| Data config | `configs/rdd2022.yaml` |
| Split | grouped 70/15/15, seed 6845 |
| Epochs (full) | 100 (smoke: ≤2) |
| Batch @640 / @800 | 16 / 8 (smoke: ≤4 / ≤2) |
| Comparison seeds | 0, 1, 2 |
| Selection metric | val mAP@0.5 |
| Test for selection | **No** |

Configs: `configs/week5_*.yaml`. Train: `scripts/train_week5_ablation.py`.

## 6. Results table

Validation metrics (mean over completed seeds). Unfinished → **Pending**.

| Experiment | Variant | imgsz | mAP50 | mAP50-95 | P | R | F1 | small-AP | Δ mAP50 vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| yolo11n_640 | YOLO11n baseline | 640 | 0.0000 | 0.0000 | 0.0001 | 0.0384 | 0.0002 | 0.0000 | -0.0000 |
| yolo11n_p2_640 | YOLO11n + P2 head | 640 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| yolo11n_attention_640 | YOLO11n + neck attention (C2PSA@P3) | 640 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| yolo11n_800 | YOLO11n @ 800px | 800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0000 |
| combined_800 | Combined P2+attention@800 | 800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Pending | Pending | -0.0000 |

Raw rows: `results/week5/ablation_results.csv`, `results/week5/seed_summary.csv`.

## 7. Per-class comparison

| Experiment | AP D00 | AP D10 | AP D20 | AP D40 |
|---|---|---|---|---|
| yolo11n_640 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| yolo11n_p2_640 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| yolo11n_attention_640 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| yolo11n_800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| combined_800 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Figure: `figures/week5/per_class_ap_comparison.png`.

## 8. Small-object detection analysis

Small-object AP@0.5 uses GT boxes with normalized area &lt; 0.02 (see `evaluate_week5_ablation.py`).
Values are **Pending** until `--recompute-val` completes for each seed checkpoint.

Figure: `figures/week5/small_object_ap_comparison.png`.

Honest reading: if small-AP does not rise with P2 / 800 px, do not claim small-object gains.

## 9. Accuracy-versus-efficiency analysis

| Experiment | Params | GFLOPs | Latency (ms) | Peak GPU MB |
|---|---|---|---|---|
| yolo11n_640 | 2605612 | 6.57 | 83.33 | Pending |
| yolo11n_p2_640 | 2667216 | 10.46 | 104.67 | Pending |
| yolo11n_attention_640 | 2621964 | 10.72 | 129.28 | Pending |
| yolo11n_800 | 2605612 | 6.57 | 109.62 | Pending |
| combined_800 | 2683568 | 14.61 | 293.88 | Pending |

Figure: `figures/week5/accuracy_efficiency_tradeoff.png`.  
Source: `results/week5/efficiency_results.csv`.

## 10. Qualitative prediction examples

Validation overlays (not test) are written under `figures/week5/predictions/<experiment>/`
when seed-0 checkpoints exist. Absence of images means the corresponding train run is still Pending.

## 11. Limitations

- All Week 5 runs used the **synthetic fixture**, not the official RDD2022 download, and ran 1 epoch in smoke mode. Absolute mAP is ~0 and not meaningful; no relative trend in these runs supports a conclusion about the variants.
- Smoke runs understate convergence; do not overstate smoke improvements.
- Small-object AP is an approximation (area filter), not COCO-official small AP.
- Combined model confounds three factors; single-factor configs isolate effects.
- MPS/CPU runs may omit peak CUDA memory.

## 12. Candidate model for final evaluation

**yolo11n_640**

See `results/week5/model_selection_decision.md` for the locked rationale.

## 13. Justification (validation only)

Selection uses mean **validation** mAP@0.5 (and secondaries listed in the decision file).
Test metrics were not computed for selection. Week 6 will run the locked candidate (and
baselines) on the held-out test split once.

---

Reproduce:

```bash
./scripts/run_week5.sh --smoke   # quick
./scripts/run_week5.sh           # full seeds 0 1 2
```
