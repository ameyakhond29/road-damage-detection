# Week 4 Progress Report — Road Damage Detection

**Project:** Road Damage Detection Using Lightweight Object Detectors  
**Course:** DATA-255  
**Date:** 2026-11-05  

> **Data caveat:** Official RDD2022 was not present under `data/raw/` when this report was generated. Numbers below come from a **synthetic-fixture smoke run** (3 epochs, batch 4, seed 6845) used only to verify the Week 4 pipeline. They are **not** publishable RDD2022 results. After downloading the real dataset and re-running `./scripts/run_week4.sh` (without relying on the fixture), replace every smoke-tagged cell with values from the new JSON/CSV artifacts. Where a full real run has not been executed, fields remain **Pending**.

---

## 1. Project objective

Build a reproducible lightweight object-detection pipeline for multi-country road damage (RDD2022), then measure which controlled change to a small YOLO detector best improves accuracy per added compute—and whether gains survive country shift.

## 2. Research question

Starting from a small detector, which single change provides the best accuracy per added computational cost on RDD2022: a **P2 stride-4 detection head**, a **lightweight neck-attention block**, or a **larger input size**? Do these improvements remain effective on countries not seen during training?

(See `docs/research_question.md`.)

## 3. Dataset and class definitions

| Class | ID | Meaning |
|-------|----|---------|
| D00 | 0 | Longitudinal crack |
| D10 | 1 | Transverse crack |
| D20 | 2 | Alligator crack |
| D40 | 3 | Pothole |

China_Drone excluded. Unsupported labels (D50, Repair, …) dropped. Negatives kept as empty label files.

## 4. Preprocessing and split methodology

- VOC → YOLO conversion (`scripts/convert_voc_to_yolo.py`)  
- Validation (`scripts/validate_dataset.py`)  
- **Fixed** grouped 70/15/15 split: blocks of 50 filenames within country + pHash duplicate merge; seed **6845**  
- Artifacts: `data/splits/{train,val,test}.csv`, `split_summary.json`, `configs/rdd2022.yaml`  
- **Week 4 did not change the split.**

Fixture smoke split sizes (from `data/splits/split_summary.json`): train 252 / val 59 / test 50 images; duplicate cross-split pairs = 0.

## 5. YOLOv8n reference baseline (Week 3)

| Setting | Value (smoke artifact) |
|---------|-------------------------|
| Model | YOLOv8n (`yolov8n.pt`) |
| imgsz / seed | 640 / 6845 |
| Epochs / batch | 3 / 4 (smoke) |
| Selection | Best **validation** checkpoint only |
| Checkpoint | `results/week3/yolov8n_baseline/train/weights/best.pt` |
| Source | `results/week3/yolov8n_baseline/metrics.json` |

## 6. YOLO11n strong baseline (Week 4)

| Setting | Value (smoke artifact) |
|---------|-------------------------|
| Model | YOLO11n (`yolo11n.pt`) |
| Same data YAML / split / seed / imgsz | Yes (`configs/rdd2022.yaml`) |
| Epochs / batch | 3 / 4 (smoke); default script budget 50 / 16 |
| AMP | On when CUDA available (off on MPS/CPU) |
| Selection | Validation only; test never used for selection |
| Checkpoint | `results/week4/yolo11n_baseline/train/weights/best.pt` |
| Source | `results/week4/yolo11n_baseline/metrics.json` |

## 7. Experimental settings (held constant)

Same Week-3 split files, class mapping, imgsz=640, lr0=0.01, Ultralytics `optimizer=auto`, seed 6845, val-only checkpoint selection, identical evaluation script for both models.

## 8. Results table

Values loaded from `results/week4/baseline_comparison.csv` / `.json` (validation metrics of selected checkpoints).

| Metric | YOLOv8n (smoke) | YOLO11n (smoke) |
|--------|-----------------|-----------------|
| mAP@0.5 | 8.73e-05 | 0.0 |
| mAP@0.5:0.95 | 2.62e-05 | 0.0 |
| Precision | 1.43e-04 | 0.0 |
| Recall | 0.0227 | 0.0 |
| F1 | 0.000285 | Pending (P=R=0) |
| Parameters | 3,138,732 | 2,605,612 |
| GFLOPs | Pending | Pending |
| Latency (ms, batch 1) | 17.745 | 19.238 |
| Checkpoint size (MB) | 6.202 | 5.249 |
| Training time (s) | 66.1 | 83.71 |
| Smoke flag | true | true |

**Conclusion (from saved comparison JSON):** Under this smoke run, YOLOv8n mAP@0.5 is higher than YOLO11n. **Do not claim YOLO11n is better.** Re-evaluate after a full real-RDD2022 train.

Figure: `figures/week4/baseline_comparison.png`.

### Final test metrics (locked eval; not used for selection)

From `results/week4/evaluation/evaluation_summary.json`:

| Model | Test mAP@0.5 | Test Precision | Test Recall |
|-------|--------------|----------------|-------------|
| YOLOv8n | 8.35e-06 | 4.05e-05 | 0.0294 |
| YOLO11n | 1.13e-06 | 1.67e-05 | 0.0156 |

## 9. Per-class analysis (validation)

From evaluation JSON (smoke):

| Class | YOLOv8n AP50 (val) | YOLO11n AP50 (val) |
|-------|--------------------|--------------------|
| D00 | 0.0 | 0.0 |
| D10 | 3.49e-04 | 0.0 |
| D20 | 0.0 | 0.0 |
| D40 | 0.0 | 0.0 |

On real RDD2022 after longer training, replace this table from the new `evaluation.json` files.

## 10. Small-object detection observations

Definition: GT boxes with normalized area \(w \times h < 0.02\).

| Split | Model | Small GT boxes | Small recall@IoU0.5 |
|-------|-------|----------------|---------------------|
| test | YOLOv8n | 47 | 0.0 |
| test | YOLO11n | 47 | 0.0 |

Smoke models did not recover small cracks/potholes. This motivates Week 5 P2 / 800px ablations on real data.

## 11. Qualitative prediction examples

12 examples per model under:

- `figures/week4/predictions/yolov8n/`  
- `figures/week4/predictions/yolo11n/`  

Tags include `fn_candidate`, `fp_candidate`, and `small`. Confusion matrices and PR curves: `figures/week4/*_{val,test}_*.png`. Confidence sweeps: `figures/week4/*_confidence_analysis.png`.

## 12. Limitations and risks

1. **Fixture / smoke only** — metrics are pipeline evidence, not RDD2022 science results.  
2. **Short training** (3 epochs) — rankings can reverse after 50–100 epochs.  
3. **Test set geography (fixture):** entire test split is `United_States` in the current fixture summary; val is mostly Norway — interpret country numbers carefully.  
4. **Dedicated OOD retrain track** (train JP/IN/CZ/CN_M → eval NO/US) was **not** re-run as a new split; Week 4 uses the fixed Week-3 split plus optional in-split country stats. `data/ood/` remains available for a separate held-out protocol without changing the main split.  
5. **No CUDA on the verification host** — AMP/latency/memory figures will differ on T4/4090.  
6. GFLOPs sometimes **Pending** depending on Ultralytics `model.info()` return shape.

## 13. Plan for Week 5 ablations

Priority order (same split, val-only selection, seeds 0/1/2 when budget allows):

1. YOLO11n + **P2** head vs strong baseline (H1).  
2. YOLO11n + **neck attention** vs strong baseline.  
3. YOLO11n @ **800 px** vs 640.  
4. Combined P2+attention (declared multi-factor).  
5. Country-held-out evaluation using `data/ood/` (document clearly; do not silently replace the main split).  
6. Accuracy-per-GFLOPs / latency tables and paired multi-seed summaries.

---

## Reproduce Week 4

```bash
cd road-damage-detection
source .venv/bin/activate

# Requires Week 3 outputs
./scripts/run_week4.sh            # full (50-epoch default for YOLO11n)
./scripts/run_week4.sh --smoke    # 3-epoch pipeline check

# Manual:
python scripts/train_yolo11_baseline.py --data configs/rdd2022.yaml --seed 6845
python scripts/compare_baselines.py --seed 6845
python scripts/evaluate_week4.py --seed 6845
python scripts/write_week4_manifest.py
```

Primary artifacts: `results/week4/`, `figures/week4/`, `results/week4/run_manifest.json`.
