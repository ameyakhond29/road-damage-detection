# Week 3 — Dataset finalization and YOLOv8n baseline report

**Project:** Road Damage Detection Using Lightweight Object Detectors  
**Dataset:** RDD2022 (D00, D10, D20, D40)  
**Do not invent metrics.** Fields marked *Not yet available.* must be filled only from real run artifacts.

---

## 1. Dataset preparation summary

| Step | Script | Output |
|------|--------|--------|
| VOC → YOLO | `scripts/convert_voc_to_yolo.py` | `data/processed/RDD2022_yolo/{images,labels}/<country>/` |
| Validation | `scripts/validate_dataset.py` | `results/week3/dataset_validation.{json,csv}` |
| Grouped split | `scripts/create_week3_splits.py` | `data/splits/{train,val,test}.csv`, `split_summary.json` |
| Dataset YAML | (written by split script) | `configs/rdd2022.yaml` |
| Baseline train | `scripts/train_yolov8_baseline.py` | `results/week3/yolov8n_baseline/` |

- Original VOC XML and images are **never modified**.  
- **China_Drone** excluded (Week 1/2 protocol).  
- Unsupported labels (D50, Repair, …) dropped and counted.  
- Negative images kept as **empty** `.txt` label files.  
- Class mapping: D00→0, D10→1, D20→2, D40→3.

### Conversion issue report

See `results/week3/conversion_report.json` and `data/processed/RDD2022_yolo/conversion_issues.csv` after running conversion.

| Issue | Count |
|-------|-------|
| Malformed XML | *(from report)* |
| Missing images (orphan XML) | *(from report)* |
| Invalid boxes | *(from report)* |
| Unknown classes | *(from report)* |
| Duplicate annotations | *(from report)* |

---

## 2. Final class distribution

Populate from `results/week3/dataset_validation.json` → `class_counts` after validation on the **real** RDD2022 download:

| Class | ID | Box count |
|-------|----|-----------|
| D00 | 0 | Not yet available. |
| D10 | 1 | Not yet available. |
| D20 | 2 | Not yet available. |
| D40 | 3 | Not yet available. |

*(If only the synthetic fixture was run, label those counts as fixture-only and do not cite them as RDD2022.)*

---

## 3. Split methodology

- **Ratio:** 70% train / 15% val / 15% test  
- **Grouping:** blocks of **50** consecutive filenames within each country  
- **Leakage control:** perceptual hash (pHash) near-duplicates forced into the same group  
- **Seed:** **6845**  
- **Artifacts:**  
  - `data/splits/train.csv`  
  - `data/splits/val.csv`  
  - `data/splits/test.csv`  
  - `data/splits/split_summary.json`  

Materialized Ultralytics dirs: `data/processed/RDD2022_yolo/images/{train,val,test}` and matching `labels/`.

### Split summary (fill from `split_summary.json`)

| Split | Images | Boxes | % negatives |
|-------|--------|-------|-------------|
| train | Not yet available. | Not yet available. | Not yet available. |
| val | Not yet available. | Not yet available. | Not yet available. |
| test | Not yet available. | Not yet available. | Not yet available. |

Duplicate pairs crossing splits: must be **0** (`duplicate_leakage.passed: true`).

---

## 4. Data-quality checks

Validator checks:

- Every image has a label file  
- Each YOLO line has 5 values  
- Class IDs ∈ {0,1,2,3}  
- Coordinates normalized to [0, 1]  
- Width/height > 0  
- Boxes within image bounds  
- No corrupted images  
- Negatives preserved  

Primary artifacts: `results/week3/dataset_validation.json`, `results/week3/dataset_validation.csv`.

Overall validation status: Not yet available. *(or PASSED/FAILED from the JSON `passed` field)*

---

## 5. Exact baseline configuration

| Setting | Value |
|---------|--------|
| Model | YOLOv8n (`yolov8n.pt`, COCO-pretrained) |
| Image size | 640 |
| Epochs (default) | 50 (T4/Colab/Kaggle budget); `--smoke` → 3 |
| Batch | 16 (smoke: 4) |
| Seed | 6845 |
| LR0 | 0.01 |
| Optimizer | Ultralytics `auto` |
| AMP | On when CUDA available |
| Patience | 20 |
| Data config | `configs/rdd2022.yaml` |
| Selection | Best **validation** checkpoint only |
| Test during train | **No** |

Run card: `results/week3/yolov8n_baseline/run_config.json`.

---

## 6. Training command

```bash
# Full Week 3 pipeline (auto-falls back to fixture if raw data missing)
./scripts/run_week3.sh

# Smoke / no-GPU friendly
./scripts/run_week3.sh --smoke

# Or step-by-step on real RDD2022:
python scripts/convert_voc_to_yolo.py --raw-dir data/raw/RDD2022 --seed 6845
python scripts/validate_dataset.py --seed 6845
python scripts/create_week3_splits.py --seed 6845
python scripts/train_yolov8_baseline.py --data configs/rdd2022.yaml --seed 6845
```

---

## 7. Validation results

Populate from `results/week3/yolov8n_baseline/metrics.json` after training:

| Metric | Value |
|--------|--------|
| mAP@0.5 | Not yet available. |
| mAP@0.5:0.95 | Not yet available. |
| Precision | Not yet available. |
| Recall | Not yet available. |
| Training time (s) | Not yet available. |
| GPU name | Not yet available. |
| Best checkpoint | Not yet available. |

Plots (when generated):

- `results/week3/yolov8n_baseline/curves_results.png`  
- `results/week3/yolov8n_baseline/curves_precision_recall.png`  
- `results/week3/yolov8n_baseline/confusion_matrix.png`  
- `results/week3/yolov8n_baseline/val_batch0_pred.jpg`  

---

## 8. Known limitations

- Full RDD2022 must be downloaded for official numbers; fixture runs are pipeline checks only.  
- Default 50 epochs is a **progress-report budget**, not the final Week 4 multi-seed matrix (protocol uses seeds 0/1/2 and often 100 epochs).  
- P2 / attention variants remain Week 3+ architecture work (placeholders under `configs/models/`).  
- Paths with spaces require absolute image lists or materialized dirs (handled by Week 3 layout).  
- Test and held-out country eval are **not** used for model selection.

---

## 9. Pipeline verification status (this repository)

A **synthetic fixture** smoke run (`./scripts/run_week3.sh --smoke` with `RAW_DIR=data/fixture/RDD2022`) confirmed:

- Conversion + issue reporting  
- `VALIDATION PASSED`  
- Split leakage check `passed: true` after pHash consistency enforcement  
- YOLOv8n smoke train (3 epochs, seed 6845) wrote `results/week3/yolov8n_baseline/metrics.json`  

Those fixture smoke metrics are **not** official RDD2022 results. After downloading real RDD2022 to `data/raw/RDD2022`, re-run without the fixture and replace every “Not yet available.” field above from the new artifacts.

## 10. Evidence checklist for the progress report

- [x] Conversion / validation / split / train scripts exist and smoke-tested on fixture  
- [ ] Validation JSON shows `passed: true` on **real** RDD2022  
- [ ] Split summary + zero cross-split duplicates on real data  
- [x] `configs/rdd2022.yaml` lists train/val/test dirs, nc=4, imgsz, split_id  
- [ ] Baseline `metrics.json` + curves on real data (50-epoch default)  
- [ ] This report updated with real numbers (no estimates)  
