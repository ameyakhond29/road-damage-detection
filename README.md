# Road Damage Detection Using Lightweight Object Detectors

DATA-255 deep-learning computer-vision project.

Detect road damage (cracks and potholes) in multi-country **RDD2022** imagery using lightweight YOLO detectors (YOLOv8n / YOLO11n and Week 5 ablations), with a **validation-only** selection protocol and a **frozen** held-out test evaluation.

## Research question

Starting from a small detector, which single change best improves **accuracy per added computational cost** on RDD2022: a **P2 stride-4 detection head**, a **lightweight neck-attention block**, or a **larger input size (800 px)**? Does a combined model justify its cost?

Protocol: `docs/research_question.md`, `docs/week2_experiment_protocol.md`, `docs/final_evaluation_protocol.md`.

## Status and data provenance (read before citing any number)

- **Active pipeline = official RDD2022** (<https://github.com/sekilab/RoadDamageDetector>). Every script defaults to the official data (`configs/rdd2022.yaml`, `data/processed/`, `data/splits/`) and writes to `results/official_rdd2022/`. Status: `docs/official_dataset_replacement_status.md`.
- **Official-data model results: pending.** The only model run on official data so far is a 2-epoch pipeline smoke test (`results/official_rdd2022/smoke_test/`) — it verifies the pipeline and is **not** model performance.
- **LEGACY fixture artifacts:** `results/week3`–`results/week7`, `results/final/`, `tables/final/`, `figures/final/` and the Week 1–7 reports were produced on a **synthetic fixture** (`data/fixture/RDD2022/`). Their metrics are near zero and are not evidence about road-damage detection. They are kept for traceability only (`results/LEGACY_FIXTURE_RESULTS.md`).
- The fixture is used only for unit tests and debugging (`--dataset fixture`); its results go to `results/fixture_runs/` and can never be written into `results/official_rdd2022/`.

## Dataset and link

- **Official source:** https://github.com/sekilab/RoadDamageDetector (RDD2022, CRDDC2022 release)
- Archive used: Figshare mirror https://doi.org/10.6084/m9.figshare.21431547 (`RDD2022_released_through_CRDDC2022.zip`, 13.3 GB; the S3 link in the source README returned HTTP 403)

**Do not commit the raw dataset, the ZIP, or processed images** (all gitignored). See `data/README.md` and `docs/dataset_card.md`.

| Stage | Count |
|---|---|
| Labeled train images (all countries) | 38,385 (China_Drone 2,401 excluded) |
| Modeling images (converted) | 35,984 (14,136 negatives kept) |
| Boxes D00 / D10 / D20 / D40 | 24,590 / 10,567 / 10,323 / 6,458 (51,938) |
| Split train / val / test | 25,355 / 5,163 / 5,018 (448 near-duplicates purged from val/test) |
| Official unlabeled test images | 9,035 — never used |

### Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

`China_Drone` is excluded. All other labels (D01, D11, D43, D44, D50, Repair) are dropped and counted. Fixed split: `grouped_70_15_15_seed6845` in `data/splits/` (`dataset_hash a9bad058971ade44`).

## Repository structure

```text
road-damage-detection/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/                 # Dataset + experiment + frozen final protocol
├── data/
│   ├── README.md
│   ├── raw/RDD2022/         # gitignored official download
│   ├── processed/           # gitignored: images/ (hardlinks) + labels/ (YOLO)
│   ├── splits/              # Tracked official split CSVs (+ legacy_fixture/)
│   └── fixture/             # synthetic fixture — unit tests / debugging only
├── scripts/                 # Week runners + setup / reproduce
├── src/                     # Dataset, training, models, evaluation
├── results/                 # official_rdd2022/ (gitignored) · week3–week7 + final/ = LEGACY fixture
├── figures/                 # week* + final/
├── tables/final/            # Packaged CSV tables
├── docs/                    # Reports, cards, checklists
├── presentation/            # Slide outline
└── tests/
```

## Environment setup

**Local**

```bash
cd road-damage-detection
bash scripts/setup_environment.sh
# or: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python scripts/check_environment.py
```

`configs/rdd2022.yaml` is portable (no machine paths). The Ultralytics data YAML with absolute paths is generated per machine at `data/processed/ultralytics/rdd2022_official.yaml` by the training scripts.

Install a CUDA PyTorch build matching the runtime if needed: https://pytorch.org/get-started/locally/

Optional conda: `conda env create -f environment.yml && conda activate rdd-yolo`

## Dataset preparation (official RDD2022)

One command (download → audit → convert → validate → split → statistics → smoke):

```bash
bash scripts/run_official_dataset_pipeline.sh
# existing extraction elsewhere:  RDD2022_ROOT=/path/to/RDD2022 bash scripts/run_official_dataset_pipeline.sh
```

Step by step:

```bash
python scripts/download_rdd2022.py                       # ~13.3 GB; resumable; extracts to data/raw/RDD2022
python scripts/01_audit_dataset.py --data-root data/raw/RDD2022
python scripts/convert_voc_to_yolo.py --data-root data/raw/RDD2022
python scripts/validate_dataset.py --config configs/rdd2022.yaml
python scripts/create_week3_splits.py --config configs/rdd2022.yaml --seed 6845
python scripts/validate_dataset.py --config configs/rdd2022.yaml --require-splits
python scripts/smoke_test_official_rdd2022.py --config configs/rdd2022.yaml   # pipeline check only
```

Data root resolution: `--data-root` > `$RDD2022_ROOT` > `data/raw/RDD2022`; the root is searched recursively for the country folders, and a missing dataset stops with download instructions. Outputs: `results/official_rdd2022/` (audit, conversion manifest, unsupported labels, validation report, split leakage report, smoke test).

Fixture (unit tests / debugging only): `python scripts/00_create_fixture.py`, then pass `--dataset fixture`.

## Training instructions (official data by default)

```bash
# Week 3 — YOLOv8n baseline  -> results/official_rdd2022/week3/yolov8n_baseline/
python scripts/train_yolov8_baseline.py --epochs 50 --imgsz 640 --batch 16 --seed 0 --device 0

# Week 4 — YOLO11n baseline  -> results/official_rdd2022/week4/yolo11n_baseline/
python scripts/train_yolo11_baseline.py --epochs 50 --imgsz 640 --batch 16 --seed 0 --device 0

# Week 5 — one ablation run  -> results/official_rdd2022/week5/<experiment>/seed_<seed>/
python scripts/train_week5_ablation.py --experiment yolo11n_640 --seed 0 --device 0
```

Shared flags: `--dataset {official,fixture}`, `--data-root`, `--config`, `--epochs`, `--imgsz`, `--batch`, `--seed`, `--device`, `--output-dir`. Each run records the dataset source, version, `dataset_hash` and per-split hashes in its run card. Runs refuse to write into the legacy `results/week*` folders, and fixture runs cannot write into `results/official_rdd2022/`.

The `run_week3.sh`–`run_week6.sh` wrappers reproduce the **legacy fixture** protocol and exit unless `ALLOW_LEGACY_FIXTURE=1` is set.

Checkpoint selection uses **validation mAP@0.5 only**. Never tune on test.

## Colab / Kaggle T4 (official data)

Colab (GPU runtime, ~100 GB disk):

```bash
!git clone <your-repo-url> road-damage-detection && cd road-damage-detection && pip install -q -r requirements.txt
!cd road-damage-detection && python scripts/download_rdd2022.py --output-dir /content/rdd_zip --extract-dir /content/RDD2022 --train-only && rm -f /content/rdd_zip/*.zip
!cd road-damage-detection && RDD2022_ROOT=/content/RDD2022 SKIP_SMOKE=1 bash scripts/run_official_dataset_pipeline.sh
!cd road-damage-detection && python scripts/smoke_test_official_rdd2022.py --config configs/rdd2022.yaml --device 0
!cd road-damage-detection && python scripts/train_yolo11_baseline.py --epochs 50 --imgsz 640 --batch 32 --seed 0 --device 0
```

Kaggle (GPU T4, Internet on). `/kaggle/working` is capped at ~20 GB, so keep the archive and raw data in `/tmp` (check with `df -h /tmp`):

```bash
!git clone <your-repo-url> /kaggle/working/road-damage-detection
%cd /kaggle/working/road-damage-detection
!pip install -q -r requirements.txt
!python scripts/download_rdd2022.py --output-dir /tmp/rdd_zip --extract-dir /tmp/RDD2022 --train-only && rm -f /tmp/rdd_zip/*.zip
!RDD2022_ROOT=/tmp/RDD2022 SKIP_SMOKE=1 bash scripts/run_official_dataset_pipeline.sh
!python scripts/smoke_test_official_rdd2022.py --config configs/rdd2022.yaml --device 0
!python scripts/train_yolo11_baseline.py --epochs 50 --imgsz 640 --batch 32 --seed 0 --device 0
```

If the raw data and the repo are on different filesystems, processed images fall back from hardlinks to symlinks automatically. Copy `results/official_rdd2022/` (and `best.pt`) out before the session ends.

## Evaluation instructions

```bash
# Week 6 — freeze conf on val, then one-shot test (refuses silent re-run)
./scripts/run_week6.sh --smoke
# or piecewise:
python scripts/run_final_evaluation.py
python scripts/error_analysis.py
python scripts/final_efficiency_benchmark.py
python scripts/write_week6_tables.py
```

Protocol: `configs/final_evaluation.yaml`, `docs/final_evaluation_protocol.md`.

## Reproduction instructions (LEGACY fixture protocol)

> Everything from here to "Example predictions" describes the Weeks 3–7 **synthetic-fixture** results. They are not official-data results.

Primary model: **YOLO11n @ 640** (`results/week5/yolo11n_640/seed_0/train/weights/best.pt`).

```bash
bash scripts/setup_environment.sh
python scripts/check_environment.py
python scripts/package_final_results.py
python scripts/reproduce_main_result.py
# full suite:
bash scripts/run_all_checks.sh
```

Report: `docs/reproduction_report.md` · CSV: `results/week7/reproduction_results.csv`

## Final results table (LEGACY fixture — not official RDD2022)

Values from `tables/final/final_comparison_table.csv` (Validation vs Test separated), produced on the synthetic fixture. Official-data results are pending.

| Model | Val mAP@0.5 | Test mAP@0.5 | Params | GFLOPs |
|-------|-------------|--------------|--------|--------|
| YOLOv8n (Week 3) | see table CSV | see table CSV | ~3.1M | ~8.8 |
| YOLO11n (Week 4) | see table CSV | see table CSV | ~2.6M | ~6.6 |
| **Final: YOLO11n@640** | see table CSV | see table CSV | ~2.6M | ~6.6 |

Full summary: `docs/final_results_summary.md` · Week 6 report: `docs/week6_final_results_report.md`

## Example predictions / figures

- `figures/final/false_positives.png`, `false_negatives.png`, `missed_small_objects.png`  
- `figures/final/confusion_matrix.png`, `precision_recall_curves.png`  
- `figures/final/accuracy_efficiency_tradeoff.png`  
- `figures/final/country_performance.png`

## Limitations

- No official-data model has been trained beyond the pipeline smoke test; all reported week metrics are from the legacy fixture.  
- The official split covers six countries in every split (China_MotorBike, Czech, India, Japan, Norway, United_States); it is not a country-held-out protocol.  
- 448 val/test images with a pHash ≤ 4 neighbour in another split are purged (see `docs/dataset_card.md`).  
- Efficiency may be measured on MPS/CPU when T4 is unavailable — see efficiency CSV labels.  
- Checkpoints (`*.pt`) and raw/processed images are gitignored; share them out-of-band for teammates.

## Team contributions

See `docs/team_contributions.md` (Ameya · Prathmesh · Tejas · Vrishin).

## AI assistance disclosure

See `docs/AI_use.md`:

> Generative AI tools were used to brainstorm the project scope, refine the research question, organize documentation, improve grammar, and assist with code planning. The team independently reviewed the dataset information, experimental design, implementation, results, and conclusions. No experimental results or performance metrics were fabricated by AI. The team remains responsible for implementing, verifying, interpreting, and presenting all project work.

## References

1. Arya et al., RDD2022 / CRDDC materials — https://crddc2022.sekilab.global/  
2. Ultralytics YOLO — https://docs.ultralytics.com/  
3. Project docs: `docs/related_work_summary.md`, week reports `docs/week*_*.md`

## Week map

| Week | Focus |
|------|--------|
| 1–2 | Audit, conversion, protocol, configs |
| 3 | Splits + YOLOv8n baseline |
| 4 | YOLO11n baseline + comparison |
| 5 | P2 / attention / 800 / combined ablations |
| 6 | Frozen test eval, errors, efficiency |
| 7 | Reproducibility, packaging, presentation |
