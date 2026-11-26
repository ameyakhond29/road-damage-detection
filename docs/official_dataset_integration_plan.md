# Official RDD2022 Integration Plan (audit before editing)

Date: 2026-11-26. Scope: replace the synthetic fixture as the *active* dataset with the
official RDD2022 release, keep the fixture for unit tests / debugging only, and keep every
existing fixture-derived result as a clearly labelled legacy artifact. Nothing is committed
or pushed.

## 1. Official source

| Item | Value |
|------|-------|
| Repository | <https://github.com/sekilab/RoadDamageDetector> |
| Primary archive (per README) | `https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip` |
| Status of primary URL (2026-11-26) | **HTTP 403 AccessDenied** |
| Official mirror (linked from the same README) | Figshare article 21431547, file `RDD2022_released_through_CRDDC2022.zip` (13.26 GB), `https://ndownloader.figshare.com/files/38030910` |
| Layout (from `Directory_Structure_CRDDC_RDD2022.txt`) | `RDD2022/<Country>/train/{images,annotations/xmls}` and `RDD2022/<Country>/test/images` (test is unlabeled) |

`scripts/download_rdd2022.py` tries the S3 URL first and falls back to the Figshare mirror.

## 2. Where the fixture / hard-coded paths are used today

| Location | Current behaviour | Risk |
|----------|-------------------|------|
| `data/fixture/RDD2022/` (from `scripts/00_create_fixture.py`) | Synthetic images + VOC XML | Was the only data ever processed |
| `data/processed/RDD2022_yolo/` | Converted fixture (+ materialized `images/{train,val,test}`) | Default of `src/paths.default_processed_dir()` |
| `configs/rdd2022.yaml` | Absolute Mac path to fixture processed dir; gitignored; rewritten by `ensure_rdd2022_data_yaml()` | Machine path; fixture is the implicit default |
| `configs/rdd2022_smoke.yaml` | Absolute Mac path, fixture smoke subset | Machine path |
| `configs/rdd2022_ood.yaml` | Relative path to fixture processed dir | Points at fixture |
| `data/splits/{train,val,test}.csv`, `data/splits/split_summary.json`, `data/{train,val,test}_split.csv` | Splits computed over the fixture | Would be silently reused by official training |
| `scripts/run_week3.sh` | Falls back to `data/fixture/RDD2022` when raw data is missing | Silent fixture use |
| `scripts/run_week4.sh`, `run_week5.sh`, `run_week6.sh` | Use `configs/rdd2022.yaml` (fixture) | Silent fixture use |
| `scripts/02_convert_annotations.py`, `03_create_splits.py`, `04_write_yolo_config.py`, `05_smoke_test.py` | Week 1 legacy pipeline; `04` overwrites `configs/rdd2022.yaml` | Would clobber the new tracked config |
| `scripts/create_week3_splits.py` | Overwrites `configs/rdd2022.yaml` with an absolute path | Clobbers tracked config |
| `scripts/train_yolov8_baseline.py`, `train_yolo11_baseline.py`, `train_week5_ablation.py` | Pass `configs/rdd2022.yaml` straight to Ultralytics; write to `results/week{3,4,5}` | Would overwrite fixture results |
| `scripts/evaluate_week4.py`, `evaluate_week5_ablation.py`, `compare_baselines.py`, `run_final_evaluation.py`, `error_analysis.py`, `amend_full_test_counts.py`, `reproduce_main_result.py`, `check_environment.py`, `run_all_checks.sh` | Evaluate fixture-trained checkpoints against the fixture processed dir | Must not be pointed at official data with fixture checkpoints |
| `results/week1`–`results/week7`, `figures/`, `docs/*week*` | All metrics are fixture-derived | Must never be presented as RDD2022 results |

## 3. Changes

1. **Paths (`src/paths.py`)**: `RDD2022_ROOT` env var → raw root; a CLI `--data-root` overrides
   it; default `data/raw/RDD2022`. `--dataset fixture` is the only way to select
   `data/fixture/RDD2022`. Official processed root is `data/processed/{images,labels}`.
   Legacy fixture processed dir becomes `legacy_fixture_processed_dir()`. Missing data raises
   a clear error that prints the download command.
2. **Tracked, portable `configs/rdd2022.yaml`**: project dataset config (relative paths,
   class names, version, seed, excluded folders). Ultralytics needs an absolute `path`, so
   a runtime YAML is generated per machine at `data/processed/ultralytics/rdd2022_official.yaml`
   (gitignored) from `data/splits/*.csv`. Week 5/baseline configs keep `data: configs/rdd2022.yaml`.
3. **Fixture runtime YAML**: legacy scripts get `data/processed/RDD2022_yolo/rdd2022_fixture.yaml`
   via `ensure_legacy_fixture_data_yaml()`; they never touch `configs/rdd2022.yaml` again.
4. **New/updated official scripts**: `download_rdd2022.py`, `01_audit_dataset.py`,
   `convert_voc_to_yolo.py`, `validate_dataset.py`, `create_week3_splits.py`,
   `smoke_test_official_rdd2022.py`, `run_official_dataset_pipeline.sh`. Shared logic lives in
   `src/dataset/official.py`. All outputs go to `results/official_rdd2022/`.
5. **Training scripts**: default `--dataset official`; `--dataset fixture` is explicit;
   outputs `results/official_rdd2022/week{3,4,5}/` (fixture runs → `results/fixture_runs/`);
   writing into legacy `results/week*` is refused; run cards store dataset source + hash.
6. **Legacy runners** (`run_week3-6.sh`, Week 1 `02`–`05` scripts): refuse to run unless
   `ALLOW_LEGACY_FIXTURE=1`, and their defaults point at the legacy fixture paths.
7. **Legacy split files**: copy current fixture `data/splits/*` and `data/*_split.csv`
   into `data/splits/legacy_fixture/` before official splits overwrite `data/splits/`.
8. **Nothing deleted**: fixture, `data/processed/RDD2022_yolo`, `results/week*` stay; they are
   marked legacy in docs and `results/LEGACY_FIXTURE_RESULTS.md`.

## 4. Dataset rules (official)

- Labeled `train` folders only; official `test/images` (unlabeled) are counted but never used.
- Class mapping D00→0, D10→1, D20→2, D40→3; other labels counted as unsupported and dropped.
- `China_Drone` excluded from conversion/splits (still counted in the audit).
- Images with an XML but no supported boxes are negatives (empty label file); images with
  no XML are reported as unlabeled and excluded.
- Original JPG/XML never modified; processed images are hardlinks (fallback symlink → copy).
- Split: deterministic, seed 6845, groups = blocks of 50 consecutive filenames per country,
  merged with exact-byte and pHash duplicate components; 70/15/15 per country, choosing among
  seeded candidate assignments the one closest to the global class distribution.

## 5. Disk / runtime notes

Archive 13.26 GB + extracted ~13 GB; hardlinked processed images add ~0 GB. Audit, conversion,
hashing and validation use multiprocessing. Torch training on this Mac must run outside the
Cursor sandbox (MPS/torch segfault inside it).
