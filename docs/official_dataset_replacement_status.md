# Official RDD2022 replacement — status

**Date:** 2026-11-26 · **Scope:** replace the synthetic fixture in the active pipeline with the
official RDD2022 dataset. Local changes only; nothing committed or pushed.

Every number in this document comes from `results/official_rdd2022/` (official data). No
synthetic-fixture metric appears here.

## What changed

| Area | Before | After |
|---|---|---|
| Default dataset | Synthetic fixture (`data/fixture/RDD2022`, `data/processed/RDD2022_yolo`) | Official RDD2022 (`data/raw/RDD2022` → `data/processed`) |
| Dataset config | `configs/rdd2022.yaml` with an absolute Mac path, gitignored | Portable, tracked; `RDD2022_ROOT` env var or `--data-root` |
| Ultralytics YAML | Same file as the config | Generated per machine: `data/processed/ultralytics/rdd2022_official.yaml` |
| Splits | `data/splits/*` (fixture, 50-image test) | `data/splits/{train,val,test}.csv` (official); fixture copies in `data/splits/legacy_fixture/` |
| Results | `results/week3`–`week7`, `final/` | `results/official_rdd2022/` (official); old folders marked LEGACY (`results/LEGACY_FIXTURE_RESULTS.md`) |
| Fixture | Implicit default | Only with `--dataset fixture`; runs go to `results/fixture_runs/` |

The fixture and every old result file are kept unchanged.

## Where the data is

- Raw (gitignored): `data/raw/RDD2022/<Country>/train/{images,annotations/xmls}` — original JPG/XML, never modified
- Processed (gitignored): `data/processed/images/<Country>/` (hardlinks to the raw JPGs), `data/processed/labels/<Country>/` (YOLO txt)
- Splits (tracked): `data/splits/train.csv`, `val.csv`, `test.csv`, `split_summary.json`, `near_duplicate_purged.csv`
- Source: <https://github.com/sekilab/RoadDamageDetector>; archive from the Figshare mirror (13,264,172,619 bytes) because the S3 link returned HTTP 403

## File counts

| Item | Count |
|---|---:|
| Files extracted | 85,805 |
| Labeled train images = XMLs (7 countries) | 38,385 |
| China_Drone (excluded) | 2,401 |
| Modeling images converted (6 countries) | 35,984 |
| YOLO label files written | 35,984 |
| Official unlabeled test images (never used) | 9,035 |

Per country (modeling images / negatives / supported boxes): China_MotorBike 1,977 / 43 / 4,650 ·
Czech 2,829 / 1,757 / 1,745 · India 7,706 / 4,483 / 6,831 · Japan 10,506 / 2,606 / 16,469 ·
Norway 8,161 / 5,247 / 11,229 · United_States 4,805 / 0 / 11,014.

## Labels converted

| Class | ID | Boxes |
|---|---:|---:|
| D00 longitudinal crack | 0 | 24,590 |
| D10 transverse crack | 1 | 10,567 |
| D20 alligator crack | 2 | 10,323 |
| D40 pothole | 3 | 6,458 |
| **Total** | | **51,938** |

Negatives kept (empty label file): 14,136 (39.3%).
Unsupported labels dropped and counted (`unsupported_labels.csv`): D44 5,057 · D50 3,581 · D43 793 ·
Repair 277 · D01 179 · D11 45 · D0w0 1.

## Invalid files

- Corrupt images (full decode): 0 · missing image/XML pairs: 0 · malformed XML/objects: 0
- Invalid boxes: 1 (`Japan_001265`, D20, xmin = xmax = 198), rejected and logged in `conversion_issues.csv`
- Exact-duplicate images: 1 pair; identical-pHash pairs: 58 (all within one country)

## Splits (`grouped_70_15_15_seed6845`, `dataset_hash a9bad058971ade44`)

| Split | Images | Objects | Negatives | D00 | D10 | D20 | D40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 25,355 | 36,545 | 9,969 | 17,328 | 7,448 | 7,222 | 4,547 |
| Val | 5,163 | 7,540 | 1,985 | 3,493 | 1,541 | 1,563 | 943 |
| Test (internal) | 5,018 | 7,300 | 1,947 | 3,414 | 1,506 | 1,467 | 913 |
| Purged near-duplicates | 448 | 553 | 235 | | | | |

Every country appears in every split. The internal test split is carved from labeled train data.

## Leakage

- 0 image ids, 0 groups (50-image blocks), 0 duplicate clusters and 0 md5 values cross splits
- 0 within-country pairs at pHash distance ≤ 4 cross splits, after purging 448 val/test images
  (216 val, 232 test) that had a near-identical frame in another split. Purged images stay on disk.
- For information, 872 pairs at distance 6 still cross; these look like similar scenes, not repeated frames.
- Report: `results/official_rdd2022/split_leakage_report.json` → `passed: true`
- Validation (`validate_dataset.py --require-splits`): **passed** (`validation_report.json`)

## Smoke test status

`results/official_rdd2022/smoke_test/` — **completed**. YOLOv8n, 300 train + 100 val images from the
official splits (test split not used), 2 epochs, imgsz 320, CPU, 44 s of training. Loss falls
(train cls_loss 8.03 → 4.71). Ground-truth overlays were checked visually and boxes align with damage.

This run is **pipeline verification only**. Its metrics are not model performance and must not be
reported as results.

## Remaining work

1. Train YOLOv8n / YOLO11n baselines on official data on a T4 (`results/official_rdd2022/week3`, `week4`).
2. Run the Week 5 ablations (seeds 0/1/2) on official data.
3. Port the Week 6 frozen evaluation, error analysis and T4 efficiency benchmark to official data; until then those scripts stay on the legacy fixture.
4. Regenerate the final tables and figures from official runs; update `docs/final_results_summary.md`.
5. Team decision: whether to remap the single `D0w0` label to D00 (currently dropped).

## Discrepancies from the fixture

| Property | Fixture (legacy) | Official |
|---|---|---|
| Modeling images (train/val/test) | 361 synthetic (252 / 59 / 50) | 35,536 real (25,355 / 5,163 / 5,018) |
| Test split geography | United States only | All six countries |
| Negatives | 32–39% per split | 38.4–39.3% per split |
| Unsupported labels | D50, Repair only | Also D44 (5,057), D43, D01, D11, `D0w0` |
| Near-duplicates | none | consecutive dashcam frames; purge needed |
| Invalid boxes | none | 1 |
| Week 1–7 metrics | near zero on synthetic data | no official metric yet |
