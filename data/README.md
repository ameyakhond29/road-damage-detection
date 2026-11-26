# Data directory

The **active dataset is the official RDD2022 release**. The synthetic fixture is kept only
for unit tests and debugging and is never used unless `--dataset fixture` is passed.

## Layout

```text
data/
  raw/
    RDD2022_released_through_CRDDC2022.zip   # official archive (13.26 GB, gitignored)
    RDD2022/<Country>/train/images/*.jpg      # original JPGs   (gitignored, never modified)
    RDD2022/<Country>/train/annotations/xmls/*.xml  # original Pascal VOC (never modified)
    RDD2022/<Country>/test/images/*.jpg       # official test images — UNLABELED, never used
  processed/                                  # gitignored
    images/<Country>/*.jpg                    # hardlinks to the raw JPGs (no extra disk)
    labels/<Country>/*.txt                    # YOLO labels (empty file = negative image)
    manifest.csv                              # one row per converted image
    ultralytics/                              # per-machine runtime YAML + image lists
  splits/                                     # tracked: official split CSVs
    train.csv  val.csv  test.csv  split_summary.json  near_duplicate_purged.csv
    legacy_fixture/                           # LEGACY synthetic-fixture splits (Weeks 1–7)
  fixture/RDD2022/                            # synthetic fixture — tests/debugging ONLY
  processed/RDD2022_yolo/                     # LEGACY processed fixture (Weeks 1–7 results)
  ood/                                        # optional country-held-out track (legacy)
```

## Source

- Repository: <https://github.com/sekilab/RoadDamageDetector>
- Archive listed there: `https://bigdatacup.s3.ap-northeast-1.amazonaws.com/2022/CRDDC2022/RDD2022/RDD2022.zip`
  (returned HTTP 403 on 2026-11-26)
- Official mirror linked from the same README (used): Figshare article 21431547,
  `RDD2022_released_through_CRDDC2022.zip`, <https://doi.org/10.6084/m9.figshare.21431547>

The archive holds one zip per country (`RDD2022/<Country>.zip`); the download script
extracts them in place without writing a second copy.

## Download / prepare (one command)

```bash
bash scripts/run_official_dataset_pipeline.sh          # download → audit → convert → validate → split → stats → smoke
```

Or step by step:

```bash
python scripts/download_rdd2022.py                      # skips download if a valid archive exists
python scripts/01_audit_dataset.py --data-root data/raw/RDD2022
python scripts/convert_voc_to_yolo.py --data-root data/raw/RDD2022
python scripts/validate_dataset.py --config configs/rdd2022.yaml
python scripts/create_week3_splits.py --config configs/rdd2022.yaml --seed 6845
python scripts/validate_dataset.py --config configs/rdd2022.yaml --require-splits
```

Use an existing copy elsewhere with `export RDD2022_ROOT=/path/to/RDD2022` (a `--data-root`
argument overrides the variable). The root is searched recursively for the country folders.

## Rules

- Class mapping: D00=0 (longitudinal crack), D10=1 (transverse), D20=2 (alligator), D40=3 (pothole).
  All other labels (D01, D11, D43, D44, D50, Repair, and one `D0w0` typo) are dropped and counted
  in `results/official_rdd2022/unsupported_labels.csv`.
- `China_Drone` is excluded (aerial camera; counted in the audit only).
- Only labeled `train` folders are used for train/val/internal test. The official `test/` images
  have no labels and are never used for training or model selection.
- Images whose XML has no supported box are kept as negatives (empty label file).
- Invalid boxes (xmax ≤ xmin, ymax ≤ ymin, fully outside the image) are rejected and logged.

## Split

`grouped_70_15_15_seed6845`: per country, blocks of 50 consecutive filenames form groups;
groups linked by exact-md5 or identical-pHash images are merged; among 200 seeded candidate
assignments the one closest to 70/15/15 for images, negatives and every class is kept.
Then, for every within-country image pair at pHash distance ≤ 4 that ends up in different
splits, the val/test member is dropped from the split lists (not from disk) and logged in
`splits/near_duplicate_purged.csv` (448 images).

Result: train 25,355 · val 5,163 · test 5,018 (35,536 of 35,984 converted images), 0 leakage.
Leakage report: `results/official_rdd2022/split_leakage_report.json`.
Do not regenerate the split for model shopping.

## Never commit data

`data/raw/`, `data/processed/`, `*.zip`, `results/official_rdd2022/`, `runs/`, `weights/` and
`checkpoints/` are gitignored. Only code, configs, docs and the split CSVs are tracked.

## Colab / Kaggle

See the "Colab / Kaggle T4" section of the top-level `README.md`. On Kaggle the raw dataset can
live read-only under `/kaggle/input/...`; set `RDD2022_ROOT` to it — processed images then fall
back to symlinks/copies inside the working directory.
