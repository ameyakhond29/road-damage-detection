# Dataset audit — RDD2022

## Dataset source

- **RDD2022** (Road Damage Dataset 2022), released for CRDDC’2022  
- Official download portal: https://crddc2022.sekilab.global/data/  
- Figshare: https://doi.org/10.6084/m9.figshare.21431547  
- GitHub index / country zips: https://github.com/sekilab/RoadDamageDetector  

## Download instructions

1. Download `RDD2022.zip` or the per-country archives (`RDD2022_Japan.zip`, `RDD2022_India.zip`, …).  
2. Extract into this repository as:

```text
data/raw/RDD2022/
├── China_Drone/
├── China_MotorBike/
├── Czech/
├── India/
├── Japan/
├── Norway/
└── United_States/
```

Each country should contain `train/images/` and `train/annotations/xmls/` (Pascal VOC XML).

3. Confirm discovery:

```bash
python scripts/01_audit_dataset.py --raw-dir data/raw/RDD2022 --seed 6845
```

If the path is wrong or empty, the script prints the same download instructions and exits with code 1.

**Do not upload the dataset to GitHub** (see `.gitignore`).

## Audit command

```bash
python scripts/01_audit_dataset.py \
  --raw-dir data/raw/RDD2022 \
  --results-dir results \
  --figures-dir figures \
  --seed 6845
```

Optional flags:

- `--skip-corrupt-check` — faster audit (skips PIL verify/load)  
- `--exclude-china-drone` — omit China_Drone from reported stats  

## What the audit reports

| Artifact | Contents |
|----------|----------|
| `results/dataset_statistics.csv` | Per-country image/XML counts, missing pairs, corrupt images, class totals |
| `results/class_distribution.csv` | D00 / D10 / D20 / D40 bounding-box counts |
| `results/box_statistics.csv` | Width / height / area mean·std·median (per class + ALL) |
| `results/unsupported_labels.csv` | Labels outside the four main classes (e.g. D50, Repair) |
| `results/audit_summary.json` | Machine-readable summary |
| `figures/class_distribution.png` | Class bar chart |
| `figures/box_size_distribution.png` | Width / height / log-area histograms |

## Class mapping (YOLO)

| Class | ID |
|-------|----|
| D00 | 0 |
| D10 | 1 |
| D20 | 2 |
| D40 | 3 |

Unsupported labels (D50, Repair, …) are ignored during conversion and counted in `data/processed/RDD2022_yolo/removed_unsupported_labels.csv`.

## Filtering policy (used in conversion)

- **Exclude** `China_Drone` (aerial / different camera setup).  
- **Keep** D00, D10, D20, D40 only.  
- **Preserve** images with no boxes as negative examples (empty YOLO `.txt`).  
- **Never delete** original files; write to `data/processed/RDD2022_yolo/`.

## Split method (leakage-safe)

1. Within each country, sort filenames and group every **50 consecutive** names into one block (`{country}__blockXXXX`).  
2. Run **perceptual-hash** duplicate detection; merge groups that contain near-duplicates so duplicates cannot land in different splits.  
3. Assign groups to **train / val / test ≈ 70 / 15 / 15**, stratified by country when enough groups exist.  
4. Write `data/train_split.csv`, `data/val_split.csv`, `data/test_split.csv`, and `results/duplicate_report.csv`.

Default seed: **6845**.

## Current audit numbers

Official RDD2022 was **not downloaded** when Week 1 was implemented. The CSVs currently under `results/` come from a **synthetic fixture** used only to verify the pipeline.

After placing the real dataset at `data/raw/RDD2022`, re-run:

```bash
python scripts/01_audit_dataset.py --raw-dir data/raw/RDD2022 --seed 6845
```

Then paste the printed country list, image/XML totals, and class counts into this section (never invent numbers).
