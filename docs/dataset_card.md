# Dataset card — RDD2022 (official release, project subset)

> **Provenance:** every number on this card comes from the **official** RDD2022 release
> (`RDD2022_released_through_CRDDC2022.zip`), processed on 2026-11-26 by
> `scripts/run_official_dataset_pipeline.sh` steps 1–6. Earlier versions of this card described
> the synthetic fixture; those counts are now legacy and live in `data/splits/legacy_fixture/`.
> No model metric on this card — model results on official data are still pending.
> 35,984 images are converted; 35,536 are in train/val/test after the near-duplicate purge.

## Name and source

- **Name:** RDD2022 (Road Damage Dataset 2022, CRDDC2022 release)
- **Repository:** <https://github.com/sekilab/RoadDamageDetector>
- **Archive used:** Figshare mirror (article 21431547), 13.26 GB; the S3 URL in the README returned HTTP 403
- **Version tag in configs:** `RDD2022_released_through_CRDDC2022`

## Purpose

Train and evaluate lightweight YOLO detectors for road-surface damage (cracks, potholes) in
multi-country imagery.

## Raw contents (audit: `results/official_rdd2022/audit/`)

| Country | Labeled train images (= XMLs) | Official test images (unlabeled, unused) | Used |
|---|---:|---:|---|
| China_Drone | 2,401 | 0 | excluded |
| China_MotorBike | 1,977 | 500 | yes |
| Czech | 2,829 | 709 | yes |
| India | 7,706 | 1,959 | yes |
| Japan | 10,506 | 2,627 | yes |
| Norway | 8,161 | 2,040 | yes |
| United_States | 4,805 | 1,200 | yes |
| **Total** | **38,385** | **9,035** | **35,984 modeling images** |

Integrity (modeling set): 0 missing image/XML pairs, 0 malformed XML, 0 corrupt images (full
decode), 0 XML/image size mismatches, 0 partially out-of-bounds boxes, **1 invalid box**
(`Japan_001265`, D20 with xmin = xmax = 198; rejected).

## Class definitions and counts (modeling set, after conversion)

| ID | Class | Description | Boxes |
|---|---|---|---:|
| 0 | D00 | Longitudinal crack | 24,590 |
| 1 | D10 | Transverse crack | 10,567 |
| 2 | D20 | Alligator crack | 10,323 |
| 3 | D40 | Pothole | 6,458 |
| | | **Total** | **51,938** |

Unsupported labels dropped (modeling set): D44 5,057 · D50 3,581 · D43 793 · Repair 277 ·
D01 179 · D11 45 · `D0w0` 1 (probable typo for D00; dropped under the strict mapping).

Per-country supported boxes (D00 / D10 / D20 / D40): China_MotorBike 2678/1096/641/235 ·
Czech 988/399/161/197 · India 1555/68/2021/3187 · Japan 4049/3979/6199/2243 ·
Norway 8570/1730/468/461 · United_States 6750/3295/834/135.

## Negative images

14,136 of 35,984 images (39.3%) have no supported box and keep an empty YOLO label file
(Czech 1,757 · India 4,483 · Japan 2,606 · Norway 5,247 · China_MotorBike 43 · United_States 0).

## Box sizes

Median relative box area (box / image): D00 0.011, D10 0.010, D20 0.083, D40 0.005 — potholes
and thin cracks are mostly small objects (`results/official_rdd2022/audit/box_statistics.csv`).

## Train/val/test split (`grouped_70_15_15_seed6845`)

| Split | Images | Objects | Negatives | D00 | D10 | D20 | D40 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 25,355 (71.4%) | 36,545 | 9,969 | 17,328 | 7,448 | 7,222 | 4,547 |
| Val | 5,163 (14.5%) | 7,540 | 1,985 | 3,493 | 1,541 | 1,563 | 943 |
| Test (internal) | 5,018 (14.1%) | 7,300 | 1,947 | 3,414 | 1,506 | 1,467 | 913 |
| Purged (in no split) | 448 | 553 | 235 | | | | |

Country images per split (train / val / test): China_MotorBike 1427/247/198 · Czech 2000/408/376 ·
India 5406/1133/1126 · Japan 7356/1590/1531 · Norway 5761/1089/1089 · United_States 3405/696/698.

Split fingerprint: `dataset_hash = a9bad058971ade44` (train `00361dce5cc9daec`, val
`948f55c977dd7c52`, test `102823c5c22f3d4d`). Training runs record this hash.

- **Groups:** 50 consecutive filenames per country, assigned whole to one split (seeded search
  over 200 candidate assignments per country, scored on 70/15/15 image, negative and class balance).
- **Exact / identical duplicates:** 1 exact-md5 pair and 58 identical-pHash pairs (all within one
  country) are merged into the same group (54 groups merged) — they can never be split apart.
- **Near-duplicates (pHash distance 1–4):** consecutive frames often land in different 50-image
  blocks. Merging them transitively collapses whole countries into train, so instead the split
  applies a non-transitive **purge**: for every within-country pair at distance ≤ 4
  (`near_duplicate_purge_hamming`) that crosses splits, the val/test member is removed from the
  split lists (448 images: 216 val, 232 test). The files stay on disk and are listed in
  `data/splits/near_duplicate_purged.csv`.
- **Leakage result:** 0 image ids, 0 groups, 0 duplicate clusters, 0 md5 values and **0 pairs at
  pHash distance ≤ 4** cross splits (`results/official_rdd2022/split_leakage_report.json`).
  For information, 872 within-country pairs at distance 6 still cross splits; these are
  visually similar road scenes rather than repeated frames and are not removed.
- The internal test split is carved from labeled train data. The official unlabeled test
  images are never used.

## Annotation limitations

- Competition labels; boundary noise and missed thin cracks are possible.
- Label vocabulary differs by country (D43/D44/D50 only in India/Japan; Repair only in China_MotorBike).
- Class mix differs strongly by country (e.g. India is pothole-heavy, Norway is D00-heavy, US has no negatives).
- No lighting/weather metadata.

## Intended use

Academic coursework / research on lightweight detectors — not a certified infrastructure
inspection product.

## Ethical considerations

- Do not redistribute the raw dataset beyond the license/portal terms; never commit it.
- Avoid identifying private property beyond what public road imagery already shows.
- Report failure modes honestly (missed potholes have safety implications if misused).

## Access instructions

See `data/README.md`. Raw and processed imagery are **gitignored**.
