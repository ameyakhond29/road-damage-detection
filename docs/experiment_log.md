# Experiment log

## Official RDD2022 (active)

Data: official RDD2022 (`configs/rdd2022.yaml`), split `grouped_70_15_15_seed6845`,
`dataset_hash a9bad058971ade44` (train 25,355 · val 5,163 · test 5,018). Outputs:
`results/official_rdd2022/`.

| ID | Date | What | Status | Output |
|---|---|---|---|---|
| OFF-DATA-1 | 2026-11-26 | Download, audit, VOC→YOLO conversion, validation | done, validation passed | `results/official_rdd2022/{audit,conversion_manifest.json,validation_report.json}` |
| OFF-SPLIT-1 | 2026-11-26 | Grouped 70/15/15 split + near-duplicate purge (pHash ≤ 4, 448 images) | done, leakage check passed | `data/splits/`, `results/official_rdd2022/split_leakage_report.json` |
| OFF-SMOKE-1 | 2026-11-26 | YOLOv8n, 300 train / 100 val images, 2 epochs, imgsz 320, CPU | `completed_smoke` — **pipeline verification only, not performance** | `results/official_rdd2022/smoke_test/` |
| OFF-W3/W4/W5 | — | Baselines and ablations on official data | **Pending** (needs GPU) | `results/official_rdd2022/week{3,4,5}/` |

No official-data metric exists yet. Nothing from the legacy section below may be cited as an
official-data result.

## LEGACY — synthetic fixture (Weeks 3–7)

Every row in this section was produced on the synthetic fixture (`data/fixture/RDD2022/`).
Canonical machine-readable table: `results/week7/experiment_manifest.csv`.

## Rules

- Model selection / early stopping: **validation only**  
- Test metrics appear only for Week 6 frozen evaluation rows (`W6-EVAL-*`)  
- Smoke runs keep `status=completed_smoke` — do not treat as full-budget SOTA claims  

## Highlights

| ID | Role |
|---|---|
| W3-YOLOv8n | Reference baseline |
| W4-YOLO11n | Strong baseline |
| W5-yolo11n_640-s0 | Week 5 control → **final selected** |
| W5-*-s0 | P2 / attention / 800 / combined ablations |
| W6-EVAL-final_selected | Frozen held-out test evaluation |

Seeds 1/2 for Week 5: **Pending** unless later trained under new IDs.
