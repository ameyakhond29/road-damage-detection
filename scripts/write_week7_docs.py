#!/usr/bin/env python3
"""Generate Week 7 documentation and experiment manifest from saved artifacts only."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.week6_utils import pending, write_csv


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(v, d=4):
    if pending(v):
        return "Pending"
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return str(v)


def write_experiment_manifest() -> None:
    rows = []
    # Week 3
    m3 = read_json(ROOT / "results/week3/yolov8n_baseline/metrics.json")
    if m3:
        met = m3.get("metrics") or {}
        rows.append(
            {
                "experiment_id": "W3-YOLOv8n",
                "date": m3.get("finished_at", "Pending"),
                "model": "YOLOv8n",
                "model_variant": "reference_baseline",
                "dataset_split": "grouped_70_15_15_seed6845",
                "image_size": m3.get("imgsz", 640),
                "batch_size": m3.get("batch", "Pending"),
                "epochs": m3.get("epochs", "Pending"),
                "learning_rate": m3.get("lr0", 0.01),
                "optimizer": m3.get("optimizer", "auto"),
                "seed": m3.get("seed", 6845),
                "gpu": m3.get("gpu_name", "Pending"),
                "training_duration_sec": m3.get("training_time_sec", "Pending"),
                "checkpoint_path": m3.get("best_checkpoint", "Pending"),
                "val_map50": met.get("metrics/mAP50(B)", "Pending"),
                "val_map50_95": met.get("metrics/mAP50-95(B)", "Pending"),
                "test_map50": "see_week6",
                "test_map50_95": "see_week6",
                "status": m3.get("status", "Pending"),
                "notes": "Week 3 baseline; smoke if epochs short",
            }
        )
    m4 = read_json(ROOT / "results/week4/yolo11n_baseline/metrics.json")
    if m4:
        met = m4.get("metrics") or {}
        rows.append(
            {
                "experiment_id": "W4-YOLO11n",
                "date": m4.get("finished_at", "Pending"),
                "model": "YOLO11n",
                "model_variant": "strong_baseline",
                "dataset_split": "grouped_70_15_15_seed6845",
                "image_size": m4.get("imgsz", 640),
                "batch_size": m4.get("batch", "Pending"),
                "epochs": m4.get("epochs", "Pending"),
                "learning_rate": m4.get("lr0", 0.01),
                "optimizer": m4.get("optimizer", "auto"),
                "seed": m4.get("seed", 6845),
                "gpu": m4.get("gpu_name", "Pending"),
                "training_duration_sec": m4.get("training_time_sec", "Pending"),
                "checkpoint_path": m4.get("best_checkpoint", "Pending"),
                "val_map50": met.get("metrics/mAP50(B)", "Pending"),
                "val_map50_95": met.get("metrics/mAP50-95(B)", "Pending"),
                "test_map50": "see_week6",
                "test_map50_95": "see_week6",
                "status": m4.get("status", "Pending"),
                "notes": "Week 4 strong baseline",
            }
        )
    # Week 5 ablations
    for exp in (
        "yolo11n_640",
        "yolo11n_p2_640",
        "yolo11n_attention_640",
        "yolo11n_800",
        "combined_800",
    ):
        card = read_json(ROOT / f"results/week5/{exp}/seed_0/metrics.json")
        if not card:
            rows.append(
                {
                    "experiment_id": f"W5-{exp}-s0",
                    "date": "Pending",
                    "model": "YOLO11n",
                    "model_variant": exp,
                    "dataset_split": "grouped_70_15_15_seed6845",
                    "image_size": "Pending",
                    "batch_size": "Pending",
                    "epochs": "Pending",
                    "learning_rate": 0.01,
                    "optimizer": "auto",
                    "seed": 0,
                    "gpu": "Pending",
                    "training_duration_sec": "Pending",
                    "checkpoint_path": "Pending",
                    "val_map50": "Pending",
                    "val_map50_95": "Pending",
                    "test_map50": "n/a_week5",
                    "test_map50_95": "n/a_week5",
                    "status": "Pending",
                    "notes": "Seeds 1/2 also Pending unless trained",
                }
            )
            continue
        rows.append(
            {
                "experiment_id": f"W5-{exp}-s0",
                "date": card.get("finished_at", "Pending"),
                "model": "YOLO11n",
                "model_variant": exp,
                "dataset_split": card.get("split_id", "grouped_70_15_15_seed6845"),
                "image_size": card.get("input_resolution", "Pending"),
                "batch_size": card.get("batch_size", "Pending"),
                "epochs": card.get("epochs", "Pending"),
                "learning_rate": card.get("learning_rate", 0.01),
                "optimizer": card.get("optimizer", "auto"),
                "seed": card.get("seed", 0),
                "gpu": card.get("gpu", "Pending"),
                "training_duration_sec": card.get("training_duration_sec", "Pending"),
                "checkpoint_path": card.get("best_checkpoint", "Pending"),
                "val_map50": card.get("map50", "Pending"),
                "val_map50_95": card.get("map50_95", "Pending"),
                "test_map50": "n/a_week5",
                "test_map50_95": "n/a_week5",
                "status": card.get("status", "Pending"),
                "notes": "Week 5 val-only selection; test deferred to Week 6",
            }
        )
    # Week 6 final eval (evaluation-only)
    for r in read_csv(ROOT / "results/week6/final_test_results.csv"):
        if r.get("split_label") != "Test":
            continue
        rows.append(
            {
                "experiment_id": f"W6-EVAL-{r.get('model_id')}",
                "date": "Week6",
                "model": r.get("display_name"),
                "model_variant": r.get("role"),
                "dataset_split": "grouped_70_15_15_seed6845",
                "image_size": r.get("imgsz"),
                "batch_size": "n/a_eval",
                "epochs": "n/a_eval",
                "learning_rate": "n/a_eval",
                "optimizer": "n/a_eval",
                "seed": r.get("seed"),
                "gpu": r.get("gpu"),
                "training_duration_sec": "n/a_eval",
                "checkpoint_path": r.get("checkpoint"),
                "val_map50": "see_validation_row",
                "val_map50_95": "see_validation_row",
                "test_map50": r.get("map50"),
                "test_map50_95": r.get("map50_95"),
                "status": r.get("status"),
                "notes": f"Frozen protocol eval; conf={r.get('confidence_threshold_operating')}",
            }
        )

    out = ROOT / "results/week7/experiment_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows)
    print(f"Wrote {out}")


def write_docs() -> None:
    split = read_json(ROOT / "data/splits/split_summary.json")
    comp = read_csv(ROOT / "tables/final/final_comparison_table.csv") or read_csv(
        ROOT / "results/week6/final_comparison_table.csv"
    )
    final = next((r for r in comp if r.get("model_id") == "final_selected"), {})
    v8 = next((r for r in comp if r.get("model_id") == "yolov8n"), {})
    v11 = next((r for r in comp if r.get("model_id") == "yolo11n_week4"), {})
    err = read_json(ROOT / "results/week6/error_analysis_summary.json")
    lock = read_json(ROOT / "results/week6/CONFIDENCE_THRESHOLD_LOCKED.json")
    proto = yaml.safe_load((ROOT / "configs/final_evaluation.yaml").read_text()) or {}
    country = read_csv(ROOT / "tables/final/country_level_results.csv") or read_csv(
        ROOT / "results/week6/country_results.csv"
    )
    repro = read_csv(ROOT / "results/week7/reproduction_results.csv")

    # --- repository_audit.md ---
    (ROOT / "docs/repository_audit.md").write_text(
        f"""# Repository audit (Week 7)

_Generated: {datetime.now(timezone.utc).isoformat()}_

## Structure (target)

`README.md`, `requirements.txt`, `environment.yml`, `configs/`, `data/` (+ README), `scripts/`, `src/`, `results/`, `figures/`, `docs/`, `presentation/`, `tests/`, `tables/final/`.

## Missing / added in Week 7

| Item | Status |
|---|---|
| `environment.yml` | Added |
| `scripts/setup_environment.sh` | Added |
| `scripts/check_environment.py` | Added |
| `scripts/run_all_checks.sh` | Added |
| `scripts/reproduce_main_result.py` | Added |
| `presentation/` | Added |
| `results/final/`, `figures/final/`, `tables/final/` | Packaged |
| `data/README.md` | Added |

## Broken / risky paths

| Issue | Severity | Action |
|---|---|---|
| Absolute machine path in `configs/rdd2022.yaml` | High | **Mitigated, not removed.** Ultralytics ignores relative `path` values, so `src/paths.py::ensure_rdd2022_data_yaml()` writes an absolute path for each machine. The file is now gitignored and regenerated by `check_environment.py` / `setup_environment.sh`. |
| Absolute paths inside Week 6 CSV/JSON result fields | Medium | Left as historical run logs; configs now portable |
| Root `yolo11n.pt` / `yolov8n.pt` downloads | Low | gitignored; Ultralytics auto-downloads if missing |

## Duplicate / overlapping scripts

| Pair | Notes |
|---|---|
| `02_convert_annotations.py` vs `convert_voc_to_yolo.py` | Prefer Week 3 wrapper `convert_voc_to_yolo.py` |
| `03_create_splits.py` vs `create_week3_splits.py` | Prefer `create_week3_splits.py` + tracked `data/splits/` |
| `configs/models/yolo11n_p2.yaml` (stub) vs `yolo11n-p2.yaml` | Stub aliases to real YAML via `src/models/resolve.py` |
| Legacy `data/train_split.csv` vs `data/splits/train.csv` | Canonical = `data/splits/` |

## Configuration inconsistencies

- Week 3/4 baselines used seed **6845**; Week 5 principal seeds **0/1/2** (only seed 0 completed). Documented in protocol.
- Smoke epoch budgets ≠ full 100-epoch recipe — status fields say `completed_smoke`.
- Efficiency measured on Apple MPS, not T4 — labeled in efficiency CSV.

## Missing metadata

- Seeds 1/2 for Week 5 ablations: **Pending**
- Peak GPU memory on MPS: often **Pending**
- Designed country-held-out OOD: not replacing main split

## Reproducibility risks

1. Smoke-scale metrics (~0 mAP) are unstable across devices.
2. Checkpoints are gitignored (`*.pt`) — must be shared separately or retrained under new IDs.
3. Processed images gitignored — must rebuild from raw RDD2022.
4. Ultralytics version drift can change eval numerics slightly.

## Recommended cleanup (done or deferred)

- [ ] Portable dataset path in `rdd2022.yaml` — not possible with Ultralytics; file is machine-generated and gitignored instead
- [x] Final results package folders
- [x] Environment + reproduction scripts
- [ ] Optional: delete unused legacy scripts after team agrees
- [ ] Optional: upload checkpoints to shared drive (not GitHub) for teammates
""",
        encoding="utf-8",
    )

    # --- dataset_card.md ---
    tr = (split.get("splits") or {}).get("train", {})
    va = (split.get("splits") or {}).get("val", {})
    te = (split.get("splits") or {}).get("test", {})
    (ROOT / "docs/dataset_card.md").write_text(
        f"""# Dataset card — RDD2022 (project subset)

> **Provenance warning:** the official RDD2022 download was **not** present in `data/raw/` for any run in this repository. All processed images, splits-as-materialized, and every metric in `results/` were produced from the **synthetic fixture** in `data/fixture/RDD2022/` (created by `scripts/00_create_fixture.py`). The split CSVs in `data/splits/` are real grouping outputs over the fixture. Do not cite any count or metric here as RDD2022 statistics until the real download is used.

## Name and source

- **Name:** RDD2022 Road Damage Dataset  
- **Portal:** https://crddc2022.sekilab.global/data/  
- **Figshare:** https://doi.org/10.6084/m9.figshare.21431547  
- **Reference code:** https://github.com/sekilab/RoadDamageDetector  

## Purpose

Train and evaluate lightweight YOLO detectors for road surface damage (cracks, potholes) in multi-country imagery.

## Image / annotation counts (this repo’s fixed split)

| Split | Images | Boxes | Negatives |
|---|---|---|---|
| Train | {tr.get('n_images', 'Pending')} | {tr.get('n_boxes', 'Pending')} | {tr.get('n_negatives', 'Pending')} |
| Val | {va.get('n_images', 'Pending')} | {va.get('n_boxes', 'Pending')} | {va.get('n_negatives', 'Pending')} |
| Test | {te.get('n_images', 'Pending')} | {te.get('n_boxes', 'Pending')} | {te.get('n_negatives', 'Pending')} |

Source: `data/splits/split_summary.json` (seed={split.get('seed')}).

## Class definitions

| ID | Class | Description |
|---|---|---|
| 0 | D00 | Longitudinal crack |
| 1 | D10 | Transverse crack |
| 2 | D20 | Alligator crack |
| 3 | D40 | Pothole |

Unsupported VOC labels (e.g. D50, Repair) are dropped during conversion.

## Country distribution

- Train: {tr.get('country_distribution', 'Pending')}  
- Val: {va.get('country_distribution', 'Pending')}  
- Test: {te.get('country_distribution', 'Pending')}  

`China_Drone` excluded by default (different camera setup).

## Train/val/test split

- **ID:** `grouped_70_15_15_seed6845`  
- **Ratios:** 70 / 15 / 15  
- **Grouping:** consecutive filename blocks of size {split.get('block_size', 50)} per country  
- **Duplicate leakage:** pHash near-duplicates forced into the same split (`duplicate_leakage.passed={ (split.get('duplicate_leakage') or {}).get('passed') }`)

## Negative images

Images with no in-scope boxes keep empty YOLO label files and remain in the split.

## Class imbalance

See `results/class_distribution.csv` and per-split `per_class_boxes` in `split_summary.json`. D10/D20 may dominate depending on country mix.

## Annotation limitations

- Crowdsourced / competition labels; boundary noise and missed thin cracks possible.  
- Box size varies; many thin cracks are small-area objects.  
- No official lighting/weather metadata tags in this pipeline.

## Intended use

Academic coursework / research on lightweight detectors — not a certified infrastructure inspection product.

## Ethical considerations

- Do not redistribute the raw dataset beyond license/portal terms.  
- Avoid identifying private property beyond what public road imagery already shows.  
- Report failure modes honestly (missed potholes can have safety implications if misused).

## Access instructions

See `data/README.md`. Raw and processed imagery are **gitignored**.
""",
        encoding="utf-8",
    )

    # --- experiment_log.md ---
    (ROOT / "docs/experiment_log.md").write_text(
        """# Experiment log

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
""",
        encoding="utf-8",
    )

    # --- final_results_summary.md ---
    country_lines = []
    for r in country:
        if r.get("model_id") not in (None, "final_selected"):
            continue
        country_lines.append(
            f"| {r.get('split_label')} | {r.get('country')} | {r.get('n_images')} | "
            f"{r.get('n_objects')} | {fmt(r.get('box_recall_iou50'))} |"
        )
    if not country_lines:
        country_lines = ["| Pending | Pending | Pending | Pending | Pending |"]

    (ROOT / "docs/final_results_summary.md").write_text(
        f"""# Final results summary

_Values loaded from packaged / Week 6 saved files only. No invented metrics._

> **Provenance warning:** all values are from the synthetic fixture, not the official RDD2022 download. The original final test run used smoke mode. Its FP/FN counts and error analysis were re-computed on the full 50-image test split under protocol amendment `week6_full_test_counts_v1` (counts only; model, thresholds, and selection unchanged). See `docs/known_issues_and_resolutions.md`.

## 1. Best model

**{proto.get('final_model', {}).get('display_name', 'YOLO11n @ 640')}**  
Checkpoint: `{proto.get('final_model', {}).get('checkpoint')}`  
Selected on **validation** (Week 5). Operating conf **{lock.get('selected_conf', proto.get('confidence_threshold'))}** locked on validation.

## 2. Baseline comparison (Test mAP@0.5)

| Model | Val mAP50 | Test mAP50 |
|---|---|---|
| YOLOv8n | {fmt(v8.get('val_map50'))} | {fmt(v8.get('test_map50'))} |
| YOLO11n Week 4 | {fmt(v11.get('val_map50'))} | {fmt(v11.get('test_map50'))} |
| Final selected | {fmt(final.get('val_map50'))} | {fmt(final.get('test_map50'))} |

## 3. Main performance metrics (final model)

| Metric | Validation | Test |
|---|---|---|
| mAP@0.5 | {fmt(final.get('val_map50'))} | {fmt(final.get('test_map50'))} |
| mAP@0.5:0.95 | {fmt(final.get('val_map50_95'))} | {fmt(final.get('test_map50_95'))} |
| Precision | {fmt(final.get('val_precision'))} | {fmt(final.get('test_precision'))} |
| Recall | {fmt(final.get('val_recall'))} | {fmt(final.get('test_recall'))} |
| F1 | {fmt(final.get('val_f1'))} | {fmt(final.get('test_f1'))} |

## 4. Per-class AP@0.5 (final)

| Class | Val | Test |
|---|---|---|
| D00 | {fmt(final.get('val_ap50_D00'))} | {fmt(final.get('test_ap50_D00'))} |
| D10 | {fmt(final.get('val_ap50_D10'))} | {fmt(final.get('test_ap50_D10'))} |
| D20 | {fmt(final.get('val_ap50_D20'))} | {fmt(final.get('test_ap50_D20'))} |
| D40 | {fmt(final.get('val_ap50_D40'))} | {fmt(final.get('test_ap50_D40'))} |

## 5. Small-object AP

| Split | Value |
|---|---|
| Validation | {fmt(final.get('val_small_object_ap'))} |
| Test | {fmt(final.get('test_small_object_ap'))} |

## 6. Computational trade-offs (final)

| Item | Value |
|---|---|
| Parameters | {fmt(final.get('parameter_count'), 0)} |
| GFLOPs | {fmt(final.get('gflops'), 2)} |
| Latency ms (mean) | {fmt(final.get('latency_ms_mean'), 2)} |
| Peak GPU MB | {fmt(final.get('peak_gpu_memory_mb'), 1)} |
| Training time sec | {fmt(final.get('training_time_sec'), 1)} |

Week 5 ablations increased GFLOPs (P2 ~10.5, combined ~14.6) without validated mAP gains under smoke budgets.

## 7. Country-level robustness

| Split | Country | Images | Objects | Box recall@0.5 |
|---|---|---|---|---|
{chr(10).join(country_lines)}

Test set is United States–only under the fixed split (not a designed OOD protocol).

## 8. Main failure cases

From `error_analysis_summary.json`: images={err.get('n_images', 'Pending')}, error rows={err.get('n_error_rows', 'Pending')}, pothole FNs={err.get('n_pothole_fn', 'Pending')}, multi-damage images={err.get('n_multi_damage_images', 'Pending')}.  
Figures: `figures/final/false_positives.png`, `false_negatives.png`, `missed_small_objects.png`.

## 9. Project limitations

- Smoke training → near-zero absolute mAP; do not overclaim.  
- Multi-seed incomplete (1/2 Pending).  
- No T4 numbers in local efficiency run.  
- Thin cracks remain hard (small-object AP).

## 10. Final conclusion

Under the frozen Week 6 protocol, the team advances **YOLO11n@640** as the selected lightweight detector. Ablations did not beat the control on available **validation** metrics; held-out **test** metrics are reported separately and were not used for selection.
""",
        encoding="utf-8",
    )

    # AI, contributions, quality checklist — static with verified checks
    (ROOT / "docs/AI_use.md").write_text(
        """# AI assistance disclosure

Generative AI tools were used to brainstorm the project scope, refine the research question, organize documentation, improve grammar, and assist with code planning. The team independently reviewed the dataset information, experimental design, implementation, results, and conclusions. No experimental results or performance metrics were fabricated by AI. The team remains responsible for implementing, verifying, interpreting, and presenting all project work.
""",
        encoding="utf-8",
    )

    (ROOT / "docs/team_contributions.md").write_text(
        """# Team contributions

## Ameya — data pipeline, conversion, splits, validation

- RDD2022 VOC→YOLO conversion and unsupported-label handling  
- Dataset audit / statistics scripts  
- Grouped leakage-safe 70/15/15 split + duplicate pHash enforcement  
- Week 3 split materialization and dataset validation  
- Data README / dataset card inputs  

## Prathmesh — training pipeline and baselines

- Shared training helpers (`src/training/baseline.py`)  
- YOLOv8n Week 3 baseline training & run cards  
- YOLO11n Week 4 strong baseline training  
- Baseline comparison tooling and Week 3/4 shell runners  

## Tejas — architecture changes and ablations

- YOLO11n P2 / attention / combined model YAMLs  
- Week 5 ablation configs and `train_week5_ablation.py`  
- Model registry / resolve helpers  
- Unit tests for custom model components  

## Vrishin — evaluation, benchmarking, error analysis

- Week 4–6 evaluation scripts (val/test labeling)  
- Confidence lock, final held-out evaluation, country analysis  
- Error analysis galleries and FP/FN CSV  
- Efficiency benchmarking and Week 6/7 packaging / reproduction  

## Collaboration

**Status:** this section was drafted with AI assistance from the repository structure and has **not** been verified by each team member. Each member must confirm or correct their own section before submission. The collaboration process described here has not been confirmed by the team.
""",
        encoding="utf-8",
    )

    repro_fail = sum(1 for r in repro if r.get("pass_fail") == "FAIL")

    def _is_zero(v) -> bool:
        try:
            return float(v) == 0.0
        except (TypeError, ValueError):
            return False

    repro_degenerate = bool(repro) and all(
        _is_zero(r.get("original_value")) and _is_zero(r.get("reproduced_value")) for r in repro
    )
    if not repro:
        repro_status = "Pending — run reproduce_main_result.py"
    elif repro_degenerate:
        repro_status = "Runs, but DEGENERATE: every compared value is 0.0, so this is not evidence of reproduction"
    elif repro_fail == 0:
        repro_status = "PASS (non-degenerate values)"
    else:
        repro_status = f"FAIL ({repro_fail} metrics)"
    (ROOT / "docs/week7_quality_checklist.md").write_text(
        f"""# Week 7 quality checklist

| Check | Status |
|---|---|
| Scripts have usage (`argparse` / shell headers) | Not audited script-by-script |
| Paths relative or via `src/paths.py` | Mitigated: `rdd2022.yaml` is absolute by necessity (Ultralytics), machine-generated, gitignored |
| No personal machine paths in **configs** | **Fails** for `configs/rdd2022.yaml` (generated per machine; gitignored) |
| Historical absolute paths in result CSV/JSON | Present; accepted as run logs, not used as configs |
| Raw dataset not committed | Yes (`.gitignore` excludes `data/raw/**`); no git commits exist yet to verify against history |
| Test metrics not used for selection | Yes per saved fields (`test_used_for_selection: false`) |
| Final test evaluation coverage | Original run used `--smoke` (FP/FN and error analysis on 12 of 50 images). **Re-computed on the full split** under amendment `week6_full_test_counts_v1`; original retained. See `docs/known_issues_and_resolutions.md`. |
| Results traceable to experiment IDs | Yes (`experiment_manifest.csv`) |
| Tables match saved CSV/JSON | Yes for packaged copies of saved files |
| Figures have titles/labels | Not verified by visual review at time of generation |
| README usable from clean env | Not verified at time of generation |
| Reproduction script runs | {repro_status} |
| AI disclosure present | Yes (`docs/AI_use.md`) |
| Team contributions present | Present but **not verified by team members** |
| Dataset links present | Yes (`data/README.md`); the data itself was never downloaded here |
| Official RDD2022 data used for results | **Fails**: all results use the synthetic fixture |

## Commands

```bash
bash scripts/setup_environment.sh
python scripts/check_environment.py
bash scripts/run_all_checks.sh
```
""",
        encoding="utf-8",
    )
    print("Wrote Week 7 markdown docs")


def main() -> int:
    write_experiment_manifest()
    write_docs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
