#!/usr/bin/env python3
"""Write results/week5/model_selection_decision.md from val-only seed summaries.

Never reads or requires test metrics.
Also refreshes docs/week5_ablation_report.md result tables from CSVs.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.resolve import MODEL_REGISTRY

EXPERIMENTS = list(MODEL_REGISTRY.keys())


def pending(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return True
    s = str(v).strip()
    return s == "" or s.lower() == "pending"


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt(v, digits=4):
    if pending(v):
        return "Pending"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    week5 = ROOT / "results" / "week5"
    summaries = read_csv(week5 / "seed_summary.csv")
    rows = read_csv(week5 / "ablation_results.csv")
    eff = read_csv(week5 / "efficiency_results.csv")
    smap = {r["experiment"]: r for r in summaries}
    emap = {r["experiment"]: r for r in eff}

    # Select by highest mean val mAP50 among experiments with ≥1 completed seed
    candidates = []
    for e in EXPERIMENTS:
        s = smap.get(e)
        if not s:
            continue
        if pending(s.get("map50_mean")):
            continue
        candidates.append((e, float(s["map50_mean"]), s))

    decision_path = week5 / "model_selection_decision.md"
    now = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append("# Week 5 — Model selection decision (validation only)\n")
    lines.append(f"_Generated: {now}_\n")
    lines.append("## Selection rule\n")
    lines.append(
        "- Primary criterion: **highest mean validation mAP@0.5** across seeds 0/1/2 "
        "(or available seeds).\n"
        "- Secondary (ties / near-ties): small-object AP@0.5, then per-class AP on D00/D10/D40, "
        "then lower GFLOPs / latency.\n"
        "- Checkpoints were selected during training using **validation mAP@0.5 only**.\n"
        "- **No test-set metrics were inspected or used for this decision.**\n"
    )

    if not candidates:
        lines.append("## Selected final candidate\n")
        lines.append(
            "**Pending** — no completed validation mAP@0.5 results yet. "
            "Run `./scripts/run_week5.sh` (or `--smoke`) before locking a Week 6 candidate.\n"
        )
        lines.append("## Why no test results were used\n")
        lines.append(
            "Week 5 protocol forbids test evaluation for model selection. "
            "Final locked test evaluation is reserved for Week 6.\n"
        )
        decision_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {decision_path} (Pending — no completed runs)")
        _write_report(summaries, rows, eff, selected=None)
        return 0

    # Prefer non-smoke full runs when both exist; still select on available means.
    # Tie-break toward the YOLO11n@640 control when means are equal (e.g. all-zero smoke).
    candidates.sort(key=lambda t: (t[1], t[0] == "yolo11n_640"), reverse=True)
    best_exp, best_map, best_s = candidates[0]
    base_s = smap.get("yolo11n_640", {})
    base_map = base_s.get("map50_mean")
    improvement = best_s.get("improvement_map50_vs_baseline", "Pending")

    # Consistency across seeds
    seed_maps = []
    for r in rows:
        if r.get("experiment") == best_exp and not pending(r.get("map50")):
            seed_maps.append((int(r["seed"]), float(r["map50"])))
    consistent = "n/a"
    if len(seed_maps) >= 2:
        vals = [m for _, m in seed_maps]
        consistent = (
            f"yes — all {len(vals)} seeds within "
            f"{max(vals) - min(vals):.4f} absolute mAP@0.5"
            if (max(vals) - min(vals)) < 0.05
            else f"mixed — seed spread {max(vals) - min(vals):.4f} mAP@0.5; interpret cautiously"
        )
    elif len(seed_maps) == 1:
        consistent = "only one seed completed — treat as provisional"

    # Cost vs baseline from efficiency table or summary
    cost_bits = []
    be = emap.get("yolo11n_640", {})
    ee = emap.get(best_exp, {})
    for key, label in (
        ("parameter_count", "params"),
        ("gflops", "GFLOPs"),
        ("inference_latency_ms_mean", "latency_ms"),
        ("peak_gpu_memory_mb", "peak_GPU_MB"),
    ):
        b, a = be.get(key), ee.get(key)
        if not pending(b) and not pending(a):
            try:
                bv, av = float(b), float(a)
                delta = av - bv
                pct = (100.0 * delta / bv) if bv else float("nan")
                cost_bits.append(f"{label}: {av:.3g} (Δ {delta:+.3g}, {pct:+.1f}% vs baseline)")
            except (TypeError, ValueError):
                pass
    if not cost_bits:
        for key, label in (
            ("parameter_count_mean", "params"),
            ("gflops_mean", "GFLOPs"),
            ("inference_latency_ms_mean", "latency_ms"),
        ):
            v = best_s.get(key)
            if not pending(v):
                cost_bits.append(f"{label}: {fmt(v, 3)}")

    lines.append("## Selected final candidate\n")
    lines.append(f"**{best_exp}** — {MODEL_REGISTRY[best_exp]['label']}\n")
    lines.append("## Why it was selected\n")
    lines.append(
        f"- Highest available mean **validation mAP@0.5** among completed ablations: "
        f"**{fmt(best_map)}**.\n"
        f"- Baseline (yolo11n_640) mean val mAP@0.5: **{fmt(base_map)}**.\n"
        f"- Improvement vs baseline: **{fmt(improvement)}** "
        f"(negative means the candidate is worse — reported honestly).\n"
        f"- Architecture / factors: `{MODEL_REGISTRY[best_exp]['architecture']}` / "
        f"`{MODEL_REGISTRY[best_exp]['factors']}`.\n"
    )
    if best_exp == "yolo11n_640":
        lines.append(
            "- No ablation beat the YOLO11n@640 control on mean val mAP@0.5 "
            "(or only the control has completed runs). The baseline remains the candidate.\n"
        )

    lines.append("## Validation metrics supporting the decision\n")
    lines.append(
        f"| Metric | Value |\n|---|---|\n"
        f"| map50_mean | {fmt(best_s.get('map50_mean'))} |\n"
        f"| map50_std | {fmt(best_s.get('map50_std'))} |\n"
        f"| map50_best / worst | {fmt(best_s.get('map50_best'))} / {fmt(best_s.get('map50_worst'))} |\n"
        f"| map50_95_mean | {fmt(best_s.get('map50_95_mean'))} |\n"
        f"| small_object_ap_mean | {fmt(best_s.get('small_object_ap_mean'))} |\n"
        f"| ap_D00 / D10 / D20 / D40 mean | "
        f"{fmt(best_s.get('ap_D00_mean'))} / {fmt(best_s.get('ap_D10_mean'))} / "
        f"{fmt(best_s.get('ap_D20_mean'))} / {fmt(best_s.get('ap_D40_mean'))} |\n"
        f"| n_seeds_completed | {best_s.get('n_seeds_completed', 'Pending')} |\n"
    )

    lines.append("## Consistency across seeds\n")
    lines.append(f"- {consistent}\n")
    if seed_maps:
        lines.append("- Per-seed val mAP@0.5: " + ", ".join(f"seed {s}={m:.4f}" for s, m in seed_maps) + "\n")

    lines.append("## Computational cost added\n")
    if cost_bits:
        for b in cost_bits:
            lines.append(f"- {b}\n")
    else:
        lines.append("- Pending (run `scripts/measure_efficiency.py`).\n")

    lines.append("## Why no test results were used\n")
    lines.append(
        "Per the project protocol and Week 5 objective, the test split is locked until Week 6. "
        "All checkpoint selection used validation mAP@0.5 only; this decision file was written "
        "from `seed_summary.csv` / `ablation_results.csv` validation aggregates exclusively. "
        "`evaluate_week5_ablation.py` refuses `--split test` without `--allow-test`.\n"
    )

    lines.append("## Ranking (mean val mAP@0.5)\n")
    lines.append("| Rank | Experiment | map50_mean | Δ vs baseline |\n|---|---|---|---|\n")
    for i, (e, m, s) in enumerate(candidates, 1):
        lines.append(
            f"| {i} | {e} | {fmt(m)} | {fmt(s.get('improvement_map50_vs_baseline'))} |\n"
        )
    pending_exps = [e for e in EXPERIMENTS if e not in {c[0] for c in candidates}]
    if pending_exps:
        lines.append("\nPending (no val mAP yet): " + ", ".join(pending_exps) + "\n")

    # Honesty note for smoke
    smoke_notes = []
    for r in rows:
        if r.get("experiment") == best_exp and "smoke" in str(r.get("status", "")).lower():
            smoke_notes.append(r)
    if smoke_notes or any("smoke" in str(r.get("status", "")).lower() for r in rows):
        lines.append("\n## Caveat\n")
        lines.append(
            "Some or all runs are **smoke** (short epoch budget). "
            "Treat the selected candidate as **provisional** until full 100-epoch × seeds 0/1/2 "
            "results replace Pending / smoke rows.\n"
        )
    if best_map < 0.01:
        lines.append("\n## Honesty note (near-zero validation mAP)\n")
        lines.append(
            "Completed smoke runs currently show ~0 validation mAP@0.5 (fixture / 1–2 epoch budget). "
            "This does **not** support claiming that P2, attention, or 800 px improves accuracy. "
            "The control `yolo11n_640` is retained as the provisional Week 6 candidate until "
            "full multi-seed validation results are available. Seeds 1 and 2 remain **Pending**.\n"
        )
    if best_map < 0.01:
        lines.append("\n## Honesty note (near-zero validation mAP)\n")
        lines.append(
            "Completed smoke runs currently show ~0 validation mAP@0.5 (fixture / 1–2 epoch budget). "
            "This does **not** support claiming that P2, attention, or 800 px improves accuracy. "
            "The control `yolo11n_640` is retained as the provisional Week 6 candidate until "
            "full multi-seed validation results are available. Seeds 1 and 2 remain **Pending**.\n"
        )

    decision_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {decision_path} → selected {best_exp} (map50_mean={fmt(best_map)})")
    _write_report(summaries, rows, eff, selected=best_exp)
    return 0


def _write_report(summaries, rows, eff, selected):
    """Create / refresh docs/week5_ablation_report.md with live tables when available."""
    report = ROOT / "docs" / "week5_ablation_report.md"
    smap = {r["experiment"]: r for r in summaries}
    emap = {r["experiment"]: r for r in eff}

    def row_metric(e, key):
        return fmt(smap.get(e, {}).get(key))

    table_rows = []
    for e in EXPERIMENTS:
        table_rows.append(
            f"| {e} | {MODEL_REGISTRY[e]['label']} | {MODEL_REGISTRY[e]['imgsz_default']} | "
            f"{row_metric(e, 'map50_mean')} | {row_metric(e, 'map50_95_mean')} | "
            f"{row_metric(e, 'precision_mean')} | {row_metric(e, 'recall_mean')} | "
            f"{row_metric(e, 'f1_mean')} | {row_metric(e, 'small_object_ap_mean')} | "
            f"{row_metric(e, 'improvement_map50_vs_baseline')} |"
        )

    per_class = []
    for e in EXPERIMENTS:
        per_class.append(
            f"| {e} | {row_metric(e, 'ap_D00_mean')} | {row_metric(e, 'ap_D10_mean')} | "
            f"{row_metric(e, 'ap_D20_mean')} | {row_metric(e, 'ap_D40_mean')} |"
        )

    eff_rows = []
    for e in EXPERIMENTS:
        er = emap.get(e, {})
        sr = smap.get(e, {})
        params = er.get("parameter_count", sr.get("parameter_count_mean", "Pending"))
        gflops = er.get("gflops", sr.get("gflops_mean", "Pending"))
        lat = er.get("inference_latency_ms_mean", sr.get("inference_latency_ms_mean", "Pending"))
        mem = er.get("peak_gpu_memory_mb", "Pending")
        eff_rows.append(
            f"| {e} | {fmt(params, 0) if not pending(params) else 'Pending'} | "
            f"{fmt(gflops, 2)} | {fmt(lat, 2)} | {fmt(mem, 1)} |"
        )

    selected_txt = selected or "Pending"
    body = f"""# Week 5 — Ablation report

Project: Road Damage Detection Using Lightweight Object Detectors  
Dataset: RDD2022 (grouped leakage-safe 70/15/15, `split_id=grouped_70_15_15_seed6845`)  
Classes: D00, D10, D20, D40

## 1. Week 5 objective

Evaluate whether lightweight architectural and input-resolution changes improve road-damage
detection—especially for small cracks and potholes—without an unreasonable computational increase.
Checkpoint selection uses **validation mAP@0.5 only**. The test set is **not** used for
selection or tuning (deferred to Week 6).

## 2. Ablation research questions

1. Does a **P2 (stride-4) detection head** improve thin crack / small damage AP at 640 px?
2. Does a **lightweight neck attention** block (C2PSA on P3) help without large cost?
3. Does raising input resolution to **800 px** help more than architectural changes alone?
4. Does the **combined** model (P2 + attention + 800) justify its extra compute?
5. Which candidate should advance to Week 6 final evaluation **based on validation only**?

## 3. Description of each model variant

| Experiment | Variant | Design |
|---|---|---|
| `yolo11n_640` | YOLO11n @ 640 | Stock YOLO11n (P3/P4/P5). Control. |
| `yolo11n_p2_640` | + P2 head @ 640 | Same backbone; Detect on P2/P3/P4/P5 (`configs/models/yolo11n-p2.yaml`). |
| `yolo11n_attention_640` | + attention @ 640 | C2PSA on P3 neck feature (`configs/models/yolo11n-attn.yaml`). |
| `yolo11n_800` | YOLO11n @ 800 | Stock YOLO11n; imgsz=800; batch 8 (documented memory policy). |
| `combined_800` | P2 + attn @ 800 | `yolo11n-p2-attn.yaml` + imgsz=800; batch 8. |

Attention uses Ultralytics built-in **C2PSA** (no custom CUDA ops). P2 follows the YOLO11 +
yolov8-p2 head pattern.

## 4. Controlled variables

- Identical dataset split (`configs/rdd2022.yaml` → Week 3 materialized dirs)
- Identical class mapping (D00=0 … D40=3)
- Same optimizer / LR schedule family (`optimizer: auto`, `lr0=0.01`)
- Same augmentation recipe
- Same early-stopping patience family (reduced only under `--smoke`)
- AMP when CUDA is available
- Seeds **0, 1, 2** for principal comparisons (smoke may use seed 0 only)
- Best checkpoint = Ultralytics best on **validation** fitness / mAP@0.5

Documented adjustment: batch **16→8** at 800 px (`training_defaults.yaml` `batch_800`).

## 5. Experimental settings

| Item | Value |
|---|---|
| Data config | `configs/rdd2022.yaml` |
| Split | grouped 70/15/15, seed 6845 |
| Epochs (full) | 100 (smoke: ≤2) |
| Batch @640 / @800 | 16 / 8 (smoke: ≤4 / ≤2) |
| Comparison seeds | 0, 1, 2 |
| Selection metric | val mAP@0.5 |
| Test for selection | **No** |

Configs: `configs/week5_*.yaml`. Train: `scripts/train_week5_ablation.py`.

## 6. Results table

Validation metrics (mean over completed seeds). Unfinished → **Pending**.

| Experiment | Variant | imgsz | mAP50 | mAP50-95 | P | R | F1 | small-AP | Δ mAP50 vs baseline |
|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(table_rows)}

Raw rows: `results/week5/ablation_results.csv`, `results/week5/seed_summary.csv`.

## 7. Per-class comparison

| Experiment | AP D00 | AP D10 | AP D20 | AP D40 |
|---|---|---|---|---|
{chr(10).join(per_class)}

Figure: `figures/week5/per_class_ap_comparison.png`.

## 8. Small-object detection analysis

Small-object AP@0.5 uses GT boxes with normalized area &lt; 0.02 (see `evaluate_week5_ablation.py`).
Values are **Pending** until `--recompute-val` completes for each seed checkpoint.

Figure: `figures/week5/small_object_ap_comparison.png`.

Honest reading: if small-AP does not rise with P2 / 800 px, do not claim small-object gains.

## 9. Accuracy-versus-efficiency analysis

| Experiment | Params | GFLOPs | Latency (ms) | Peak GPU MB |
|---|---|---|---|---|
{chr(10).join(eff_rows)}

Figure: `figures/week5/accuracy_efficiency_tradeoff.png`.  
Source: `results/week5/efficiency_results.csv`.

## 10. Qualitative prediction examples

Validation overlays (not test) are written under `figures/week5/predictions/<experiment>/`
when seed-0 checkpoints exist. Absence of images means the corresponding train run is still Pending.

## 11. Limitations

- All Week 5 runs used the **synthetic fixture**, not the official RDD2022 download, and ran 1 epoch in smoke mode. Absolute mAP is ~0 and not meaningful; no relative trend in these runs supports a conclusion about the variants.
- Smoke runs understate convergence; do not overstate smoke improvements.
- Small-object AP is an approximation (area filter), not COCO-official small AP.
- Combined model confounds three factors; single-factor configs isolate effects.
- MPS/CPU runs may omit peak CUDA memory.

## 12. Candidate model for final evaluation

**{selected_txt}**

See `results/week5/model_selection_decision.md` for the locked rationale.

## 13. Justification (validation only)

Selection uses mean **validation** mAP@0.5 (and secondaries listed in the decision file).
Test metrics were not computed for selection. Week 6 will run the locked candidate (and
baselines) on the held-out test split once.

---

Reproduce:

```bash
./scripts/run_week5.sh --smoke   # quick
./scripts/run_week5.sh           # full seeds 0 1 2
```
"""
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(body, encoding="utf-8")
    print(f"Wrote {report}")


if __name__ == "__main__":
    raise SystemExit(main())
