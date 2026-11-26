#!/usr/bin/env python3
"""Reproduce the primary Week 6 final-model test evaluation.

Loads the frozen protocol + selected checkpoint (does NOT retrain by default).
Compares reproduced Test metrics to saved Week 6 originals.

Does not change thresholds, splits, or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.week6_utils import (
    extract_val_metrics,
    f1_score,
    load_protocol,
    resolve_ckpt,
    to_float,
    write_csv,
    write_json,
)
from src.paths import ensure_legacy_fixture_data_yaml, legacy_fixture_splits_dir
from src.training.baseline import gpu_name, pick_device, software_versions

WEEK6 = ROOT / "results" / "week6"
WEEK7 = ROOT / "results" / "week7"
PROTOCOL = ROOT / "configs" / "final_evaluation.yaml"
LOCK = WEEK6 / "CONFIDENCE_THRESHOLD_LOCKED.json"

# Absolute tolerance for near-zero smoke metrics; relative for larger values
ABS_TOL = 1e-4
REL_TOL = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Week 6 final selected model test eval.")
    p.add_argument("--protocol", type=str, default=str(PROTOCOL))
    p.add_argument("--device", type=str, default="")
    p.add_argument("--split", type=str, default="test", choices=["test", "val"])
    p.add_argument(
        "--allow-retrain",
        action="store_true",
        help="If checkpoint missing, refuse by default; retrain is out of scope for frozen protocol.",
    )
    return p.parse_args()


def sha16(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_original_row(split_label: str) -> dict:
    path = WEEK6 / "final_test_results.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model_id") == "final_selected" and row.get("split_label") == split_label:
                return row
    return {}


def pass_fail(orig, repro) -> tuple[str, float | None, float | None, str]:
    if orig in (None, "", "Pending") or repro in (None, "", "Pending"):
        return "FAIL", None, None, "Missing original or reproduced value"
    try:
        o, r = float(orig), float(repro)
    except (TypeError, ValueError):
        return "FAIL", None, None, "Non-numeric metric"
    abs_diff = abs(r - o)
    rel_diff = abs_diff / abs(o) if abs(o) > 1e-12 else (0.0 if abs_diff <= ABS_TOL else float("inf"))
    # Pass if absolute close OR relative close when original is meaningful
    if abs_diff <= ABS_TOL:
        return "PASS", abs_diff, rel_diff if math.isfinite(rel_diff) else None, "Within absolute tolerance"
    if abs(o) > 1e-6 and rel_diff <= REL_TOL:
        return "PASS", abs_diff, rel_diff, "Within relative tolerance"
    reason = (
        f"abs_diff={abs_diff:.3g} exceeds tol={ABS_TOL}; "
        f"rel_diff={rel_diff if math.isfinite(rel_diff) else 'inf'}"
    )
    # Hardware / nondeterminism note for tiny values
    if abs(o) < 1e-5 and abs(r) < 1e-5:
        reason += " (both near-zero; smoke-scale metrics are unstable across devices)"
        if abs_diff <= 1e-3:
            return "PASS", abs_diff, rel_diff if math.isfinite(rel_diff) else None, reason + "; accepted as near-zero"
    return "FAIL", abs_diff, rel_diff if math.isfinite(rel_diff) else None, reason


def main() -> int:
    args = parse_args()
    WEEK7.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(Path(args.protocol))
    device = pick_device(args.device)
    gname = gpu_name(device)

    fm = protocol["final_model"]
    ckpt = resolve_ckpt(fm["checkpoint"])
    if ckpt is None:
        msg = f"Missing checkpoint: {fm['checkpoint']}"
        if args.allow_retrain:
            msg += " (--allow-retrain set, but automatic retrain is disabled to protect the freeze)."
        print(msg, file=sys.stderr)
        write_json(
            WEEK7 / "reproduction_status.json",
            {"status": "FAILED", "reason": msg, "timestamp": datetime.now(timezone.utc).isoformat()},
        )
        return 1

    # Locked conf
    if LOCK.is_file():
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        conf_op = float(lock["selected_conf"])
        conf_source = str(LOCK)
    else:
        conf_op = float(protocol.get("confidence_threshold", 0.25))
        conf_source = "protocol_yaml_fallback"

    imgsz = int(protocol.get("imgsz", 640))
    iou = float(protocol.get("iou_threshold", 0.7))
    conf_map = float(protocol.get("conf_for_map", 0.001))
    # The frozen Week 6 result is a LEGACY synthetic-fixture artifact.
    data_yaml = ensure_legacy_fixture_data_yaml()

    split = args.split
    split_label = "Test" if split == "test" else "Validation"
    original = load_original_row(split_label)

    split_hashes = {s: sha16(legacy_fixture_splits_dir() / f"{s}.csv") for s in ("train", "val", "test")}
    # Compare to Week 5 hashes if present
    w5 = ROOT / "results" / "week5" / "yolo11n_640" / "seed_0" / "metrics.json"
    w5_hashes = None
    if w5.is_file():
        w5_hashes = json.loads(w5.read_text(encoding="utf-8")).get("split_hashes")

    print(f"Reproducing final_selected on {split_label}")
    print(f"  ckpt={ckpt}")
    print(f"  data={data_yaml} imgsz={imgsz} conf_map={conf_map} iou={iou} conf_op={conf_op}")
    print(f"  device={device} gpu={gname}")
    print(f"  split_hashes={split_hashes}")

    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    res = model.val(
        data=str(data_yaml.resolve()),
        split=split,
        imgsz=imgsz,
        device=device,
        conf=conf_map,
        iou=iou,
        plots=False,
        verbose=False,
    )
    metrics = extract_val_metrics(res)

    reproduced = {
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }
    for c, v in metrics.get("per_class_ap50", {}).items():
        reproduced[f"ap50_{c}"] = v

    metric_names = [
        "map50",
        "map50_95",
        "precision",
        "recall",
        "f1",
        "ap50_D00",
        "ap50_D10",
        "ap50_D20",
        "ap50_D40",
    ]
    rows = []
    n_fail = 0
    for name in metric_names:
        orig_v = original.get(name, "Pending")
        repro_v = reproduced.get(name, "Pending")
        status, abs_d, rel_d, explain = pass_fail(orig_v, repro_v)
        if status == "FAIL":
            n_fail += 1
        rows.append(
            {
                "metric_name": name,
                "split_label": split_label,
                "original_value": orig_v,
                "reproduced_value": repro_v,
                "absolute_difference": abs_d if abs_d is not None else "Pending",
                "relative_difference": rel_d if rel_d is not None else "Pending",
                "pass_fail": status,
                "explanation": explain,
            }
        )

    def _is_zero(v) -> bool:
        try:
            return float(v) == 0.0
        except (TypeError, ValueError):
            return False

    # If every compared value is zero, PASS is trivially true and is NOT evidence of reproduction.
    degenerate = bool(rows) and all(
        _is_zero(r["original_value"]) and _is_zero(r["reproduced_value"]) for r in rows
    )
    if degenerate:
        print(
            "WARNING: all compared metrics are 0.0. The PASS result is degenerate and does not demonstrate reproduction.",
            file=sys.stderr,
        )

    write_csv(WEEK7 / "reproduction_results.csv", rows)
    write_json(
        WEEK7 / "reproduction_results.json",
        {
            "model_id": "final_selected",
            "checkpoint": str(ckpt),
            "split_label": split_label,
            "split": split,
            "imgsz": imgsz,
            "seed": fm.get("seed"),
            "confidence_threshold_operating": conf_op,
            "conf_for_map": conf_map,
            "iou_threshold": iou,
            "conf_source": conf_source,
            "device": device,
            "gpu": gname,
            "software": software_versions(),
            "split_hashes": split_hashes,
            "week5_split_hashes": w5_hashes,
            "split_hashes_match_week5": (split_hashes == w5_hashes) if w5_hashes else None,
            "original_source": str(WEEK6 / "final_test_results.csv"),
            "n_fail": n_fail,
            "degenerate_all_zero": degenerate,
            "abs_tol": ABS_TOL,
            "rel_tol": REL_TOL,
            "rows": rows,
            "reproduced_raw": reproduced,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_used_for_selection": False,
            "note": "Evaluation-only reproduction of frozen checkpoint; no retraining.",
        },
    )

    # Write / refresh reproduction report
    report = ROOT / "docs" / "reproduction_report.md"
    degenerate_warning = (
        [
            "> **Degenerate check:** every compared metric is 0.0. The PASS results below only show that the evaluation code runs deterministically on this machine. They do **not** demonstrate that a meaningful result was reproduced.",
            "",
        ]
        if degenerate
        else []
    )
    lines = [
        "# Reproduction report — primary final experiment",
        "",
        *degenerate_warning,
        f"_Generated: {datetime.now(timezone.utc).isoformat()}_",
        "",
        "## Experiment",
        "",
        f"- Model: **{fm.get('display_name')}** (`final_selected`)",
        f"- Checkpoint: `{fm.get('checkpoint')}`",
        f"- Protocol: `configs/final_evaluation.yaml` (`{protocol.get('protocol_id')}`)",
        f"- Split evaluated: **{split_label}** (`{split}`)",
        f"- imgsz={imgsz}, seed={fm.get('seed')}, conf_map={conf_map}, NMS IoU={iou}, operating conf={conf_op}",
        f"- Device: {device} / {gname}",
        f"- Software: {software_versions()}",
        "",
        "## Procedure",
        "",
        "1. Load frozen protocol and locked confidence (validation-selected).",
        "2. Confirm dataset split CSVs and hashes.",
        "3. Load the selected checkpoint (no retrain).",
        "4. Run Ultralytics `model.val` with the same imgsz / conf_for_map / IoU.",
        "5. Compare to Week 6 `final_test_results.csv` row for `final_selected`.",
        "",
        "## Split integrity",
        "",
        f"- Current hashes: `{split_hashes}`",
        f"- Week 5 recorded hashes: `{w5_hashes}`",
        f"- Match: **{(split_hashes == w5_hashes) if w5_hashes else 'Pending'}**",
        "",
        "## Comparison table",
        "",
        "| Metric | Original | Reproduced | Abs diff | Rel diff | Pass/Fail | Explanation |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['metric_name']} | {r['original_value']} | {r['reproduced_value']} | "
            f"{r['absolute_difference']} | {r['relative_difference']} | **{r['pass_fail']}** | {r['explanation']} |"
        )
    lines += [
        "",
        f"**Failures:** {n_fail} / {len(rows)}",
        "",
        "## Discrepancy notes",
        "",
        "- Smoke-trained checkpoints yield near-zero mAP; tiny absolute differences across MPS/CUDA/CPU are expected.",
        "- Nondeterminism (cuDNN, MPS, DataLoader workers) can change floating-point results slightly.",
        "- This script does **not** retune thresholds or change the model after observing results.",
        "- If checkpoint files are missing on another machine, copy `results/week*/**/weights/best.pt` or retrain under a **new** experiment ID — do not overwrite frozen Week 6 numbers.",
        "",
        "## Pass criterion",
        "",
        f"- Absolute tolerance `{ABS_TOL}` or relative tolerance `{REL_TOL}` when |original| > 1e-6.",
        "- Near-zero pairs (|o|<1e-5 and |r|<1e-5 with abs_diff≤1e-3) accepted with explanation.",
        "",
        f"CSV: `results/week7/reproduction_results.csv`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {WEEK7 / 'reproduction_results.csv'} (failures={n_fail})")
    print(f"Wrote {report}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
