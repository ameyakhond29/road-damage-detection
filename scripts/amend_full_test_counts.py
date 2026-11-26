#!/usr/bin/env python3
"""Protocol amendment: full-coverage FP/FN and country counts on the test split.

Background (see docs/known_issues_and_resolutions.md, item B1):
the Week 6 final test run was launched with --smoke. Its mAP/precision/recall/F1
cover the full test split, but its FP/FN counts and per-country rows covered only
a subset (first 12 images; 8 images per country).

This script re-reads the test split ONCE more, with the SAME frozen checkpoint,
the SAME operating confidence (validation-locked), NMS IoU, and matching IoU as
Week 6. It does not retrain, does not change any threshold, and does not touch
any selection. Outputs go to a separate folder; the original smoke-run files
are NOT overwritten.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.week6_utils import load_protocol, resolve_ckpt, write_csv, write_json  # noqa: E402
from src.paths import (  # noqa: E402
    ensure_legacy_fixture_data_yaml,
    legacy_fixture_processed_dir,
    legacy_fixture_splits_dir,
)
from src.training.baseline import pick_device  # noqa: E402

WEEK6 = ROOT / "results" / "week6"
OUT = WEEK6 / "amendment_full_test_counts"
LOCK = WEEK6 / "CONFIDENCE_THRESHOLD_LOCKED.json"


def _load_run_final_evaluation():
    """Import count_fp_fn / country_metrics from the Week 6 script without running main()."""
    spec = importlib.util.spec_from_file_location("run_final_evaluation", ROOT / "scripts" / "run_final_evaluation.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_original_smoke_counts() -> dict:
    path = WEEK6 / "final_test_results.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model_id") == "final_selected" and row.get("split") == "test":
                return {
                    "n_detections": row.get("n_detections"),
                    "false_positive_count": row.get("false_positive_count"),
                    "false_negative_count": row.get("false_negative_count"),
                    "true_positive_count": row.get("true_positive_count"),
                    "localization_poor_count": row.get("localization_poor_count"),
                    "fn_small_count": row.get("fn_small_count"),
                }
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-coverage test counts (protocol amendment).")
    parser.add_argument("--protocol", type=str, default=str(ROOT / "configs" / "final_evaluation.yaml"))
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    from ultralytics import YOLO

    rpe = _load_run_final_evaluation()
    protocol = load_protocol(Path(args.protocol))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    conf = float(lock["selected_conf"])
    iou_nms = float(protocol.get("iou_threshold", 0.7))
    imgsz = int(protocol.get("imgsz", 640))
    ckpt = resolve_ckpt(protocol["final_model"]["checkpoint"])
    if ckpt is None:
        print("Missing final checkpoint", file=sys.stderr)
        return 1

    device = pick_device(args.device)
    processed = legacy_fixture_processed_dir()
    data_yaml = ensure_legacy_fixture_data_yaml()
    model = YOLO(str(ckpt))

    print(f"Full-test recount: ckpt={ckpt} conf={conf} iou_nms={iou_nms} imgsz={imgsz} device={device}")
    tot, pred_rows = rpe.count_fp_fn(
        model,
        processed,
        "test",
        imgsz,
        device,
        conf,
        iou_thr=0.5,
        max_images=None,  # FULL split: this is the whole point of the amendment
    )
    print(f"  images read: {len(list((processed / 'images' / 'test').iterdir()))}")

    country_rows = rpe.country_metrics(
        model,
        processed,
        legacy_fixture_splits_dir() / "test.csv",
        "test",
        "Test",
        imgsz,
        device,
        conf,
        data_yaml,
        smoke=False,  # FULL split: no per-country cap
    )

    OUT.mkdir(parents=True, exist_ok=True)
    counts = {
        "scope": "full_test_split",
        "n_test_images": len(list((processed / "images" / "test").iterdir())),
        "n_detections": tot["n_detections"],
        "false_positive_count": tot["fp"],
        "false_negative_count": tot["fn"],
        "true_positive_count": tot["tp"],
        "localization_poor_count": tot["localization_poor"],
        "fn_small_count": tot["fn_small"],
        "n_gt_objects": tot["n_gt"],
    }
    original = _read_original_smoke_counts()
    record = {
        "amendment_id": "week6_full_test_counts_v1",
        "reason": "Week 6 final test run was launched with --smoke. FP/FN counts and error analysis covered a subset (12 images; country rows capped at 8 per country). This re-reads the full test split for counts only.",
        "model_id": "final_selected",
        "checkpoint": str(ckpt),
        "confidence_threshold_operating": conf,
        "confidence_source": str(LOCK.relative_to(ROOT)),
        "iou_nms": iou_nms,
        "iou_match": 0.5,
        "imgsz": imgsz,
        "device": device,
        "model_changed": False,
        "thresholds_changed": False,
        "selection_changed": False,
        "retrained": False,
        "original_smoke_counts": original,
        "full_test_counts": counts,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT / "full_test_counts.json", record)
    write_csv(OUT / "full_test_country_results.csv", country_rows)
    write_csv(OUT / "full_test_predictions.csv", pred_rows)
    print(f"Wrote {OUT / 'full_test_counts.json'}")
    print(f"  full split: FP={counts['false_positive_count']} FN={counts['false_negative_count']} TP={counts['true_positive_count']} (GT={counts['n_gt_objects']})")
    print(f"  original smoke-scope: {original}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
