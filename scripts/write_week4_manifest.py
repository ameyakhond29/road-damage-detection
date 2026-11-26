#!/usr/bin/env python3
"""Write results/week4/run_manifest.json for reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import DEFAULT_SEED
from src.training.baseline import software_versions, write_json


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--started-at", type=str, default="")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--commands",
        type=str,
        nargs="*",
        default=[],
        help="Commands executed during the Week 4 run",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = args.started_at or datetime.now(timezone.utc).isoformat()
    ended = datetime.now(timezone.utc).isoformat()

    split_hashes = {
        name: file_hash(ROOT / "data" / "splits" / name)
        for name in ("train.csv", "val.csv", "test.csv", "split_summary.json")
    }

    manifest = {
        "week": 4,
        "project": "Road Damage Detection Using Lightweight Object Detectors",
        "started_at": started,
        "ended_at": ended,
        "smoke": bool(args.smoke),
        "seed": DEFAULT_SEED,
        "git_commit": git_commit(),
        "repository_root": str(ROOT),
        "platform": platform.platform(),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "software": software_versions(),
        },
        "dataset_split": {
            "split_id": "grouped_70_15_15_seed6845",
            "files": {
                "train": "data/splits/train.csv",
                "val": "data/splits/val.csv",
                "test": "data/splits/test.csv",
            },
            "sha256": split_hashes,
            "unchanged_from_week3": True,
        },
        "configs": {
            "data": "configs/rdd2022.yaml",
            "baseline_yolov8n": "configs/baseline_yolov8n.yaml",
            "baseline_yolo11n": "configs/baseline_yolo11n.yaml",
        },
        "models": {
            "reference": "yolov8n.pt",
            "strong": "yolo11n.pt",
        },
        "commands_executed": args.commands
        or [
            "python scripts/train_yolo11_baseline.py",
            "python scripts/compare_baselines.py",
            "python scripts/evaluate_week4.py",
        ],
        "outputs": {
            "yolo11n_baseline": "results/week4/yolo11n_baseline/",
            "baseline_comparison_csv": "results/week4/baseline_comparison.csv",
            "baseline_comparison_json": "results/week4/baseline_comparison.json",
            "evaluation": "results/week4/evaluation/",
            "figures": "figures/week4/",
            "progress_report": "docs/week4_progress_report.md",
        },
        "protocol": {
            "checkpoint_selection": "validation only",
            "test_used_for_selection": False,
        },
    }
    out = ROOT / "results" / "week4" / "run_manifest.json"
    write_json(out, manifest)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
