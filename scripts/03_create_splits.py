#!/usr/bin/env python3
"""LEGACY (Week 1, synthetic fixture) — Leakage-safe 70/15/15 split generation.

Defaults point at the synthetic fixture. Official RDD2022 splits are created by
scripts/create_week3_splits.py --config configs/rdd2022.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import DEFAULT_SEED, GROUP_BLOCK_SIZE
from src.dataset.splits import create_leakage_safe_splits
from src.paths import legacy_fixture_processed_dir, project_root
from src.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create leakage-safe train/val/test splits (block groups + phash)."
    )
    p.add_argument("--processed-dir", type=str, default=str(legacy_fixture_processed_dir()))
    p.add_argument(
        "--data-dir",
        type=str,
        default=str(project_root() / "results" / "fixture_runs" / "week1"),
        help="Where to write train_split.csv / val_split.csv / test_split.csv",
    )
    p.add_argument("--results-dir", type=str,
                   default=str(project_root() / "results" / "fixture_runs" / "week1"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--block-size", type=int, default=GROUP_BLOCK_SIZE)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument(
        "--max-hamming",
        type=int,
        default=0,
        help="Max perceptual-hash Hamming distance for near-duplicates (0=exact phash)",
    )
    p.add_argument(
        "--skip-phash",
        action="store_true",
        help="Skip perceptual-hash duplicate detection (not recommended)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    processed = Path(args.processed_dir)
    if not (processed / "images").is_dir():
        print(
            f"Processed dataset not found at {processed}.\n"
            "Run: python scripts/02_convert_annotations.py",
            file=sys.stderr,
        )
        return 1

    print(f"Processed dir: {processed}")
    print(f"Seed: {args.seed} | block_size: {args.block_size}")

    result = create_leakage_safe_splits(
        processed,
        Path(args.data_dir),
        Path(args.results_dir),
        seed=args.seed,
        block_size=args.block_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_hamming=args.max_hamming,
        skip_phash=args.skip_phash,
    )

    print("\n=== Split summary ===")
    for k, v in result["counts"].items():
        print(f"  {k}: {v}")
    print("Wrote:")
    for k, v in result["paths"].items():
        print(f"  {k}: {v}")

    summary_path = Path(args.results_dir) / "split_summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
