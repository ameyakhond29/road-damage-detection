#!/usr/bin/env python3
"""LEGACY (Week 1, synthetic fixture) — Convert VOC XML to YOLO format.

Defaults point at the synthetic fixture. For the official RDD2022 dataset use
scripts/convert_voc_to_yolo.py (or scripts/run_official_dataset_pipeline.sh).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import DEFAULT_SEED, EXCLUDE_COUNTRIES
from src.dataset.convert import convert_dataset
from src.paths import fixture_raw_dir, legacy_fixture_processed_dir, resolve_raw_dir
from src.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert RDD2022 VOC XML → YOLO labels (D00=0..D40=3)."
    )
    p.add_argument("--raw-dir", type=str, default=str(fixture_raw_dir()))
    p.add_argument(
        "--processed-dir",
        type=str,
        default=str(legacy_fixture_processed_dir()),
        help="Output directory for YOLO images/labels (originals untouched)",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--include-china-drone",
        action="store_true",
        help="Keep China_Drone (excluded by default)",
    )
    p.add_argument(
        "--symlink-images",
        action="store_true",
        help="Symlink images instead of copying (falls back to copy on failure)",
    )
    p.add_argument(
        "--max-images-per-country",
        type=int,
        default=None,
        help="Optional cap per country (useful for conversion smoke tests)",
    )
    p.add_argument(
        "--countries",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of country folder names",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    try:
        raw_dir = resolve_raw_dir(args.raw_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    exclude = set() if args.include_china_drone else set(EXCLUDE_COUNTRIES)
    print(f"Raw dataset: {raw_dir}")
    print(f"Processed output: {args.processed_dir}")
    print(f"Excluding countries: {sorted(exclude) or '(none)'}")

    result = convert_dataset(
        raw_dir,
        Path(args.processed_dir),
        exclude_countries=exclude,
        copy_images=not args.symlink_images,
        symlink_images=args.symlink_images,
        countries=args.countries,
        max_images_per_country=args.max_images_per_country,
    )

    print("\n=== Conversion summary ===")
    print(f"Images converted: {result['n_converted']}")
    print(f"Negative examples (empty labels): {result['n_negatives']}")
    print(f"Unsupported boxes removed: {result['removed_unsupported_total']}")
    if result["removed_unsupported"]:
        for lbl, cnt in sorted(result["removed_unsupported"].items()):
            print(f"  {lbl}: {cnt}")
    print(f"Manifest: {result['manifest_path']}")

    out = Path(args.processed_dir) / "conversion_summary.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Summary JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
