#!/usr/bin/env python3
"""Validate the processed official RDD2022 YOLO dataset.

Checks: every image has a label (empty allowed), 5 values per line, class ids 0–3,
normalized coordinates, w/h > 0, boxes inside the image, images readable, country
present, no orphan labels, and — once data/splits/*.csv exist for the official
dataset — no duplicate/near-duplicate/group leakage across train/val/test.

Outputs: results/official_rdd2022/validation_report.{json,csv}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.official import (  # noqa: E402
    default_workers,
    excluded_folders,
    load_dataset_config,
    official_paths,
    validate_official,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate official RDD2022 YOLO data.")
    p.add_argument("--config", default=str(ROOT / "configs" / "rdd2022.yaml"))
    p.add_argument("--data-root", default=None, help="Processed root (default data/processed)")
    p.add_argument("--out-dir", default=None, help="Default results/official_rdd2022")
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--skip-image-check", action="store_true")
    p.add_argument("--require-splits", action="store_true",
                   help="Fail if official split CSVs are missing (use after create_week3_splits.py)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_dataset_config(args.config)
    paths = official_paths(cfg, args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else paths["results"]
    processed = paths["processed"]
    if not (processed / "images").is_dir():
        print(f"Processed dataset not found at {processed}.\n"
              "Run: python scripts/convert_voc_to_yolo.py --data-root data/raw/RDD2022", file=sys.stderr)
        return 1

    print(f"Validating: {processed}")
    res = validate_official(processed, paths["splits"], exclude=excluded_folders(cfg), workers=args.workers,
                            check_images=not args.skip_image_check, require_splits=args.require_splits,
                            max_hamming=int(cfg.get("near_duplicate_max_hamming", 0)),
                            purge_hamming=int(cfg.get("near_duplicate_purge_hamming", 0)))
    s = res["summary"]
    write_json(out_dir / "validation_report.json", s)
    write_csv(out_dir / "validation_report.csv", res["rows"])

    print("\n=== Validation summary ===")
    print(f"  images={s['n_images']} boxes={s['n_boxes']} negatives={s['n_negatives_empty_labels']} "
          f"orphan_labels={s['n_orphan_labels']}")
    print(f"  class_counts={s['class_counts']}")
    print(f"  issue_counts={s['issue_counts']}")
    for k, v in s["checks"].items():
        print(f"  {k}: {v}")
    sc = s["split_check"]
    if sc["status"] == "checked":
        lk = sc["leakage"]
        print(f"  split rows={sc['n_rows']} unassigned={sc['processed_images_not_in_any_split']} "
              f"dup_clusters={lk['duplicate_clusters']} crossing={lk['duplicate_clusters_crossing_splits']}")
    else:
        print(f"  split leakage: {sc['status']} ({sc.get('reason')})")
    print(f"Wrote {out_dir / 'validation_report.json'}")
    print(f"Wrote {out_dir / 'validation_report.csv'}")
    if s["passed"]:
        print("VALIDATION PASSED")
        return 0
    print("VALIDATION FAILED — see validation_report.csv", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
