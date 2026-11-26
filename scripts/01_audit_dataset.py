#!/usr/bin/env python3
"""Audit the official RDD2022 dataset (raw VOC) — or a processed YOLO tree.

Outputs (official): results/official_rdd2022/audit/
  audit_summary.json, class_distribution.csv, box_statistics.csv,
  per_country_summary.csv, issues.csv, exact_duplicates.csv, near_duplicates.csv,
  image_index.csv (md5 / pHash / decode cache reused by the split step)

Use ``--dataset fixture`` only for debugging; fixture audits go to
results/fixture_runs/audit/ and never into results/official_rdd2022/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.official import (  # noqa: E402
    audit_processed,
    default_workers,
    excluded_folders,
    load_dataset_config,
    looks_processed,
    official_paths,
    resolve_raw_root,
    run_official_audit,
)
from src.paths import DATASET_CHOICES, resolve_dataset_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit RDD2022 images/annotations.")
    p.add_argument("--data-root", default=None,
                   help="Dataset root (searched recursively). Default: $RDD2022_ROOT or data/raw/RDD2022")
    p.add_argument("--dataset", choices=DATASET_CHOICES, default="official")
    p.add_argument("--config", default=str(ROOT / "configs" / "rdd2022.yaml"))
    p.add_argument("--out-dir", default=None, help="Default results/official_rdd2022/audit")
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--skip-image-check", action="store_true",
                   help="Skip full decode / md5 / pHash (no corrupt or duplicate detection)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_dataset_config(args.config)
    exclude = excluded_folders(cfg)
    max_ham = int(cfg.get("near_duplicate_max_hamming", 0))

    if args.dataset == "fixture":
        out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "fixture_runs" / "audit"
    else:
        out_dir = Path(args.out_dir) if args.out_dir else official_paths(cfg)["results"] / "audit"

    candidate = Path(args.data_root).expanduser().resolve() if args.data_root else None
    if candidate is not None and candidate.is_dir() and looks_processed(candidate):
        print(f"Data root looks like PROCESSED YOLO data: {candidate}")
        s = audit_processed(candidate, out_dir)
        print(f"data_kind={s['data_kind']} per_country={s['per_country']} class_counts={s['class_counts']}")
        print(f"Wrote {out_dir / 'audit_summary.json'}")
        return 0

    try:
        if args.dataset == "fixture":
            raw = resolve_dataset_root(args.data_root, "fixture")
        else:
            raw = resolve_raw_root(cfg, args.data_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Dataset:        {args.dataset}")
    print(f"Raw data root:  {raw}")
    print(f"Output dir:     {out_dir}")
    print(f"Excluded:       {sorted(exclude)} (counted, not used for modeling)")
    s = run_official_audit(raw, out_dir, exclude=exclude, workers=args.workers,
                           check_images=not args.skip_image_check, max_hamming=max_ham)

    t, m = s["totals_all_countries"], s["totals_modeling_set"]
    print("\n=== RDD2022 audit (raw VOC) ===")
    print(f"Countries: {', '.join(s['countries'])}")
    print(f"Labeled train images (all / modeling set): {t.get('train_images', 0)} / {m.get('train_images', 0)}")
    print(f"Train XML annotations (all / modeling):    {t.get('train_xmls', 0)} / {m.get('train_xmls', 0)}")
    print(f"Official test images (unlabeled, unused):  {t.get('official_test_images_unlabeled', 0)}")
    print(f"Negatives (no supported boxes, modeling):  {m.get('negatives_no_supported_boxes', 0)}")
    print("Supported class box counts (modeling set):")
    for k, v in s["class_box_counts_modeling_set"].items():
        print(f"  {k}: {v}")
    print(f"Unsupported labels (modeling set): {s['unsupported_label_counts_modeling_set']}")
    print("Issues:")
    for k in ("xml_without_image", "image_without_xml", "malformed_xml", "malformed_objects",
              "corrupt_images", "invalid_boxes", "box_partially_out_of_bounds", "xml_size_mismatch"):
        print(f"  {k}: all={t.get(k, 0)} modeling={m.get(k, 0)}")
    print(f"Duplicates: {s['duplicates']}")
    print("\nWrote:")
    for k, v in s["outputs"].items():
        print(f"  {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
