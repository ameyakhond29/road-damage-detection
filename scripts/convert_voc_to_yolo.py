#!/usr/bin/env python3
"""Convert official RDD2022 Pascal VOC annotations to YOLO format.

* Mapping D00=0, D10=1, D20=2, D40=3; other labels are dropped and counted.
* China_Drone (``excluded_folders`` in configs/rdd2022.yaml) is excluded.
* Labeled ``train`` folders only; images without XML and the unlabeled official
  ``test`` images are not used.
* Negatives (XML without supported boxes) get an empty label file.
* Invalid boxes (xmax<=xmin, ymax<=ymin, fully outside the image) are rejected.
* Original JPG/XML are never modified; processed images are hardlinks to them
  (fallback: symlink, then copy).

Outputs: data/processed/{images,labels}/<country>/, data/processed/manifest.csv,
results/official_rdd2022/{conversion_manifest.json,conversion_manifest.csv,
unsupported_labels.csv,conversion_issues.csv}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CLASS_TO_ID  # noqa: E402
from src.dataset.official import (  # noqa: E402
    convert_official,
    excluded_folders,
    load_dataset_config,
    official_paths,
    resolve_raw_root,
)
from src.paths import DATASET_CHOICES, legacy_fixture_processed_dir, resolve_dataset_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RDD2022 VOC → YOLO (D00=0, D10=1, D20=2, D40=3).")
    p.add_argument("--data-root", default=None,
                   help="Raw RDD2022 root. Default: $RDD2022_ROOT or data/raw/RDD2022")
    p.add_argument("--output-dir", default=None, help="Processed root. Default: data/processed")
    p.add_argument("--dataset", choices=DATASET_CHOICES, default="official")
    p.add_argument("--config", default=str(ROOT / "configs" / "rdd2022.yaml"))
    p.add_argument("--results-dir", default=None, help="Default results/official_rdd2022")
    p.add_argument("--link-mode", choices=("hardlink", "symlink", "copy"), default="hardlink")
    p.add_argument("--countries", nargs="*", default=None)
    p.add_argument("--max-images-per-country", type=int, default=None, help="Debug only")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_dataset_config(args.config)
    exclude = excluded_folders(cfg)
    paths = official_paths(cfg, args.output_dir)
    try:
        if args.dataset == "fixture":
            raw = resolve_dataset_root(args.data_root, "fixture")
            out = Path(args.output_dir) if args.output_dir else ROOT / "results" / "fixture_runs" / "processed"
            results = Path(args.results_dir) if args.results_dir else ROOT / "results" / "fixture_runs"
        else:
            raw = resolve_raw_root(cfg, args.data_root)
            out = paths["processed"]
            results = Path(args.results_dir) if args.results_dir else paths["results"]
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.dataset == "official" and out.resolve() == legacy_fixture_processed_dir().resolve():
        print("ERROR: refusing to write official data into the legacy fixture dir.", file=sys.stderr)
        return 1

    print("Class mapping:", CLASS_TO_ID)
    print(f"Raw dataset:      {raw}")
    print(f"Processed output: {out}")
    print(f"Results:          {results}")
    print(f"Excluding:        {sorted(exclude)}")
    print("Original VOC XML and images are read-only (never modified).")
    s = convert_official(raw, out, results, exclude=exclude, link_mode=args.link_mode,
                         countries=args.countries, max_images_per_country=args.max_images_per_country)

    print("\n=== Conversion summary ===")
    print(f"Images converted (label files written): {s['n_images_converted']}")
    print(f"Negatives (empty label files):          {s['n_negatives_empty_labels']}")
    print(f"Objects written:                        {s['n_objects']}")
    print(f"Class counts:                           {s['class_counts']}")
    print(f"Unsupported labels dropped:             {s['unsupported_labels_dropped']}")
    print("Counters:")
    for k, v in s["counters"].items():
        print(f"  {k}: {v}")
    print("Per country:")
    for c, v in s["per_country"].items():
        print(f"  {c}: {v}")
    print(f"Manifest: {s['manifest_csv']}")
    print(f"Summary:  {results / 'conversion_manifest.json'}")
    print(f"Unsupported labels: {results / 'unsupported_labels.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
