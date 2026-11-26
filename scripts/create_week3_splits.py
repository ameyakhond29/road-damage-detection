#!/usr/bin/env python3
"""Deterministic grouped 70/15/15 split of the processed official RDD2022 data.

* Groups = blocks of 50 consecutive filenames per country (``group_block_size``).
* Groups connected by exact (md5) or near-duplicate (pHash) images are merged, so
  duplicates always land in the same split.
* Per country, among seeded candidate assignments the one closest to 70/15/15 for
  images, negatives and every class's box count is kept (country and class balance).
* Only labeled train images are used; official unlabeled test images never are.

Outputs: data/splits/{train,val,test}.csv, data/splits/split_summary.json,
results/official_rdd2022/split_leakage_report.json. The previous (synthetic fixture)
split files are copied once to data/splits/legacy_fixture/ before being replaced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.official import (  # noqa: E402
    create_official_splits,
    default_workers,
    load_dataset_config,
    official_paths,
)
from src.paths import DATASET_CHOICES  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create leakage-safe official RDD2022 splits.")
    p.add_argument("--config", default=str(ROOT / "configs" / "rdd2022.yaml"))
    p.add_argument("--dataset", choices=DATASET_CHOICES, default="official")
    p.add_argument("--data-root", default=None, help="Processed root (default data/processed)")
    p.add_argument("--seed", type=int, default=None, help="Default: split_seed from config (6845)")
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--n-candidates", type=int, default=200)
    p.add_argument("--workers", type=int, default=default_workers())
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset == "fixture":
        from src.dataset.week3_layout import create_week3_splits as legacy_splits
        from src.paths import legacy_fixture_processed_dir

        out = ROOT / "results" / "fixture_runs" / "splits"
        processed = Path(args.data_root) if args.data_root else legacy_fixture_processed_dir()
        print(f"[fixture/debug] splitting {processed} -> {out} (data/splits is not touched)")
        res = legacy_splits(processed, out, out, seed=args.seed or 6845, block_size=args.block_size or 50,
                            symlink=True, skip_phash=False)
        print(f"[fixture/debug] summary: {res['week3']['summary_path']}")
        return 0
    cfg = load_dataset_config(args.config)
    paths = official_paths(cfg, args.data_root)
    seed = args.seed if args.seed is not None else int(cfg.get("split_seed", 6845))
    block = args.block_size or int(cfg.get("group_block_size", 50))
    ratios = tuple(float(x) for x in cfg.get("split_ratios", [0.7, 0.15, 0.15]))
    processed = paths["processed"]
    if not (processed / "manifest.csv").is_file():
        print(f"Processed dataset not found at {processed}.\n"
              "Run: python scripts/convert_voc_to_yolo.py --data-root data/raw/RDD2022", file=sys.stderr)
        return 1

    print(f"Processed dir: {processed}")
    print(f"Seed={seed} block_size={block} ratios={ratios} "
          f"near_duplicate_max_hamming={cfg.get('near_duplicate_max_hamming', 0)}")
    res = create_official_splits(
        processed, paths["splits"], paths["results"], seed=seed, block_size=block, ratios=ratios,
        max_hamming=int(cfg.get("near_duplicate_max_hamming", 0)),
        purge_hamming=int(cfg.get("near_duplicate_purge_hamming", 0)), n_candidates=args.n_candidates,
        image_index_csv=paths["results"] / "audit" / "image_index.csv", workers=args.workers,
    )
    s, leak = res["summary"], res["leakage"]
    print("\n=== Split summary ===")
    for name, info in s["splits"].items():
        tot = s["total"]["n_images"]
        print(f"  {name:<5} images={info['n_images']:>6} ({100 * info['n_images'] / tot:.1f}%) "
              f"objects={info['n_objects']:>6} negatives={info['n_negatives']:>5} classes={info['class_counts']}")
    print(f"  countries per split: { {k: v['countries'] for k, v in s['splits'].items()} }")
    print(f"  leakage: dup_clusters={leak['duplicate_clusters']} crossing={leak['duplicate_clusters_crossing_splits']} "
          f"groups_crossing={leak['groups_in_multiple_splits']} passed={leak['passed']}")
    print(f"  near-duplicate purge (pHash<={s['near_duplicate_purge_hamming']}): "
          f"{s['n_purged_near_duplicates']} val/test images dropped {s['purged']} "
          f"-> {paths['splits'] / 'near_duplicate_purged.csv'}")
    print(f"  crossing pairs by pHash distance after purge: {leak['informational_crossing_pairs_by_phash_distance']}")
    if s["legacy_fixture_backup"]:
        print(f"  backed up fixture split files: {s['legacy_fixture_backup']}")
    print(f"Wrote {paths['splits']}/train.csv|val.csv|test.csv and split_summary.json")
    print(f"Wrote {paths['results'] / 'split_leakage_report.json'}")
    if not leak["passed"]:
        print("ERROR: leakage detected across splits.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
