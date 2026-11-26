#!/usr/bin/env python3
"""Derive country-held-out (OOD) split CSVs from the fixed Week 1 splits.

Does NOT reassign images across train/val/test. It only filters the existing
fixed split files by country for the generalization experiment:

  Train / val countries: Japan, India, Czech, China_MotorBike
  Held-out eval:         Norway, United_States

Held-out images are never used for checkpoint selection.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import DEFAULT_SEED
from src.seed import set_seed

TRAIN_COUNTRIES = ("Japan", "India", "Czech", "China_MotorBike")
HELDOUT_COUNTRIES = ("Norway", "United_States")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare country-held-out split CSVs.")
    p.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    p.add_argument("--out-dir", type=str, default=str(ROOT / "data" / "ood"))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    required = ["train_split.csv", "val_split.csv", "test_split.csv"]
    for name in required:
        path = data_dir / name
        if not path.is_file():
            print(f"Missing {path}. Run Week 1 split generation first.", file=sys.stderr)
            return 1

    train = pd.read_csv(data_dir / "train_split.csv")
    val = pd.read_csv(data_dir / "val_split.csv")
    test = pd.read_csv(data_dir / "test_split.csv")
    all_df = pd.concat([train, val, test], ignore_index=True)

    ood_train = train[train["country"].isin(TRAIN_COUNTRIES)].copy()
    ood_val = val[val["country"].isin(TRAIN_COUNTRIES)].copy()
    # Held-out eval: all Norway/US rows from the fixed splits (eval only).
    heldout = all_df[all_df["country"].isin(HELDOUT_COUNTRIES)].copy()
    heldout["split"] = "heldout_eval"

    # Also keep an empty-or-filtered test stub of source-country test images
    # for optional in-distribution reporting (still not for selection).
    ood_test_id = test[test["country"].isin(TRAIN_COUNTRIES)].copy()

    paths = {
        "train_split.csv": ood_train,
        "val_split.csv": ood_val,
        "test_split.csv": ood_test_id,  # source-country test only; not for selection
        "heldout_eval.csv": heldout,
    }
    for name, df in paths.items():
        out = out_dir / name
        df.to_csv(out, index=False)
        print(f"Wrote {out} ({len(df)} rows)")

    summary = out_dir / "ood_split_summary.txt"
    summary.write_text(
        "\n".join(
            [
                "Country-held-out splits derived from fixed Week 1 splits.",
                f"Train countries: {', '.join(TRAIN_COUNTRIES)}",
                f"Held-out eval countries: {', '.join(HELDOUT_COUNTRIES)}",
                f"ood train: {len(ood_train)}",
                f"ood val: {len(ood_val)}",
                f"ood source-country test (optional): {len(ood_test_id)}",
                f"held-out eval: {len(heldout)}",
                "Checkpoint selection uses ood val only — never heldout_eval or test.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary}")
    if len(heldout) == 0:
        print(
            "Warning: held-out eval is empty (Norway/US absent in current splits). "
            "Re-run after converting full RDD2022.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
