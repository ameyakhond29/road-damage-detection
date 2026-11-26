#!/usr/bin/env python3
"""Plot the class distribution of the *modeling set* (after conversion and filtering).

The Week 1 audit figure (`figures/class_distribution.png`) counts the raw
fixture XMLs including China_Drone and before unsupported-label removal, so it
does NOT describe the 361-image dataset used for training and evaluation.
This script plots the counts recorded in `results/week3/dataset_validation.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "results" / "week3" / "dataset_validation.json"
OUT = ROOT / "figures" / "class_distribution_modeling_set.png"


def main() -> int:
    data = json.loads(VALIDATION.read_text(encoding="utf-8"))
    counts = data["class_counts"]
    total = sum(counts.values())
    labels = list(counts.keys())
    values = [counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, values, color="#2b5d7c")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", va="bottom")
    ax.set_title(
        f"Modeling set class distribution (fixture-derived, {data['n_images']} images, {total} boxes)"
    )
    ax.set_xlabel("Damage class")
    ax.set_ylabel("Bounding-box count")
    fig.text(
        0.01,
        0.01,
        "Source: results/week3/dataset_validation.json. China_Drone excluded; D50/Repair dropped. "
        "Data is the synthetic fixture, not RDD2022.",
        fontsize=7,
        color="dimgray",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
