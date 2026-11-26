"""Dataset audit: counts, missing pairs, corrupt images, class/box stats."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.constants import CLASS_NAMES, CLASS_TO_ID, EXCLUDE_COUNTRIES
from src.dataset.voc import parse_voc_xml
from src.paths import (
    country_train_dirs,
    discover_countries,
    list_images,
)


def _is_corrupt_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        # verify() can leave the file in a bad state; reopen to load fully
        with Image.open(path) as img:
            img.load()
        return False
    except (UnidentifiedImageError, OSError, ValueError):
        return True


def run_audit(
    raw_dir: Path,
    results_dir: Path,
    figures_dir: Path,
    *,
    check_corrupt: bool = True,
    exclude_china_drone_from_stats: bool = False,
) -> dict:
    """Audit RDD2022 under ``raw_dir`` and write CSV/PNG summaries.

    Returns a dictionary of summary statistics (real counts only).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    countries = discover_countries(raw_dir)
    class_counts: Counter[str] = Counter()
    unsupported_counts: Counter[str] = Counter()
    per_country_rows: list[dict] = []
    box_rows: list[dict] = []

    total_images = 0
    total_xmls = 0
    missing_images = 0  # XML without matching image
    missing_annotations = 0  # image without XML
    corrupt_images = 0
    images_with_no_damage = 0  # XML present but zero supported-class boxes
    images_with_empty_or_missing_ann = 0

    for country in tqdm(countries, desc="Auditing countries"):
        if exclude_china_drone_from_stats and country in EXCLUDE_COUNTRIES:
            continue

        images_dir, ann_dir = country_train_dirs(raw_dir, country)
        images = list_images(images_dir)
        xml_files = sorted(ann_dir.glob("*.xml")) if ann_dir.is_dir() else []

        image_stems = {p.stem: p for p in images}
        xml_stems = {p.stem: p for p in xml_files}

        country_missing_img = 0
        country_missing_ann = 0
        country_corrupt = 0
        country_no_damage = 0
        country_class = Counter()

        for stem, xml_path in xml_stems.items():
            if stem not in image_stems:
                country_missing_img += 1
                continue

            img_path = image_stems[stem]
            if check_corrupt and _is_corrupt_image(img_path):
                country_corrupt += 1
                continue

            ann = parse_voc_xml(xml_path)
            supported_boxes = 0
            for box in ann.boxes:
                if box.label in CLASS_TO_ID:
                    country_class[box.label] += 1
                    class_counts[box.label] += 1
                    supported_boxes += 1
                    w = max(0.0, box.xmax - box.xmin)
                    h = max(0.0, box.ymax - box.ymin)
                    box_rows.append(
                        {
                            "country": country,
                            "image": img_path.name,
                            "class": box.label,
                            "class_id": CLASS_TO_ID[box.label],
                            "width_px": w,
                            "height_px": h,
                            "area_px": w * h,
                            "img_width": ann.width,
                            "img_height": ann.height,
                        }
                    )
                else:
                    unsupported_counts[box.label] += 1

            if supported_boxes == 0:
                country_no_damage += 1

        for stem, img_path in image_stems.items():
            if stem not in xml_stems:
                country_missing_ann += 1
                # Treat as potential negative / unlabeled — count separately
                if check_corrupt and _is_corrupt_image(img_path):
                    country_corrupt += 1

        # Also count images that have XML but no supported boxes already done.
        # Images with no XML at all are "missing annotations" (kept as negatives later).

        n_img = len(images)
        n_xml = len(xml_files)
        total_images += n_img
        total_xmls += n_xml
        missing_images += country_missing_img
        missing_annotations += country_missing_ann
        corrupt_images += country_corrupt
        images_with_no_damage += country_no_damage
        images_with_empty_or_missing_ann += country_no_damage + country_missing_ann

        per_country_rows.append(
            {
                "country": country,
                "excluded_by_default": country in EXCLUDE_COUNTRIES,
                "n_images": n_img,
                "n_annotations_xml": n_xml,
                "missing_images_for_xml": country_missing_img,
                "missing_annotations_for_image": country_missing_ann,
                "corrupt_images": country_corrupt,
                "images_with_no_supported_damage": country_no_damage,
                "D00": country_class.get("D00", 0),
                "D10": country_class.get("D10", 0),
                "D20": country_class.get("D20", 0),
                "D40": country_class.get("D40", 0),
                "images_dir": str(images_dir),
                "annotations_dir": str(ann_dir),
            }
        )

    # --- Write CSVs ---
    dataset_stats = pd.DataFrame(per_country_rows)
    dataset_stats_path = results_dir / "dataset_statistics.csv"
    dataset_stats.to_csv(dataset_stats_path, index=False)

    class_dist = pd.DataFrame(
        [
            {
                "class": c,
                "class_id": CLASS_TO_ID[c],
                "count": int(class_counts.get(c, 0)),
            }
            for c in CLASS_NAMES
        ]
    )
    class_dist_path = results_dir / "class_distribution.csv"
    class_dist.to_csv(class_dist_path, index=False)

    if box_rows:
        box_df = pd.DataFrame(box_rows)
        box_summary = (
            box_df.groupby("class")
            .agg(
                n_boxes=("area_px", "count"),
                width_mean=("width_px", "mean"),
                width_std=("width_px", "std"),
                width_median=("width_px", "median"),
                height_mean=("height_px", "mean"),
                height_std=("height_px", "std"),
                height_median=("height_px", "median"),
                area_mean=("area_px", "mean"),
                area_std=("area_px", "std"),
                area_median=("area_px", "median"),
                area_min=("area_px", "min"),
                area_max=("area_px", "max"),
            )
            .reset_index()
        )
        # Also append overall row
        overall = {
            "class": "ALL",
            "n_boxes": len(box_df),
            "width_mean": box_df["width_px"].mean(),
            "width_std": box_df["width_px"].std(),
            "width_median": box_df["width_px"].median(),
            "height_mean": box_df["height_px"].mean(),
            "height_std": box_df["height_px"].std(),
            "height_median": box_df["height_px"].median(),
            "area_mean": box_df["area_px"].mean(),
            "area_std": box_df["area_px"].std(),
            "area_median": box_df["area_px"].median(),
            "area_min": box_df["area_px"].min(),
            "area_max": box_df["area_px"].max(),
        }
        box_summary = pd.concat([box_summary, pd.DataFrame([overall])], ignore_index=True)
    else:
        box_df = pd.DataFrame(
            columns=[
                "country",
                "image",
                "class",
                "class_id",
                "width_px",
                "height_px",
                "area_px",
                "img_width",
                "img_height",
            ]
        )
        box_summary = pd.DataFrame(
            columns=[
                "class",
                "n_boxes",
                "width_mean",
                "width_std",
                "width_median",
                "height_mean",
                "height_std",
                "height_median",
                "area_mean",
                "area_std",
                "area_median",
                "area_min",
                "area_max",
            ]
        )

    box_stats_path = results_dir / "box_statistics.csv"
    box_summary.to_csv(box_stats_path, index=False)

    unsupported_path = results_dir / "unsupported_labels.csv"
    pd.DataFrame(
        [{"label": k, "count": v} for k, v in sorted(unsupported_counts.items())]
    ).to_csv(unsupported_path, index=False)

    # --- Figures ---
    _plot_class_distribution(class_dist, figures_dir / "class_distribution.png")
    _plot_box_size_distribution(box_df, figures_dir / "box_size_distribution.png")

    summary = {
        "countries": countries,
        "total_images": total_images,
        "total_xmls": total_xmls,
        "missing_images": missing_images,
        "missing_annotations": missing_annotations,
        "corrupt_images": corrupt_images,
        "images_with_no_damage": images_with_no_damage,
        "images_with_empty_or_missing_ann": images_with_empty_or_missing_ann,
        "class_counts": {c: int(class_counts.get(c, 0)) for c in CLASS_NAMES},
        "unsupported_counts": dict(unsupported_counts),
        "paths": {
            "dataset_statistics": str(dataset_stats_path),
            "class_distribution": str(class_dist_path),
            "box_statistics": str(box_stats_path),
            "unsupported_labels": str(unsupported_path),
        },
    }
    return summary


def _plot_class_distribution(class_dist: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(class_dist["class"], class_dist["count"], color="#2c5f7c")
    ax.set_xlabel("Damage class")
    ax.set_ylabel("Bounding-box count")
    ax.set_title("RDD2022 class distribution (D00/D10/D20/D40)")
    for i, row in class_dist.iterrows():
        ax.text(i, row["count"], str(int(row["count"])), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_box_size_distribution(box_df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    if box_df.empty:
        for ax, title in zip(axes, ["Width (px)", "Height (px)", "Area (px²)"]):
            ax.text(0.5, 0.5, "No boxes", ha="center", va="center")
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        axes[0].hist(box_df["width_px"], bins=40, color="#2c5f7c", edgecolor="white")
        axes[0].set_title("Box width (px)")
        axes[0].set_xlabel("width")
        axes[0].set_ylabel("count")

        axes[1].hist(box_df["height_px"], bins=40, color="#3d7a4a", edgecolor="white")
        axes[1].set_title("Box height (px)")
        axes[1].set_xlabel("height")

        # Log-scale area helps long-tail distributions.
        areas = box_df["area_px"].clip(lower=1.0)
        axes[2].hist(np.log10(areas), bins=40, color="#8b4513", edgecolor="white")
        axes[2].set_title("Box area (log10 px²)")
        axes[2].set_xlabel("log10(area)")

    fig.suptitle("Bounding-box size distributions", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
