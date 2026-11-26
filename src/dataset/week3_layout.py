"""Week 3 split exports and Ultralytics directory materialization."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from src.constants import CLASS_NAMES, DEFAULT_SEED, GROUP_BLOCK_SIZE
from src.dataset.splits import create_leakage_safe_splits


def _count_classes_in_label(label_path: Path) -> Counter:
    counts: Counter = Counter()
    if not label_path.is_file():
        return counts
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return counts
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cid = int(float(parts[0]))
        except ValueError:
            continue
        if 0 <= cid <= 3:
            counts[cid] += 1
    return counts


def materialize_split_directories(
    processed_dir: Path,
    full_df: pd.DataFrame,
    *,
    symlink: bool = True,
) -> dict[str, int]:
    """Create images/{train,val,test} and labels/{train,val,test} from split table.

    Source files remain under images/<country>/ and labels/<country>/.
    """
    counts = {"train": 0, "val": 0, "test": 0}
    for split_name in ("train", "val", "test"):
        img_out = processed_dir / "images" / split_name
        lbl_out = processed_dir / "labels" / split_name
        # Clean previous materialization of this split only
        if img_out.exists():
            shutil.rmtree(img_out)
        if lbl_out.exists():
            shutil.rmtree(lbl_out)
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        subset = full_df[full_df["split"] == split_name]
        for row in subset.itertuples(index=False):
            src_img = processed_dir / row.rel_image
            src_lbl = processed_dir / row.rel_label
            # Unique names across countries
            dest_name = f"{row.country}__{row.filename}"
            dest_img = img_out / dest_name
            dest_lbl = lbl_out / f"{Path(dest_name).stem}.txt"
            if src_img.is_file():
                if symlink:
                    try:
                        if dest_img.exists() or dest_img.is_symlink():
                            dest_img.unlink()
                        dest_img.symlink_to(src_img.resolve())
                    except OSError:
                        shutil.copy2(src_img, dest_img)
                else:
                    shutil.copy2(src_img, dest_img)
            if src_lbl.is_file():
                shutil.copy2(src_lbl, dest_lbl)
            else:
                dest_lbl.write_text("", encoding="utf-8")
            counts[split_name] += 1
    return counts


def build_split_summary(
    full_df: pd.DataFrame,
    processed_dir: Path,
    *,
    seed: int,
    block_size: int,
    dup_df: pd.DataFrame,
) -> dict:
    """Rich summary for data/splits/split_summary.json."""
    summary: dict = {
        "seed": seed,
        "block_size": block_size,
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "splits": {},
        "duplicate_leakage": {},
    }

    # Duplicate leakage: pairs whose members fall in different splits
    id_to_split = dict(zip(full_df["image_id"], full_df["split"]))
    cross = 0
    if dup_df is not None and not dup_df.empty and "image_id_a" in dup_df.columns:
        for _, row in dup_df.iterrows():
            sa = id_to_split.get(str(row["image_id_a"]))
            sb = id_to_split.get(str(row["image_id_b"]))
            if sa is not None and sb is not None and sa != sb:
                cross += 1
    summary["duplicate_leakage"] = {
        "n_duplicate_pairs": int(len(dup_df)) if dup_df is not None else 0,
        "n_pairs_crossing_splits": int(cross),
        "passed": cross == 0,
    }

    for split_name in ("train", "val", "test"):
        subset = full_df[full_df["split"] == split_name]
        class_counts = Counter()
        n_neg = 0
        n_boxes = 0
        country_counts: Counter = Counter()
        for row in subset.itertuples(index=False):
            country_counts[row.country] += 1
            lbl = processed_dir / row.rel_label
            cc = _count_classes_in_label(lbl)
            if sum(cc.values()) == 0:
                n_neg += 1
            class_counts.update(cc)
            n_boxes += int(sum(cc.values()))
        n_img = int(len(subset))
        summary["splits"][split_name] = {
            "n_images": n_img,
            "n_boxes": n_boxes,
            "n_negatives": n_neg,
            "pct_negatives": round(100.0 * n_neg / n_img, 2) if n_img else 0.0,
            "per_class_boxes": {CLASS_NAMES[i]: int(class_counts.get(i, 0)) for i in range(4)},
            "country_distribution": {k: int(v) for k, v in sorted(country_counts.items())},
        }

    return summary


def create_week3_splits(
    processed_dir: Path,
    splits_dir: Path,
    results_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    block_size: int = GROUP_BLOCK_SIZE,
    symlink: bool = True,
    skip_phash: bool = False,
) -> dict:
    """Create Week 3 split CSVs, summary, YOLO lists, and train/val/test dirs."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Reuse Week 1 leakage-safe splitter (writes legacy data/*.csv + duplicate report)
    legacy_data = splits_dir.parent  # data/
    result = create_leakage_safe_splits(
        processed_dir,
        legacy_data,
        results_dir,
        seed=seed,
        block_size=block_size,
        skip_phash=skip_phash,
    )

    train_df = pd.read_csv(legacy_data / "train_split.csv")
    val_df = pd.read_csv(legacy_data / "val_split.csv")
    test_df = pd.read_csv(legacy_data / "test_split.csv")
    full = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Canonical Week 3 paths
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    dup_path = Path(result["paths"]["duplicates"])
    dup_df = pd.read_csv(dup_path) if dup_path.is_file() else pd.DataFrame()

    summary = build_split_summary(
        full, processed_dir, seed=seed, block_size=block_size, dup_df=dup_df
    )
    summary["legacy_paths"] = result["paths"]
    summary["materialized"] = materialize_split_directories(
        processed_dir, full, symlink=symlink
    )

    # Absolute path lists for Ultralytics (space-safe)
    list_dir = processed_dir / "splits"
    list_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        img_dir = processed_dir / "images" / name
        lines = sorted(str(p.resolve()) for p in img_dir.iterdir() if p.is_file())
        (list_dir / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )

    summary_path = splits_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Also copy under results/week3
    week3 = results_dir / "week3"
    week3.mkdir(parents=True, exist_ok=True)
    (week3 / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    result["week3"] = {
        "splits_dir": str(splits_dir),
        "summary_path": str(summary_path),
        "summary": summary,
    }
    return result


def write_rdd2022_yaml(
    out_path: Path,
    processed_dir: Path,
    *,
    imgsz: int = 640,
    split_id: str = "grouped_70_15_15_seed6845",
    dataset_version: str = "RDD2022-week3",
) -> Path:
    """Write Ultralytics dataset YAML with train/val/test image directories."""
    import yaml

    processed_dir = processed_dir.resolve()
    cfg = {
        "path": processed_dir.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 4,
        "names": {0: "D00", 1: "D10", 2: "D20", 3: "D40"},
        "class_ids": {"D00": 0, "D10": 1, "D20": 2, "D40": 3},
        "imgsz": imgsz,
        "dataset_version": dataset_version,
        "split_id": split_id,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# RDD2022 YOLO dataset configuration (Week 3)\n"
        "# Classes: D00=0, D10=1, D20=2, D40=3\n"
        "# Train/val/test point at materialized image directories.\n"
        "# Checkpoint selection must use val only — never test.\n"
    )
    out_path.write_text(
        header + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    # Portable copy next to data
    local = processed_dir / "rdd2022.yaml"
    local_cfg = dict(cfg)
    local_cfg["path"] = "."
    local.write_text(
        "# Portable copy (path relative to this file's directory)\n"
        + yaml.safe_dump(local_cfg, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
    return out_path
