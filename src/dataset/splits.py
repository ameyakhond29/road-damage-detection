"""Leakage-safe 70/15/15 train-val-test split generation."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.constants import DEFAULT_SEED, GROUP_BLOCK_SIZE
from src.dataset.duplicates import build_duplicate_components, find_near_duplicates


def assign_filename_groups(
    filenames: list[str],
    country: str,
    block_size: int = GROUP_BLOCK_SIZE,
) -> dict[str, str]:
    """Map each filename to a group id of ``block_size`` consecutive names.

    Filenames are sorted lexicographically within the country, then chunked
    into blocks of ``block_size``. Group id format: ``{country}__block{k:04d}``.
    """
    sorted_names = sorted(filenames)
    mapping: dict[str, str] = {}
    for idx, name in enumerate(sorted_names):
        block = idx // block_size
        mapping[name] = f"{country}__block{block:04d}"
    return mapping


def _merge_groups_by_duplicates(
    group_of: dict[str, str],
    image_id_to_key: dict[str, str],
    dup_components: dict[str, str],
) -> dict[str, str]:
    """Ensure near-duplicate images share the same split group.

    ``image_id_to_key`` maps image_id → the key used in ``group_of``
    (typically ``country/filename``).
    """
    # Map component → set of group ids that must be merged
    component_groups: dict[str, set[str]] = defaultdict(set)
    for image_id, component in dup_components.items():
        key = image_id_to_key.get(image_id)
        if key is None:
            continue
        gid = group_of.get(key)
        if gid is not None:
            component_groups[component].add(gid)

    # Union groups that share a duplicate component
    remap: dict[str, str] = {}
    for groups in component_groups.values():
        if len(groups) <= 1:
            continue
        canonical = sorted(groups)[0]
        for g in groups:
            remap[g] = canonical

    if not remap:
        return group_of

    # Flatten remap
    def final(g: str) -> str:
        seen = set()
        while g in remap and g not in seen:
            seen.add(g)
            g = remap[g]
        return g

    return {k: final(v) for k, v in group_of.items()}


def _stratified_group_split(
    groups_by_country: dict[str, list[str]],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> dict[str, str]:
    """Assign each group to train/val/test, stratified by country when possible.

    When duplicate-merging leaves very few groups overall, a global fallback
    still tries to populate val/test so downstream smoke tests can run.
    """
    rng = random.Random(seed)
    assignment: dict[str, str] = {}

    for country, groups in sorted(groups_by_country.items()):
        unique_groups = sorted(set(groups))
        rng.shuffle(unique_groups)
        n = len(unique_groups)
        if n == 0:
            continue
        if n == 1:
            # Defer single-group countries; global fallback may rebalance.
            assignment[unique_groups[0]] = "train"
            continue
        if n == 2:
            assignment[unique_groups[0]] = "train"
            assignment[unique_groups[1]] = "val"
            continue

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        # Ensure all three splits get at least one group when n >= 3
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1

        train_g = unique_groups[:n_train]
        val_g = unique_groups[n_train : n_train + n_val]
        test_g = unique_groups[n_train + n_val :]

        for g in train_g:
            assignment[g] = "train"
        for g in val_g:
            assignment[g] = "val"
        for g in test_g:
            assignment[g] = "test"

    # Global rebalance if val or test ended up empty but we have enough groups.
    splits_present = {assignment[g] for g in assignment}
    all_groups = sorted(assignment.keys())
    if len(all_groups) >= 3 and ({"val", "test"} - splits_present):
        rng.shuffle(all_groups)
        n = len(all_groups)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = max(1, n - n_val - n_test)
        assignment = {}
        for g in all_groups[:n_train]:
            assignment[g] = "train"
        for g in all_groups[n_train : n_train + n_val]:
            assignment[g] = "val"
        for g in all_groups[n_train + n_val :]:
            assignment[g] = "test"
    elif len(all_groups) == 2 and "val" not in splits_present:
        assignment[all_groups[0]] = "train"
        assignment[all_groups[1]] = "val"

    return assignment


def _enforce_component_split_consistency(
    records: list[dict],
    components: dict[str, str],
) -> None:
    """Force all images in a pHash component into a single split (in-place)."""
    by_id = {r["image_id"]: r for r in records}
    members: dict[str, list[str]] = defaultdict(list)
    for image_id, comp in components.items():
        if image_id in by_id:
            members[comp].append(image_id)

    priority = {"train": 0, "val": 1, "test": 2}
    for ids in members.values():
        if len(ids) < 2:
            continue
        splits = [by_id[i]["split"] for i in ids]
        if len(set(splits)) == 1:
            continue
        # Majority vote; ties prefer train, then val, then test
        counts: dict[str, int] = defaultdict(int)
        for s in splits:
            counts[s] += 1
        target = sorted(counts.keys(), key=lambda s: (-counts[s], priority.get(s, 9)))[0]
        for i in ids:
            by_id[i]["split"] = target


def create_leakage_safe_splits(
    processed_dir: Path,
    data_dir: Path,
    results_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    block_size: int = GROUP_BLOCK_SIZE,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_hamming: int = 0,
    skip_phash: bool = False,
) -> dict:
    """Build fixed 70/15/15 splits with group blocking + phash leakage control.

    Expects YOLO-processed layout::

        processed_dir/images/<country>/*.jpg
        processed_dir/labels/<country>/*.txt

    Writes:
      data/train_split.csv, val_split.csv, test_split.csv
      results/duplicate_report.csv
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    images_root = processed_dir / "images"
    labels_root = processed_dir / "labels"
    if not images_root.is_dir():
        raise FileNotFoundError(
            f"Processed images not found at {images_root}. "
            "Run annotation conversion first."
        )

    records: list[dict] = []
    group_of: dict[str, str] = {}  # key = country/filename

    skip_dirs = {"train", "val", "test"}
    for country_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        country = country_dir.name
        if country in skip_dirs:
            # Skip Week-3 materialized split directories
            continue
        files = sorted(
            [p for p in country_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        names = [p.name for p in files]
        name_to_group = assign_filename_groups(names, country, block_size=block_size)

        for img_path in files:
            key = f"{country}/{img_path.name}"
            gid = name_to_group[img_path.name]
            group_of[key] = gid
            label_path = labels_root / country / f"{img_path.stem}.txt"
            n_boxes = 0
            if label_path.is_file():
                text = label_path.read_text(encoding="utf-8").strip()
                n_boxes = len([ln for ln in text.splitlines() if ln.strip()])
            records.append(
                {
                    "image_id": key,
                    "country": country,
                    "filename": img_path.name,
                    "path": str(img_path.resolve()),
                    "rel_image": f"images/{country}/{img_path.name}",
                    "rel_label": f"labels/{country}/{img_path.stem}.txt",
                    "group_id": gid,
                    "n_boxes": n_boxes,
                }
            )

    # Perceptual-hash duplicates
    if skip_phash:
        dup_df = pd.DataFrame(
            columns=[
                "image_id_a",
                "image_id_b",
                "path_a",
                "path_b",
                "country_a",
                "country_b",
                "group_id_a",
                "group_id_b",
                "phash",
                "hamming",
            ]
        )
        components: dict[str, str] = {}
    else:
        dup_df = find_near_duplicates(records, max_hamming=max_hamming)
        components = build_duplicate_components(dup_df)
        image_id_to_key = {r["image_id"]: r["image_id"] for r in records}
        group_of = _merge_groups_by_duplicates(group_of, image_id_to_key, components)

    # Refresh group ids on records after merge
    for r in records:
        r["group_id"] = group_of[r["image_id"]]

    groups_by_country: dict[str, list[str]] = defaultdict(list)
    for r in records:
        groups_by_country[r["country"]].append(r["group_id"])

    split_of_group = _stratified_group_split(
        groups_by_country,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    for r in records:
        r["split"] = split_of_group[r["group_id"]]

    # Hard guarantee: every near-duplicate component shares one split.
    if components:
        _enforce_component_split_consistency(records, components)

    full = pd.DataFrame(records)
    # Drop absolute path from published CSVs for portability; keep relative paths
    export_cols = [
        "image_id",
        "country",
        "filename",
        "rel_image",
        "rel_label",
        "group_id",
        "n_boxes",
        "split",
    ]

    train_df = full[full["split"] == "train"][export_cols]
    val_df = full[full["split"] == "val"][export_cols]
    test_df = full[full["split"] == "test"][export_cols]

    train_path = data_dir / "train_split.csv"
    val_path = data_dir / "val_split.csv"
    test_path = data_dir / "test_split.csv"
    dup_path = results_dir / "duplicate_report.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    dup_df.to_csv(dup_path, index=False)

    # YOLO path lists: use absolute image paths so Ultralytics works even when
    # the project directory contains spaces (e.g. "DATA 255").
    yolo_lists = processed_dir / "splits"
    yolo_lists.mkdir(parents=True, exist_ok=True)
    abs_map = {r["image_id"]: r["path"] for r in records}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        list_path = yolo_lists / f"{name}.txt"
        lines = [abs_map[row.image_id] for row in df.itertuples(index=False)]
        list_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    counts = {
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
        "n_groups": int(full["group_id"].nunique()),
        "n_duplicate_pairs": int(len(dup_df)),
        "n_duplicate_components": int(len(set(components.values()))) if components else 0,
    }

    return {
        "counts": counts,
        "paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "duplicates": str(dup_path),
        },
        "seed": seed,
        "block_size": block_size,
    }
