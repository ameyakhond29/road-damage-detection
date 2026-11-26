"""Validate YOLO-format RDD2022 processed dataset."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from src.constants import CLASS_NAMES, IMAGE_EXTENSIONS


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png"} or path.suffix in IMAGE_EXTENSIONS


def _iter_country_images(images_root: Path) -> list[Path]:
    """Collect images under images/<country>/ (skip train/val/test dirs)."""
    if not images_root.is_dir():
        return []
    skip = {"train", "val", "test"}
    files: list[Path] = []
    for child in sorted(images_root.iterdir()):
        if not child.is_dir() or child.name in skip:
            continue
        for p in sorted(child.iterdir()):
            if p.is_file() and _is_image(p):
                files.append(p)
    return files


def validate_processed_dataset(
    processed_dir: Path,
    *,
    check_corrupt: bool = True,
) -> dict:
    """Validate images/labels under a YOLO processed directory.

    Expected layout: ``images/<country>/*`` and ``labels/<country>/*.txt``.
    """
    images_root = processed_dir / "images"
    labels_root = processed_dir / "labels"
    rows: list[dict] = []
    class_counts: Counter[int] = Counter()

    n_images = 0
    n_label_files = 0
    n_negatives = 0
    n_boxes = 0
    n_missing_labels = 0
    n_orphan_labels = 0
    n_bad_lines = 0
    n_bad_class = 0
    n_bad_norm = 0
    n_nonpositive = 0
    n_oob = 0
    n_corrupt = 0
    n_ok = 0

    images = _iter_country_images(images_root)
    image_index = {(p.parent.name, p.stem): p for p in images}

    if labels_root.is_dir():
        for country_dir in sorted(p for p in labels_root.iterdir() if p.is_dir()):
            if country_dir.name in {"train", "val", "test"}:
                continue
            for lbl in sorted(country_dir.glob("*.txt")):
                n_label_files += 1
                if (country_dir.name, lbl.stem) not in image_index:
                    n_orphan_labels += 1
                    rows.append(
                        {
                            "country": country_dir.name,
                            "image": lbl.stem,
                            "status": "orphan_label",
                            "detail": str(lbl),
                        }
                    )

    for img_path in tqdm(images, desc="Validating dataset"):
        n_images += 1
        country = img_path.parent.name
        lbl_path = labels_root / country / f"{img_path.stem}.txt"

        if not lbl_path.is_file():
            n_missing_labels += 1
            rows.append(
                {
                    "country": country,
                    "image": img_path.name,
                    "status": "missing_label",
                    "detail": str(lbl_path),
                }
            )
            continue

        try:
            if check_corrupt:
                with Image.open(img_path) as im:
                    im.verify()
            with Image.open(img_path) as im:
                im.load()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            n_corrupt += 1
            rows.append(
                {
                    "country": country,
                    "image": img_path.name,
                    "status": "corrupt_image",
                    "detail": str(exc),
                }
            )
            continue

        text = lbl_path.read_text(encoding="utf-8").strip()
        if not text:
            n_negatives += 1
            n_ok += 1
            rows.append(
                {
                    "country": country,
                    "image": img_path.name,
                    "status": "negative",
                    "detail": "empty label preserved",
                }
            )
            continue

        image_has_issue = False
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                n_bad_lines += 1
                image_has_issue = True
                rows.append(
                    {
                        "country": country,
                        "image": img_path.name,
                        "status": "bad_line_arity",
                        "detail": f"line {line_no}: expected 5 values, got {len(parts)}",
                    }
                )
                continue
            try:
                cid = int(float(parts[0]))
                cx, cy, nw, nh = map(float, parts[1:])
            except ValueError:
                n_bad_lines += 1
                image_has_issue = True
                rows.append(
                    {
                        "country": country,
                        "image": img_path.name,
                        "status": "bad_line_parse",
                        "detail": f"line {line_no}: {line}",
                    }
                )
                continue

            line_ok = True
            if cid < 0 or cid > 3:
                n_bad_class += 1
                image_has_issue = True
                line_ok = False
                rows.append(
                    {
                        "country": country,
                        "image": img_path.name,
                        "status": "bad_class_id",
                        "detail": f"line {line_no}: class_id={cid}",
                    }
                )

            if not (
                0.0 <= cx <= 1.0
                and 0.0 <= cy <= 1.0
                and 0.0 <= nw <= 1.0
                and 0.0 <= nh <= 1.0
            ):
                n_bad_norm += 1
                image_has_issue = True
                line_ok = False
                rows.append(
                    {
                        "country": country,
                        "image": img_path.name,
                        "status": "not_normalized",
                        "detail": f"line {line_no}: {cx} {cy} {nw} {nh}",
                    }
                )

            if nw <= 0 or nh <= 0:
                n_nonpositive += 1
                image_has_issue = True
                line_ok = False
                rows.append(
                    {
                        "country": country,
                        "image": img_path.name,
                        "status": "nonpositive_wh",
                        "detail": f"line {line_no}: w={nw} h={nh}",
                    }
                )

            x1, y1 = cx - nw / 2.0, cy - nh / 2.0
            x2, y2 = cx + nw / 2.0, cy + nh / 2.0
            eps = 1e-3
            if x1 < -eps or y1 < -eps or x2 > 1.0 + eps or y2 > 1.0 + eps:
                n_oob += 1
                image_has_issue = True
                line_ok = False
                rows.append(
                    {
                        "country": country,
                        "image": img_path.name,
                        "status": "out_of_bounds",
                        "detail": f"line {line_no}: [{x1:.4f},{y1:.4f},{x2:.4f},{y2:.4f}]",
                    }
                )

            if line_ok and 0 <= cid <= 3:
                class_counts[cid] += 1
                n_boxes += 1

        if not image_has_issue:
            n_ok += 1
            rows.append(
                {
                    "country": country,
                    "image": img_path.name,
                    "status": "ok",
                    "detail": "",
                }
            )

    summary = {
        "processed_dir": str(processed_dir),
        "n_images": n_images,
        "n_label_files_scanned": n_label_files,
        "n_ok": n_ok,
        "n_negatives": n_negatives,
        "n_boxes": n_boxes,
        "n_missing_labels": n_missing_labels,
        "n_orphan_labels": n_orphan_labels,
        "n_bad_lines": n_bad_lines,
        "n_bad_class": n_bad_class,
        "n_not_normalized": n_bad_norm,
        "n_nonpositive_wh": n_nonpositive,
        "n_out_of_bounds": n_oob,
        "n_corrupt_images": n_corrupt,
        "class_counts": {CLASS_NAMES[i]: int(class_counts.get(i, 0)) for i in range(4)},
        "negatives_preserved": bool(n_negatives > 0 or n_images == 0),
        "every_image_has_label": n_missing_labels == 0,
        "passed": (
            n_missing_labels == 0
            and n_bad_lines == 0
            and n_bad_class == 0
            and n_bad_norm == 0
            and n_nonpositive == 0
            and n_oob == 0
            and n_corrupt == 0
        ),
    }
    return {"summary": summary, "rows": rows}


def write_validation_reports(result: dict, json_path: Path, csv_path: Path) -> None:
    import json

    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")
    pd.DataFrame(result["rows"]).to_csv(csv_path, index=False)
