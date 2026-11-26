"""Pascal VOC XML → YOLO label conversion."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from src.constants import CLASS_TO_ID, EXCLUDE_COUNTRIES
from src.dataset.voc import VocAnnotation, parse_voc_xml
from src.paths import country_train_dirs, discover_countries, list_images, stem_to_xml


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    """Convert absolute VOC box to normalized YOLO cx, cy, w, h."""
    if img_w <= 0 or img_h <= 0:
        return None
    xmin = max(0.0, min(float(xmin), float(img_w)))
    xmax = max(0.0, min(float(xmax), float(img_w)))
    ymin = max(0.0, min(float(ymin), float(img_h)))
    ymax = max(0.0, min(float(ymax), float(img_h)))
    bw = xmax - xmin
    bh = ymax - ymin
    if bw <= 0 or bh <= 0:
        return None
    cx = (xmin + xmax) / 2.0 / img_w
    cy = (ymin + ymax) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h
    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    nw = min(1.0, max(0.0, nw))
    nh = min(1.0, max(0.0, nh))
    return cx, cy, nw, nh


def _box_key(cid: int, cx: float, cy: float, nw: float, nh: float, decimals: int = 4) -> tuple:
    return (
        cid,
        round(cx, decimals),
        round(cy, decimals),
        round(nw, decimals),
        round(nh, decimals),
    )


def voc_xml_to_yolo_lines(
    ann: VocAnnotation,
    *,
    img_w: int | None = None,
    img_h: int | None = None,
) -> tuple[list[str], Counter, dict]:
    """Convert a parsed VOC annotation to YOLO label lines.

    Returns (lines, removed_unsupported_counts, issue_counts).
    Images with no supported boxes yield an empty list (negative example).
    """
    w = img_w if img_w is not None else ann.width
    h = img_h if img_h is not None else ann.height
    removed: Counter = Counter()
    issues: Counter = Counter()
    lines: list[str] = []
    seen: set[tuple] = set()

    if ann.parse_error:
        issues["malformed_xml"] += 1
        return [], removed, dict(issues)

    for box in ann.boxes:
        if box.label not in CLASS_TO_ID:
            removed[box.label] += 1
            issues["unknown_classes"] += 1
            continue
        if box.xmax <= box.xmin or box.ymax <= box.ymin:
            issues["invalid_boxes"] += 1
            continue
        yolo = voc_box_to_yolo(box.xmin, box.ymin, box.xmax, box.ymax, w, h)
        if yolo is None:
            issues["invalid_boxes"] += 1
            continue
        cid = CLASS_TO_ID[box.label]
        cx, cy, nw, nh = yolo
        key = _box_key(cid, cx, cy, nw, nh)
        if key in seen:
            issues["duplicate_annotations"] += 1
            continue
        seen.add(key)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines, removed, dict(issues)


def convert_dataset(
    raw_dir: Path,
    processed_dir: Path,
    *,
    exclude_countries: set[str] | None = None,
    copy_images: bool = True,
    symlink_images: bool = False,
    countries: list[str] | None = None,
    max_images_per_country: int | None = None,
) -> dict:
    """Convert filtered RDD2022 train data to YOLO layout.

    Writes under ``processed_dir``::

        images/<country>/<file>.jpg
        labels/<country>/<file>.txt

    Original files are never deleted. Unsupported labels (D50, Repair, ...)
    are dropped and counted. Empty label files are written for negatives.
    """
    exclude = exclude_countries if exclude_countries is not None else set(EXCLUDE_COUNTRIES)
    processed_dir.mkdir(parents=True, exist_ok=True)
    images_root = processed_dir / "images"
    labels_root = processed_dir / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)

    all_countries = countries or discover_countries(raw_dir)
    removed_total: Counter = Counter()
    issue_totals: Counter = Counter()
    issue_rows: list[dict] = []
    n_converted = 0
    n_negatives = 0
    n_skipped_country = 0
    n_missing_xml = 0
    n_orphan_xml = 0
    manifest_rows: list[dict] = []

    for country in tqdm(all_countries, desc="Converting countries"):
        if country in exclude:
            n_skipped_country += 1
            continue

        images_dir, ann_dir = country_train_dirs(raw_dir, country)
        images = list_images(images_dir)
        if max_images_per_country is not None:
            images = images[: max_images_per_country]

        image_stems = {p.stem for p in images}
        if ann_dir.is_dir():
            for xml_path in sorted(ann_dir.glob("*.xml")):
                if xml_path.stem not in image_stems:
                    n_orphan_xml += 1
                    issue_totals["missing_images"] += 1
                    issue_rows.append(
                        {
                            "country": country,
                            "issue": "missing_image",
                            "path": str(xml_path),
                            "detail": "XML without matching image",
                        }
                    )

        out_img_dir = images_root / country
        out_lbl_dir = labels_root / country
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(images, desc=f"  {country}", leave=False):
            xml_path = stem_to_xml(ann_dir, img_path.stem)
            from PIL import Image

            with Image.open(img_path) as im:
                img_w, img_h = im.size

            if xml_path is None:
                lines: list[str] = []
                removed: Counter = Counter()
                per_issues: dict = {}
                n_missing_xml += 1
            else:
                ann = parse_voc_xml(xml_path)
                if ann.parse_error:
                    issue_rows.append(
                        {
                            "country": country,
                            "issue": "malformed_xml",
                            "path": str(xml_path),
                            "detail": ann.parse_error,
                        }
                    )
                w = ann.width if ann.width > 0 else img_w
                h = ann.height if ann.height > 0 else img_h
                lines, removed, per_issues = voc_xml_to_yolo_lines(ann, img_w=w, img_h=h)
                for k, v in per_issues.items():
                    issue_totals[k] += int(v)
                    if k != "malformed_xml":
                        issue_rows.append(
                            {
                                "country": country,
                                "issue": k,
                                "path": str(xml_path),
                                "detail": f"count={v}",
                            }
                        )

            removed_total.update(removed)

            dest_img = out_img_dir / img_path.name
            if not dest_img.exists():
                if symlink_images:
                    try:
                        dest_img.symlink_to(img_path.resolve())
                    except OSError:
                        shutil.copy2(img_path, dest_img)
                elif copy_images:
                    shutil.copy2(img_path, dest_img)

            dest_lbl = out_lbl_dir / f"{img_path.stem}.txt"
            dest_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            n_boxes = len(lines)
            if n_boxes == 0:
                n_negatives += 1
            n_converted += 1
            manifest_rows.append(
                {
                    "country": country,
                    "image_name": img_path.name,
                    "image_path": str(dest_img.relative_to(processed_dir)),
                    "label_path": str(dest_lbl.relative_to(processed_dir)),
                    "n_boxes": n_boxes,
                    "source_image": str(img_path),
                }
            )

    import pandas as pd

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = processed_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    removed_path = processed_dir / "removed_unsupported_labels.csv"
    pd.DataFrame(
        [{"label": k, "count": int(v)} for k, v in sorted(removed_total.items())]
    ).to_csv(removed_path, index=False)

    issues_path = processed_dir / "conversion_issues.csv"
    pd.DataFrame(issue_rows).to_csv(issues_path, index=False)

    return {
        "n_converted": n_converted,
        "n_negatives": n_negatives,
        "n_skipped_countries": n_skipped_country,
        "n_missing_xml": n_missing_xml,
        "n_orphan_xml": n_orphan_xml,
        "excluded_countries": sorted(exclude),
        "removed_unsupported": dict(removed_total),
        "removed_unsupported_total": int(sum(removed_total.values())),
        "issues": {
            "malformed_xml": int(issue_totals.get("malformed_xml", 0)),
            "missing_images": int(issue_totals.get("missing_images", 0)),
            "invalid_boxes": int(issue_totals.get("invalid_boxes", 0)),
            "unknown_classes": int(issue_totals.get("unknown_classes", 0)),
            "duplicate_annotations": int(issue_totals.get("duplicate_annotations", 0)),
            "images_without_xml": int(n_missing_xml),
        },
        "manifest_path": str(manifest_path),
        "issues_path": str(issues_path),
        "processed_dir": str(processed_dir),
    }
