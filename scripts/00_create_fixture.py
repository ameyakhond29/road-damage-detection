#!/usr/bin/env python3
"""Create a tiny synthetic RDD2022-shaped fixture for pipeline verification.

This is NOT the real RDD2022 dataset. Counts/metrics from this fixture must
never be reported as official RDD2022 statistics. Use it only when the real
dataset is not yet downloaded, to verify that audit → convert → split → smoke
scripts execute end-to-end.
"""

from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import DEFAULT_SEED
from src.seed import set_seed

COUNTRIES = [
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
]

LABELS = ["D00", "D10", "D20", "D40", "D50", "Repair"]


def make_voc_xml(
    path: Path,
    filename: str,
    width: int,
    height: int,
    boxes: list[tuple[str, int, int, int, int]],
) -> None:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = filename
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    for label, xmin, ymin, xmax, ymax in boxes:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(xmin)
        ET.SubElement(bnd, "ymin").text = str(ymin)
        ET.SubElement(bnd, "xmax").text = str(xmax)
        ET.SubElement(bnd, "ymax").text = str(ymax)
    tree = ET.ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def make_image(path: Path, width: int, height: int, seed_color: int) -> None:
    """Create a visually unique synthetic road-like image.

    Images must differ enough that perceptual hashes do not collapse the
    entire fixture into one duplicate component (which would break splits).
    """
    rng = random.Random(seed_color)
    bg = (rng.randint(20, 200), rng.randint(20, 200), rng.randint(20, 200))
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    # Coarse unique tiles (fast) so phash differs across samples
    tile = 32
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            draw.rectangle([x, y, x + tile, y + tile], fill=color)
    for _ in range(12):
        x0, y0 = rng.randint(0, width - 1), rng.randint(0, height - 1)
        x1, y1 = rng.randint(0, width - 1), rng.randint(0, height - 1)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        draw.ellipse([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], outline=color, width=3)
    draw.rectangle(
        [0, height // 2, width, height],
        fill=(rng.randint(40, 80), rng.randint(40, 80), rng.randint(40, 80)),
    )
    draw.line(
        [(width // 2 + rng.randint(-80, 80), height // 2), (width // 2 + rng.randint(-40, 40), height)],
        fill=(200, 200, 80),
        width=3,
    )
    # Encode a unique id as a small barcode-like strip
    for i, bit in enumerate(f"{seed_color & 0xFFFF:016b}"):
        shade = 255 if bit == "1" else 0
        draw.rectangle([4 + i * 4, 4, 6 + i * 4, 20], fill=(shade, shade, shade))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=90)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create synthetic RDD2022-shaped fixture.")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "data" / "fixture" / "RDD2022"),
    )
    p.add_argument("--images-per-country", type=int, default=60)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=640)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for country in COUNTRIES:
        img_dir = out / country / "train" / "images"
        xml_dir = out / country / "train" / "annotations" / "xmls"
        img_dir.mkdir(parents=True, exist_ok=True)
        xml_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.images_per_country):
            # Mimic consecutive naming so block-of-50 grouping is meaningful
            name = f"{country}_{i:05d}.jpg"
            img_path = img_dir / name
            make_image(img_path, args.width, args.height, seed_color=args.seed + i + hash(country) % 1000)

            boxes: list[tuple[str, int, int, int, int]] = []
            # ~20% negatives
            if rng.random() > 0.20:
                n_boxes = rng.randint(1, 3)
                for _ in range(n_boxes):
                    label = rng.choice(LABELS)
                    w = rng.randint(20, 120)
                    h = rng.randint(20, 120)
                    xmin = rng.randint(0, args.width - w - 1)
                    ymin = rng.randint(0, args.height - h - 1)
                    boxes.append((label, xmin, ymin, xmin + w, ymin + h))

            # ~5% missing annotation (image only)
            if rng.random() < 0.05:
                total += 1
                continue

            make_voc_xml(xml_dir / f"{Path(name).stem}.xml", name, args.width, args.height, boxes)
            total += 1

        # One orphan XML without image (for audit missing-image check)
        make_voc_xml(
            xml_dir / "orphan_no_image.xml",
            "orphan_no_image.jpg",
            args.width,
            args.height,
            [("D00", 10, 10, 50, 50)],
        )

    # Near-duplicate pair for phash testing (copy nearly identical images)
    src = out / "Japan" / "train" / "images" / "Japan_00000.jpg"
    dup = out / "Japan" / "train" / "images" / "Japan_dup_of_00000.jpg"
    if src.is_file():
        Image.open(src).save(dup, quality=95)
        # Matching empty-ish annotation
        make_voc_xml(
            out / "Japan" / "train" / "annotations" / "xmls" / "Japan_dup_of_00000.xml",
            dup.name,
            args.width,
            args.height,
            [("D10", 30, 30, 90, 90)],
        )

    print(f"Synthetic fixture written to: {out}")
    print(f"Countries: {', '.join(COUNTRIES)}")
    print(f"Approx images created: {total} (+ orphan XMLs / duplicate)")
    print(
        "NOTE: This is a synthetic fixture for pipeline verification only — "
        "not real RDD2022 data."
    )
    print(f"\nUse with:\n  python scripts/01_audit_dataset.py --raw-dir {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
