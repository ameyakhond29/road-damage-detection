#!/usr/bin/env python3
"""Week 6 — Error analysis on the held-out test set (frozen conf/IoU).

Uses the locked confidence threshold from validation. Does not change the model.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CLASS_NAMES
from src.evaluation.week6_utils import (
    load_protocol,
    match_predictions,
    parse_yolo_label,
    resolve_ckpt,
    write_csv,
    write_json,
    SMALL_AREA_THR,
)
from src.paths import legacy_fixture_processed_dir
from src.training.baseline import pick_device

WEEK6 = ROOT / "results" / "week6"
FIGDIR = ROOT / "figures" / "week6"
LOCK = WEEK6 / "CONFIDENCE_THRESHOLD_LOCKED.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 6 error analysis (test set).")
    p.add_argument("--protocol", type=str, default=str(ROOT / "configs" / "final_evaluation.yaml"))
    p.add_argument("--device", type=str, default="")
    p.add_argument("--split", type=str, default="test", choices=["test", "val"])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--max-images", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=6845)
    return p.parse_args()


def locked_conf(protocol: dict) -> float:
    if LOCK.is_file():
        return float(json.loads(LOCK.read_text(encoding="utf-8"))["selected_conf"])
    return float(protocol.get("confidence_threshold", 0.25))


def draw_panel(img_path: Path, gt, preds_xyxy, preds_cls, preds_conf, title: str) -> Image.Image:
    im = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    W, H = im.size
    for cid, cx, cy, w, h in gt:
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
        x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
        draw.rectangle([x1, y1, x2, y2], outline=(0, 180, 0), width=2)
        draw.text((x1, max(0, y1 - 12)), f"GT {CLASS_NAMES[cid]}", fill=(0, 180, 0))
    for box, c, cf in zip(preds_xyxy, preds_cls, preds_conf):
        draw.rectangle(box, outline=(220, 40, 40), width=2)
        draw.text((box[0], max(0, box[1] - 12)), f"P {CLASS_NAMES[int(c)]} {cf:.2f}", fill=(220, 40, 40))
    draw.text((5, 5), title, fill=(255, 255, 0))
    return im


def collage(images: list[Image.Image], out: Path, cols: int = 2) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not images:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No examples in this category\n(Pending or none found)", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return
    w = max(im.width for im in images)
    h = max(im.height for im in images)
    resized = [im.resize((w, h)) for im in images]
    rows = (len(resized) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), (30, 30, 30))
    for i, im in enumerate(resized):
        r, c = divmod(i, cols)
        canvas.paste(im, (c * w, r * h))
    canvas.save(out, quality=90)


def main() -> int:
    args = parse_args()
    protocol = load_protocol(Path(args.protocol))
    conf = locked_conf(protocol)
    imgsz = int(protocol.get("imgsz", 640))
    ckpt = resolve_ckpt(protocol["final_model"]["checkpoint"])
    if ckpt is None:
        print("Missing final checkpoint", file=sys.stderr)
        return 1
    device = pick_device(args.device)
    processed = legacy_fixture_processed_dir()  # legacy Week 6 fixture artifact
    split = args.split
    split_label = "Test" if split == "test" else "Validation"
    img_dir = processed / "images" / split
    lbl_dir = processed / "labels" / split
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if args.max_images:
        images = images[: args.max_images]
    elif args.smoke:
        images = images[:12]

    from ultralytics import YOLO

    model = YOLO(str(ckpt))
    rng = random.Random(args.seed)

    error_cases = []
    fp_imgs, fn_imgs, small_imgs, loc_imgs = [], [], [], []
    confusion = np.zeros((4, 4), dtype=int)  # gt x pred for matched wrong-class / TP diag
    multi_damage = 0
    pothole_fn = 0

    for img in images:
        gt = parse_yolo_label(lbl_dir / f"{img.stem}.txt")
        gt_classes = {c for c, *_ in gt}
        if len(gt_classes) >= 2:
            multi_damage += 1
        preds = model.predict(source=str(img), imgsz=imgsz, conf=conf, device=device, verbose=False)
        pred_cls, pred_xywh, pred_conf, pred_xyxy = [], [], [], []
        if preds and preds[0].boxes is not None and len(preds[0].boxes):
            pred_cls = [int(c) for c in preds[0].boxes.cls.cpu().tolist()]
            pred_xywh = preds[0].boxes.xywhn.cpu().tolist()
            pred_conf = [float(c) for c in preds[0].boxes.conf.cpu().tolist()]
            pred_xyxy = preds[0].boxes.xyxy.cpu().tolist()

        stats = match_predictions(gt, pred_cls, pred_xywh, pred_conf, iou_thr=0.5)
        for pair in stats["pairs"]:
            cat = pair["type"]
            explanation = {
                "FP": "Predicted box with no matching GT of same class (IoU≥0.5).",
                "FN": "Ground-truth box missed (no same-class pred IoU≥0.5).",
                "LOC_POOR": "Same-class match exists but IoU < 0.5 (poor localization).",
                "TP": "Correct detection.",
            }.get(cat, cat)
            if cat == "FN" and pair.get("small"):
                explanation = "Missed small object (normalized area < 0.02)."
                cat_label = "missed_small_object"
            elif cat == "FN" and pair.get("gt_class") == "D40":
                explanation = "Pothole (D40) missed."
                cat_label = "pothole_fn"
                pothole_fn += 1
            elif cat == "FP":
                cat_label = "false_positive"
            elif cat == "FN":
                cat_label = "false_negative"
            elif cat == "LOC_POOR":
                cat_label = "localization_error"
            else:
                cat_label = "true_positive"

            # Confusion: for FP with a nearby different-class GT, record class confusion
            if cat == "TP" and pair["gt_class"] and pair["pred_class"]:
                gi = CLASS_NAMES.index(pair["gt_class"])
                pi = CLASS_NAMES.index(pair["pred_class"])
                confusion[gi, pi] += 1

            if cat != "TP":
                error_cases.append(
                    {
                        "image_path": str(img),
                        "split_label": split_label,
                        "ground_truth_class": pair.get("gt_class"),
                        "predicted_class": pair.get("pred_class"),
                        "confidence_score": pair.get("conf"),
                        "error_category": cat_label,
                        "iou": pair.get("iou"),
                        "gt_box_cxcywh": pair.get("gt_box"),
                        "pred_box_cxcywh": pair.get("pred_box"),
                        "short_explanation": explanation,
                        "confidence_threshold": conf,
                    }
                )

        panel = draw_panel(img, gt, pred_xyxy, pred_cls, pred_conf, img.stem)
        if stats["fp"] > 0 and len(fp_imgs) < 4:
            fp_imgs.append(panel.copy())
        if stats["fn"] > 0 and len(fn_imgs) < 4:
            fn_imgs.append(panel.copy())
        if stats["fn_small"] > 0 and len(small_imgs) < 4:
            small_imgs.append(panel.copy())
        if stats["localization_poor"] > 0 and len(loc_imgs) < 4:
            loc_imgs.append(panel.copy())

    # Also attempt class-confusion FPs: wrong class high IoU
    # (already captured as FP + FN separately in greedy matching)

    collage(fp_imgs, FIGDIR / "false_positives.png")
    collage(fn_imgs, FIGDIR / "false_negatives.png")
    collage(small_imgs, FIGDIR / "missed_small_objects.png")
    collage(loc_imgs, FIGDIR / "localization_errors.png")

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Week 6 TP confusion ({split_label}, conf={conf})")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(confusion[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    # Prefer Ultralytics confusion if present; else ours
    if not (FIGDIR / "confusion_matrix.png").is_file():
        fig.savefig(FIGDIR / "confusion_matrix.png", dpi=150)
    else:
        fig.savefig(FIGDIR / "confusion_matrix_tp_only.png", dpi=150)
    plt.close(fig)

    # PR curves: copy from ultralytics if missing
    if not (FIGDIR / "precision_recall_curves.png").is_file():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            "PR curve Pending — run run_final_evaluation.py first\n(Ultralytics BoxPR_curve copy)",
            ha="center",
            va="center",
        )
        ax.axis("off")
        fig.savefig(FIGDIR / "precision_recall_curves.png", dpi=150)
        plt.close(fig)

    # Cap CSV size but keep diversity
    rng.shuffle(error_cases)
    keep = error_cases[:200]
    write_csv(WEEK6 / "error_cases.csv", keep)
    write_json(
        WEEK6 / "error_analysis_summary.json",
        {
            "split_label": split_label,
            "split": split,
            "confidence_threshold": conf,
            "n_images": len(images),
            "n_error_rows": len(error_cases),
            "n_multi_damage_images": multi_damage,
            "n_pothole_fn": pothole_fn,
            "notes": [
                "Conf threshold locked from validation before this analysis.",
                "Difficult lighting/weather/viewpoint labels are not in RDD2022 metadata; "
                "those categories remain qualitative / Pending without manual tags.",
            ],
        },
    )
    print(f"Wrote {WEEK6 / 'error_cases.csv'} ({len(keep)} rows; {len(error_cases)} total errors)")
    print(f"Figures → {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
