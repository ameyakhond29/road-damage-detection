#!/usr/bin/env python3
"""Week 4 — Evaluation, visuals, small-object and country analysis.

Validation metrics = selection / reporting of training fitness.
Test metrics = final locked evaluation (never used for selection).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CLASS_NAMES, DEFAULT_SEED
from src.paths import (
    ensure_legacy_fixture_data_yaml,
    legacy_fixture_processed_dir,
    legacy_fixture_splits_dir,
)
from src.seed import set_seed
from src.training.baseline import pick_device, to_float, write_json

# Normalized box area threshold for "small" damage (at native label coords).
SMALL_AREA_THR = 0.02  # ~ area share of image; ~90x90 at 640² ≈ 0.02


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 4 evaluation and visual analysis.")
    p.add_argument("--data", type=str, default="",
                   help="Ultralytics YAML (default: legacy fixture YAML — Week 4 results are fixture artifacts)")
    p.add_argument(
        "--v8-ckpt",
        type=str,
        default="",
        help="Override YOLOv8n checkpoint (default: from week3 metrics.json)",
    )
    p.add_argument(
        "--v11-ckpt",
        type=str,
        default="",
        help="Override YOLO11n checkpoint (default: from week4 metrics.json)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "week4" / "evaluation"),
    )
    p.add_argument(
        "--figures-dir",
        type=str,
        default=str(ROOT / "figures" / "week4"),
    )
    p.add_argument("--processed-dir", type=str, default=str(legacy_fixture_processed_dir()))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--n-examples", type=int, default=12)
    return p.parse_args()


def resolve_ckpt(explicit: str, metrics_path: Path) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    if metrics_path.is_file():
        card = json.loads(metrics_path.read_text(encoding="utf-8"))
        ckpt = card.get("best_checkpoint")
        if isinstance(ckpt, str) and Path(ckpt).is_file():
            return Path(ckpt)
    return None


def run_split_eval(model, data_yaml: Path, split: str, device: str, imgsz: int) -> dict:
    res = model.val(
        data=str(data_yaml.resolve()),
        split=split,
        imgsz=imgsz,
        device=device,
        plots=True,
        verbose=False,
    )
    out = {
        "split": split,
        "metrics": {},
        "per_class": {},
        "save_dir": str(getattr(res, "save_dir", "")),
    }
    if hasattr(res, "results_dict") and res.results_dict:
        out["metrics"] = {k: to_float(v) for k, v in res.results_dict.items()}
    # Per-class AP50 if available
    try:
        box = res.box
        if hasattr(box, "ap50") and box.ap50 is not None:
            ap50 = box.ap50
            for i, name in enumerate(CLASS_NAMES):
                if i < len(ap50):
                    out["per_class"][name] = {
                        "AP50": to_float(ap50[i]),
                    }
        if hasattr(box, "p") and box.p is not None:
            for i, name in enumerate(CLASS_NAMES):
                out["per_class"].setdefault(name, {})
                if i < len(box.p):
                    out["per_class"][name]["precision"] = to_float(box.p[i])
                if hasattr(box, "r") and box.r is not None and i < len(box.r):
                    out["per_class"][name]["recall"] = to_float(box.r[i])
    except Exception as exc:  # noqa: BLE001
        out["per_class_error"] = str(exc)
    return out


def copy_eval_plots(save_dir: str, dest: Path, prefix: str) -> None:
    if not save_dir:
        return
    src = Path(save_dir)
    if not src.is_dir():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "BoxPR_curve.png",
        "BoxP_curve.png",
        "BoxR_curve.png",
        "BoxF1_curve.png",
    ):
        f = src / name
        if f.is_file():
            shutil.copy2(f, dest / f"{prefix}_{name}")


def parse_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cid, cx, cy, w, h = int(float(parts[0])), *map(float, parts[1:])
        boxes.append((cid, cx, cy, w, h))
    return boxes


def iou_xywh(a, b) -> float:
    # a,b = cx,cy,w,h normalized
    def to_xyxy(c):
        cx, cy, w, h = c
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def small_object_analysis(
    model,
    processed_dir: Path,
    split: str,
    *,
    device: str,
    imgsz: int,
    conf: float,
    iou_thr: float = 0.5,
) -> dict:
    """Recall on GT boxes with normalized area < SMALL_AREA_THR."""
    img_dir = processed_dir / "images" / split
    lbl_dir = processed_dir / "labels" / split
    if not img_dir.is_dir():
        return {"status": "unavailable", "reason": f"missing {img_dir}"}

    n_small = 0
    n_hit = 0
    per_class = {c: {"n": 0, "hit": 0} for c in CLASS_NAMES}

    images = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    for img_path in images:
        gt = parse_yolo_label(lbl_dir / f"{img_path.stem}.txt")
        small_gt = [(c, cx, cy, w, h) for c, cx, cy, w, h in gt if (w * h) < SMALL_AREA_THR]
        if not small_gt:
            continue
        preds = model.predict(
            source=str(img_path), imgsz=imgsz, conf=conf, device=device, verbose=False
        )
        pred_boxes = []
        if preds:
            r0 = preds[0]
            if r0.boxes is not None and len(r0.boxes):
                xywhn = r0.boxes.xywhn.cpu().tolist()
                cls = r0.boxes.cls.cpu().tolist()
                for c, box in zip(cls, xywhn):
                    pred_boxes.append((int(c), box[0], box[1], box[2], box[3]))

        for cid, cx, cy, w, h in small_gt:
            n_small += 1
            cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else str(cid)
            per_class.setdefault(cname, {"n": 0, "hit": 0})
            per_class[cname]["n"] += 1
            matched = False
            for pc, px, py, pw, ph in pred_boxes:
                if pc != cid:
                    continue
                if iou_xywh((cx, cy, w, h), (px, py, pw, ph)) >= iou_thr:
                    matched = True
                    break
            if matched:
                n_hit += 1
                per_class[cname]["hit"] += 1

    recall = (n_hit / n_small) if n_small else None
    return {
        "status": "ok",
        "split": split,
        "area_threshold": SMALL_AREA_THR,
        "n_small_gt_boxes": n_small,
        "n_small_detected_iou50": n_hit,
        "small_recall": recall,
        "per_class": {
            k: {
                "n": v["n"],
                "hit": v["hit"],
                "recall": (v["hit"] / v["n"]) if v["n"] else None,
            }
            for k, v in per_class.items()
        },
    }


def confidence_sweep(
    model, data_yaml: Path, split: str, device: str, imgsz: int, confs: list[float]
) -> list[dict]:
    rows = []
    for conf in confs:
        res = model.val(
            data=str(data_yaml.resolve()),
            split=split,
            imgsz=imgsz,
            device=device,
            conf=conf,
            plots=False,
            verbose=False,
        )
        m = res.results_dict if hasattr(res, "results_dict") and res.results_dict else {}
        rows.append(
            {
                "split": split,
                "conf": conf,
                "mAP50": m.get("metrics/mAP50(B)", None),
                "precision": m.get("metrics/precision(B)", None),
                "recall": m.get("metrics/recall(B)", None),
            }
        )
    return rows


def country_eval_from_csv(
    model,
    processed_dir: Path,
    split_csv: Path,
    *,
    device: str,
    imgsz: int,
    conf: float,
) -> dict:
    """Per-country detection recall proxy on images listed in a split CSV."""
    if not split_csv.is_file():
        return {"status": "unavailable", "reason": f"missing {split_csv}"}
    df = pd.read_csv(split_csv)
    if "country" not in df.columns:
        return {"status": "unavailable", "reason": "no country column"}

    by_country: dict[str, dict] = {}
    for country, sub in df.groupby("country"):
        n_img = 0
        n_gt = 0
        n_hit = 0
        for row in sub.itertuples(index=False):
            # Prefer materialized names country__filename
            cand = [
                processed_dir / "images" / row.split / f"{row.country}__{row.filename}",
                Path(row.path) if hasattr(row, "path") else None,
                processed_dir / row.rel_image if hasattr(row, "rel_image") else None,
            ]
            img_path = next((p for p in cand if p is not None and Path(p).is_file()), None)
            if img_path is None:
                continue
            lbl = processed_dir / "labels" / row.split / f"{img_path.stem}.txt"
            if not lbl.is_file():
                lbl = processed_dir / row.rel_label if hasattr(row, "rel_label") else lbl
            gt = parse_yolo_label(Path(lbl))
            n_img += 1
            if not gt:
                continue
            preds = model.predict(
                source=str(img_path), imgsz=imgsz, conf=conf, device=device, verbose=False
            )
            pred_boxes = []
            if preds and preds[0].boxes is not None and len(preds[0].boxes):
                xywhn = preds[0].boxes.xywhn.cpu().tolist()
                cls = preds[0].boxes.cls.cpu().tolist()
                pred_boxes = [
                    (int(c), b[0], b[1], b[2], b[3]) for c, b in zip(cls, xywhn)
                ]
            for cid, cx, cy, w, h in gt:
                n_gt += 1
                for pc, px, py, pw, ph in pred_boxes:
                    if pc == cid and iou_xywh((cx, cy, w, h), (px, py, pw, ph)) >= 0.5:
                        n_hit += 1
                        break
        by_country[str(country)] = {
            "n_images": n_img,
            "n_gt_boxes": n_gt,
            "n_detected_iou50": n_hit,
            "box_recall": (n_hit / n_gt) if n_gt else None,
        }
    return {"status": "ok", "per_country": by_country}


def save_prediction_gallery(
    model,
    processed_dir: Path,
    split: str,
    out_dir: Path,
    *,
    device: str,
    imgsz: int,
    conf: float,
    n_examples: int,
    seed: int,
) -> dict:
    """Save annotated examples, including FP/FN and small-box cases."""
    img_dir = processed_dir / "images" / split
    lbl_dir = processed_dir / "labels" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    rng = random.Random(seed)

    # Categorize
    with_small = []
    with_gt = []
    for p in images:
        gt = parse_yolo_label(lbl_dir / f"{p.stem}.txt")
        if gt:
            with_gt.append(p)
            if any(w * h < SMALL_AREA_THR for _, _, _, w, h in gt):
                with_small.append(p)

    chosen = []
    chosen.extend(rng.sample(with_small, min(4, len(with_small))))
    remain = [p for p in with_gt if p not in chosen]
    chosen.extend(rng.sample(remain, min(max(0, n_examples - len(chosen)), len(remain))))
    # Fill with any images if needed
    if len(chosen) < n_examples:
        remain2 = [p for p in images if p not in chosen]
        chosen.extend(rng.sample(remain2, min(n_examples - len(chosen), len(remain2))))

    meta = []
    for i, img_path in enumerate(chosen[:n_examples]):
        preds = model.predict(
            source=str(img_path),
            imgsz=imgsz,
            conf=conf,
            device=device,
            verbose=False,
            save=False,
        )
        gt = parse_yolo_label(lbl_dir / f"{img_path.stem}.txt")
        im = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        W, H = im.size
        # GT green
        for cid, cx, cy, w, h in gt:
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
            draw.text((x1, max(0, y1 - 10)), f"GT {CLASS_NAMES[cid]}", fill=(0, 200, 0))
        n_pred = 0
        if preds and preds[0].boxes is not None and len(preds[0].boxes):
            xyxy = preds[0].boxes.xyxy.cpu().tolist()
            cls = preds[0].boxes.cls.cpu().tolist()
            confs = preds[0].boxes.conf.cpu().tolist()
            n_pred = len(xyxy)
            for box, c, cf in zip(xyxy, cls, confs):
                draw.rectangle(box, outline=(220, 40, 40), width=2)
                draw.text(
                    (box[0], max(0, box[1] - 10)),
                    f"P {CLASS_NAMES[int(c)]} {cf:.2f}",
                    fill=(220, 40, 40),
                )
        tag = "small" if img_path in with_small else "normal"
        if n_pred == 0 and gt:
            tag = "fn_candidate"
        elif n_pred and not gt:
            tag = "fp_candidate"
        out_name = f"{split}_{i:02d}_{tag}_{img_path.stem}.jpg"
        im.save(out_dir / out_name, quality=90)
        meta.append(
            {
                "file": out_name,
                "image": img_path.name,
                "n_gt": len(gt),
                "n_pred": n_pred,
                "tag": tag,
            }
        )
    return {"n_saved": len(meta), "examples": meta, "dir": str(out_dir)}


def plot_conf_curves(rows: list[dict], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    confs = [r["conf"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("mAP50", "mAP@0.5"),
    ):
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except (TypeError, ValueError):
                vals.append(float("nan"))
        ax.plot(confs, vals, marker="o", label=label)
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data_yaml = Path(args.data) if args.data else ensure_legacy_fixture_data_yaml()
    if not data_yaml.is_file():
        print(f"Missing data yaml: {data_yaml}", file=sys.stderr)
        return 1

    v8_ckpt = resolve_ckpt(
        args.v8_ckpt, ROOT / "results" / "week3" / "yolov8n_baseline" / "metrics.json"
    )
    v11_ckpt = resolve_ckpt(
        args.v11_ckpt, ROOT / "results" / "week4" / "yolo11n_baseline" / "metrics.json"
    )
    if v8_ckpt is None or v11_ckpt is None:
        print(
            f"Need both checkpoints. v8={v8_ckpt} v11={v11_ckpt}",
            file=sys.stderr,
        )
        return 1

    device = pick_device(args.device)
    processed = Path(args.processed_dir)
    from ultralytics import YOLO

    report = {
        "protocol": {
            "selection_split": "val",
            "final_report_split": "test",
            "test_used_for_selection": False,
            "seed": args.seed,
            "imgsz": args.imgsz,
            "conf_default": args.conf,
            "small_area_threshold": SMALL_AREA_THR,
        },
        "checkpoints": {"YOLOv8n": str(v8_ckpt), "YOLO11n": str(v11_ckpt)},
        "models": {},
        "generalization_note": (
            "Country metrics use the existing Week-3 split CSVs (no new split). "
            "A dedicated Japan/India/Czech/China_MotorBike→Norway/US held-out "
            "protocol exists in docs/week2 but is optional; if OOD lists are empty "
            "or unbalanced on the current data, limitations are recorded here."
        ),
    }

    for name, ckpt in (("YOLOv8n", v8_ckpt), ("YOLO11n", v11_ckpt)):
        print(f"\n=== Evaluating {name} ===")
        model = YOLO(str(ckpt))
        model_out = out_dir / name.lower()
        model_out.mkdir(parents=True, exist_ok=True)

        val_metrics = run_split_eval(model, data_yaml, "val", device, args.imgsz)
        copy_eval_plots(val_metrics.get("save_dir", ""), model_out, "val")
        test_metrics = run_split_eval(model, data_yaml, "test", device, args.imgsz)
        copy_eval_plots(test_metrics.get("save_dir", ""), model_out, "test")

        # Also copy key plots into figures/week4
        for split_prefix in ("val", "test"):
            for plot in ("confusion_matrix.png", "BoxPR_curve.png"):
                src = model_out / f"{split_prefix}_{plot}"
                if src.is_file():
                    shutil.copy2(
                        src,
                        fig_dir / f"{name.lower()}_{split_prefix}_{plot}",
                    )

        small_val = small_object_analysis(
            model, processed, "val", device=device, imgsz=args.imgsz, conf=args.conf
        )
        small_test = small_object_analysis(
            model, processed, "test", device=device, imgsz=args.imgsz, conf=args.conf
        )

        conf_rows = confidence_sweep(
            model,
            data_yaml,
            "val",
            device,
            args.imgsz,
            confs=[0.05, 0.1, 0.25, 0.4, 0.5, 0.65],
        )
        pd.DataFrame(conf_rows).to_csv(model_out / "confidence_threshold_val.csv", index=False)
        plot_conf_curves(
            conf_rows,
            fig_dir / f"{name.lower()}_confidence_analysis.png",
            f"{name} confidence sweep (VAL)",
        )

        gallery = save_prediction_gallery(
            model,
            processed,
            "test",
            fig_dir / "predictions" / name.lower(),
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
            n_examples=args.n_examples,
            seed=args.seed,
        )

        country_val = country_eval_from_csv(
            model,
            processed,
            legacy_fixture_splits_dir() / "val.csv",
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
        )
        country_test = country_eval_from_csv(
            model,
            processed,
            legacy_fixture_splits_dir() / "test.csv",
            device=device,
            imgsz=args.imgsz,
            conf=args.conf,
        )

        model_report = {
            "validation": val_metrics,
            "test": test_metrics,
            "small_object": {"val": small_val, "test": small_test},
            "confidence_sweep_val": conf_rows,
            "prediction_gallery": gallery,
            "country_analysis": {"val": country_val, "test": country_test},
        }
        write_json(model_out / "evaluation.json", model_report)
        report["models"][name] = model_report
        print(
            f"  VAL mAP50={val_metrics['metrics'].get('metrics/mAP50(B)', 'Pending')} | "
            f"TEST mAP50={test_metrics['metrics'].get('metrics/mAP50(B)', 'Pending')}"
        )

    # Held-out protocol limitation note
    ood_held = ROOT / "data" / "ood" / "heldout_eval.csv"
    if ood_held.is_file() and len(pd.read_csv(ood_held)) > 0:
        report["ood_heldout_csv"] = str(ood_held)
        report["ood_status"] = (
            "data/ood/heldout_eval.csv exists; full OOD retrain track is separate "
            "from this in-split country analysis."
        )
    else:
        report["ood_status"] = (
            "Dedicated country-held-out eval list empty or missing. "
            "Week-4 country numbers use the existing fixed split only "
            "(no protocol change)."
        )

    write_json(out_dir / "evaluation_summary.json", report)
    print(f"\nWrote {out_dir / 'evaluation_summary.json'}")
    print(f"Figures under {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
