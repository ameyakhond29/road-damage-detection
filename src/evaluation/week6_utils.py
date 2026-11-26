"""Shared Week 6 evaluation utilities."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import yaml

from src.constants import CLASS_NAMES

ROOT = Path(__file__).resolve().parents[2]
SMALL_AREA_THR = 0.02
PROTOCOL_PATH = ROOT / "configs" / "final_evaluation.yaml"


def pending(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return True
    return str(v).strip().lower() in {"", "pending"}


def load_protocol(path: Path | None = None) -> dict:
    p = path or PROTOCOL_PATH
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def f1_score(p, r) -> float | str:
    if pending(p) or pending(r):
        return "Pending"
    p, r = float(p), float(r)
    if p + r <= 0:
        return 0.0
    return round(2 * p * r / (p + r), 6)


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


def resolve_ckpt(rel_or_abs: str) -> Path | None:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = ROOT / p
    return p if p.is_file() else None


def extract_val_metrics(res) -> dict:
    out = {
        "map50": "Pending",
        "map50_95": "Pending",
        "precision": "Pending",
        "recall": "Pending",
        "f1": "Pending",
        "per_class_ap50": {},
        "per_class_precision": {},
        "per_class_recall": {},
        "raw": {},
    }
    if hasattr(res, "results_dict") and res.results_dict:
        m = {k: to_float(v) for k, v in res.results_dict.items()}
        out["raw"] = m
        out["map50"] = m.get("metrics/mAP50(B)", "Pending")
        out["map50_95"] = m.get("metrics/mAP50-95(B)", "Pending")
        out["precision"] = m.get("metrics/precision(B)", "Pending")
        out["recall"] = m.get("metrics/recall(B)", "Pending")
        out["f1"] = f1_score(out["precision"], out["recall"])
    try:
        box = res.box
        if hasattr(box, "ap50") and box.ap50 is not None:
            for i, name in enumerate(CLASS_NAMES):
                if i < len(box.ap50):
                    out["per_class_ap50"][name] = to_float(box.ap50[i])
        if hasattr(box, "p") and box.p is not None:
            for i, name in enumerate(CLASS_NAMES):
                if i < len(box.p):
                    out["per_class_precision"][name] = to_float(box.p[i])
        if hasattr(box, "r") and box.r is not None:
            for i, name in enumerate(CLASS_NAMES):
                if i < len(box.r):
                    out["per_class_recall"][name] = to_float(box.r[i])
    except Exception:  # noqa: BLE001
        pass
    return out


def match_predictions(
    gt: list[tuple[int, float, float, float, float]],
    pred_cls: list[int],
    pred_xywh: list[list[float]],
    pred_conf: list[float],
    iou_thr: float = 0.5,
) -> dict:
    """Greedy class-aware matching. Returns FP/FN/TP and localization-poor matches."""
    used_pred = set()
    tp = 0
    fn = 0
    localization_poor = 0
    fn_small = 0
    matched_pairs = []
    for cid, cx, cy, w, h in gt:
        best_i, best_iou = None, 0.0
        for i, (pc, box) in enumerate(zip(pred_cls, pred_xywh)):
            if i in used_pred or pc != cid:
                continue
            iou = iou_xywh((cx, cy, w, h), tuple(box))
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i is None or best_iou < 1e-6:
            fn += 1
            if w * h < SMALL_AREA_THR:
                fn_small += 1
            matched_pairs.append(
                {
                    "type": "FN",
                    "gt_class": CLASS_NAMES[cid] if 0 <= cid < 4 else str(cid),
                    "pred_class": None,
                    "conf": None,
                    "iou": 0.0,
                    "gt_box": [cx, cy, w, h],
                    "pred_box": None,
                    "small": w * h < SMALL_AREA_THR,
                }
            )
            continue
        used_pred.add(best_i)
        if best_iou >= iou_thr:
            tp += 1
            matched_pairs.append(
                {
                    "type": "TP",
                    "gt_class": CLASS_NAMES[cid],
                    "pred_class": CLASS_NAMES[pred_cls[best_i]],
                    "conf": pred_conf[best_i],
                    "iou": best_iou,
                    "gt_box": [cx, cy, w, h],
                    "pred_box": list(pred_xywh[best_i]),
                    "small": w * h < SMALL_AREA_THR,
                }
            )
        else:
            localization_poor += 1
            matched_pairs.append(
                {
                    "type": "LOC_POOR",
                    "gt_class": CLASS_NAMES[cid],
                    "pred_class": CLASS_NAMES[pred_cls[best_i]],
                    "conf": pred_conf[best_i],
                    "iou": best_iou,
                    "gt_box": [cx, cy, w, h],
                    "pred_box": list(pred_xywh[best_i]),
                    "small": w * h < SMALL_AREA_THR,
                }
            )
    fp = 0
    fp_pairs = []
    for i, (pc, box, cf) in enumerate(zip(pred_cls, pred_xywh, pred_conf)):
        if i in used_pred:
            continue
        fp += 1
        fp_pairs.append(
            {
                "type": "FP",
                "gt_class": None,
                "pred_class": CLASS_NAMES[pc] if 0 <= pc < 4 else str(pc),
                "conf": cf,
                "iou": 0.0,
                "gt_box": None,
                "pred_box": list(box),
                "small": False,
            }
        )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "localization_poor": localization_poor,
        "fn_small": fn_small,
        "n_detections": len(pred_cls),
        "n_gt": len(gt),
        "pairs": matched_pairs + fp_pairs,
    }


def model_registry(protocol: dict) -> list[dict]:
    """Ordered list of models to evaluate under the frozen protocol."""
    models = []
    fm = protocol["final_model"]
    models.append(
        {
            "model_id": "final_selected",
            "role": "final_selected",
            "display_name": fm["display_name"],
            "checkpoint": fm["checkpoint"],
            "imgsz": int(fm.get("input_resolution", protocol.get("imgsz", 640))),
            "seed": fm.get("seed"),
            "experiment": fm.get("experiment"),
        }
    )
    for key, b in protocol.get("baselines", {}).items():
        models.append(
            {
                "model_id": key,
                "role": "baseline",
                "display_name": b["display_name"],
                "checkpoint": b["checkpoint"],
                "imgsz": int(b.get("input_resolution", protocol.get("imgsz", 640))),
                "seed": b.get("seed"),
                "experiment": key,
            }
        )
    w5 = protocol.get("week5_best_ablation", {})
    if not w5.get("identical_to_final"):
        models.append(
            {
                "model_id": "week5_best_ablation",
                "role": "week5_best_ablation",
                "display_name": w5.get("display_name", "Week 5 best"),
                "checkpoint": w5["checkpoint"],
                "imgsz": int(protocol.get("imgsz", 640)),
                "seed": None,
                "experiment": w5.get("experiment"),
            }
        )
    else:
        # Still list for comparison table, but mark identical
        models.append(
            {
                "model_id": "week5_best_ablation",
                "role": "week5_best_ablation",
                "display_name": w5.get("display_name", "Week 5 best"),
                "checkpoint": w5["checkpoint"],
                "imgsz": int(protocol.get("imgsz", 640)),
                "seed": fm.get("seed"),
                "experiment": w5.get("experiment"),
                "identical_to_final": True,
            }
        )
    return models
