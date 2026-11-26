#!/usr/bin/env python3
"""Week 6 — Frozen protocol: val-only confidence selection, then one-shot test eval.

Order (enforced):
  1. Load frozen protocol
  2. Select confidence threshold on VALIDATION only (writes into protocol + lock file)
  3. Record validation metrics for each model (labeled Validation)
  4. Run held-out TEST evaluation exactly once per model (labeled Test)
  5. Country breakdown (labeled Validation / Test; no new split)
  6. Write CSVs / JSON / manifest

Never retunes after test results are written. Re-runs refuse to overwrite
final_test_results.json unless --force-rerun is passed (discouraged; documents violation).
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CLASS_NAMES
from src.evaluation.week6_utils import (
    extract_val_metrics,
    f1_score,
    load_protocol,
    match_predictions,
    model_registry,
    parse_yolo_label,
    pending,
    resolve_ckpt,
    to_float,
    write_csv,
    write_json,
)
from src.paths import (
    default_configs_dir,
    ensure_legacy_fixture_data_yaml,
    legacy_fixture_processed_dir,
    legacy_fixture_splits_dir,
)
from src.training.baseline import gpu_name, pick_device, software_versions

WEEK6 = ROOT / "results" / "week6"
FIGDIR = ROOT / "figures" / "week6"
LOCK = WEEK6 / "CONFIDENCE_THRESHOLD_LOCKED.json"
TEST_DONE = WEEK6 / "TEST_EVALUATION_COMPLETED.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 6 final held-out evaluation.")
    p.add_argument("--protocol", type=str, default=str(ROOT / "configs" / "final_evaluation.yaml"))
    p.add_argument("--data", type=str, default="")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--smoke", action="store_true", help="Limit images for predict galleries / country")
    p.add_argument(
        "--force-rerun",
        action="store_true",
        help="DANGEROUS: overwrite completed test results (documents protocol breach).",
    )
    p.add_argument("--skip-test", action="store_true", help="Only val + conf selection (no test)")
    p.add_argument("--skip-country", action="store_true")
    return p.parse_args()


def file_sha16(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def select_confidence_on_val(model, data_yaml: Path, protocol: dict, device: str, imgsz: int) -> dict:
    """Sweep conf on validation; maximize F1 then precision. Never touches test."""
    candidates = list(protocol.get("conf_selection", {}).get("candidates", [0.25]))
    rows = []
    for conf in candidates:
        res = model.val(
            data=str(data_yaml.resolve()),
            split="val",
            imgsz=imgsz,
            device=device,
            conf=float(conf),
            plots=False,
            verbose=False,
        )
        m = extract_val_metrics(res)
        p, r = m["precision"], m["recall"]
        f1 = f1_score(p, r)
        rows.append(
            {
                "split_label": "Validation",
                "conf": conf,
                "map50": m["map50"],
                "precision": p,
                "recall": r,
                "f1": f1 if not pending(f1) else 0.0,
            }
        )
    # Choose max F1; tie-break higher precision; then prefer 0.25 if still tied
    def key(row):
        f1 = float(row["f1"]) if not pending(row["f1"]) else -1.0
        pr = float(row["precision"]) if not pending(row["precision"]) else -1.0
        prefer = 1.0 if abs(float(row["conf"]) - 0.25) < 1e-9 else 0.0
        return (f1, pr, prefer)

    best = max(rows, key=key)
    return {"rows": rows, "selected_conf": float(best["conf"]), "selected_row": best}


def small_object_ap(model, processed: Path, split: str, imgsz: int, device: str, area_thr: float) -> float | str:
    import yaml as _yaml

    img_dir = processed / "images" / split
    lbl_dir = processed / "labels" / split
    if not img_dir.is_dir():
        return "Pending"
    tmp = Path(tempfile.mkdtemp(prefix="w6_small_"))
    try:
        timg = tmp / "images" / split
        tlbl = tmp / "labels" / split
        timg.mkdir(parents=True)
        tlbl.mkdir(parents=True)
        n_kept = 0
        for lbl in lbl_dir.glob("*.txt"):
            lines_out = []
            for line in lbl.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                if float(parts[3]) * float(parts[4]) < area_thr:
                    lines_out.append(line.strip())
                    n_kept += 1
            if not lines_out:
                continue
            stem = lbl.stem
            src = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                cand = img_dir / f"{stem}{ext}"
                if cand.is_file():
                    src = cand
                    break
            if src is None:
                continue
            (tlbl / lbl.name).write_text("\n".join(lines_out) + "\n", encoding="utf-8")
            dest = timg / src.name
            try:
                dest.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dest)
        if n_kept == 0:
            return "Pending"
        dy = tmp / "data.yaml"
        dy.write_text(
            _yaml.safe_dump(
                {
                    "path": str(tmp),
                    "train": f"images/{split}",
                    "val": f"images/{split}",
                    "nc": 4,
                    "names": {i: n for i, n in enumerate(CLASS_NAMES)},
                }
            ),
            encoding="utf-8",
        )
        res = model.val(data=str(dy), split="val", imgsz=imgsz, device=device, plots=False, verbose=False)
        m = extract_val_metrics(res)
        return m["map50"]
    except Exception as exc:  # noqa: BLE001
        print(f"  small-object AP failed ({split}): {exc}", file=sys.stderr)
        return "Pending"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def count_fp_fn(model, processed: Path, split: str, imgsz: int, device: str, conf: float, iou_thr: float, max_images: int | None):
    img_dir = processed / "images" / split
    lbl_dir = processed / "labels" / split
    images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if max_images is not None:
        images = images[:max_images]
    tot = {"tp": 0, "fp": 0, "fn": 0, "localization_poor": 0, "fn_small": 0, "n_detections": 0, "n_gt": 0}
    pred_rows = []
    for img in images:
        gt = parse_yolo_label(lbl_dir / f"{img.stem}.txt")
        preds = model.predict(source=str(img), imgsz=imgsz, conf=conf, device=device, verbose=False)
        pred_cls, pred_xywh, pred_conf = [], [], []
        if preds and preds[0].boxes is not None and len(preds[0].boxes):
            pred_cls = [int(c) for c in preds[0].boxes.cls.cpu().tolist()]
            pred_xywh = preds[0].boxes.xywhn.cpu().tolist()
            pred_conf = [float(c) for c in preds[0].boxes.conf.cpu().tolist()]
        stats = match_predictions(gt, pred_cls, pred_xywh, pred_conf, iou_thr=iou_thr)
        for k in tot:
            tot[k] += stats[k]
        for i, (pc, box, cf) in enumerate(zip(pred_cls, pred_xywh, pred_conf)):
            pred_rows.append(
                {
                    "split_label": "Test" if split == "test" else "Validation",
                    "split": split,
                    "image": str(img),
                    "image_stem": img.stem,
                    "pred_class": CLASS_NAMES[pc] if 0 <= pc < 4 else str(pc),
                    "confidence": cf,
                    "cx": box[0],
                    "cy": box[1],
                    "w": box[2],
                    "h": box[3],
                }
            )
    return tot, pred_rows


def country_metrics(
    model,
    processed: Path,
    split_csv: Path,
    split: str,
    split_label: str,
    imgsz: int,
    device: str,
    conf: float,
    data_yaml: Path,
    smoke: bool,
) -> list[dict]:
    """Per-country image/object counts + proxy box-recall; optional mAP if ≥1 country subset.

    Does not create a new split. Uses existing split CSVs only.
    """
    from src.evaluation.week6_utils import iou_xywh

    if not split_csv.is_file():
        return [
            {
                "split_label": split_label,
                "split": split,
                "country": "ALL",
                "status": "Pending",
                "reason": f"missing {split_csv}",
            }
        ]
    df = pd.read_csv(split_csv)
    if "country" not in df.columns:
        return [
            {
                "split_label": split_label,
                "split": split,
                "country": "ALL",
                "status": "unavailable",
                "reason": "no country column — cannot analyze without new metadata",
            }
        ]

    rows = []
    for country, sub in df.groupby("country"):
        n_img = 0
        n_gt = 0
        n_hit = 0
        per_class_gt = {c: 0 for c in CLASS_NAMES}
        per_class_hit = {c: 0 for c in CLASS_NAMES}
        limit = list(sub.itertuples(index=False))
        if smoke:
            limit = limit[: min(8, len(limit))]
        for row in limit:
            img = processed / "images" / split / f"{row.country}__{row.filename}"
            if not img.is_file():
                # materialized stem may already include country
                alt = processed / "images" / split / row.filename
                img = alt if alt.is_file() else img
            if not img.is_file():
                # try any matching stem
                matches = list((processed / "images" / split).glob(f"*{Path(row.filename).stem}*"))
                img = matches[0] if matches else img
            if not img.is_file():
                continue
            lbl = processed / "labels" / split / f"{img.stem}.txt"
            gt = parse_yolo_label(lbl)
            n_img += 1
            preds = model.predict(source=str(img), imgsz=imgsz, conf=conf, device=device, verbose=False)
            pred_boxes = []
            if preds and preds[0].boxes is not None and len(preds[0].boxes):
                xywhn = preds[0].boxes.xywhn.cpu().tolist()
                cls = preds[0].boxes.cls.cpu().tolist()
                pred_boxes = [(int(c), *b) for c, b in zip(cls, xywhn)]
            for cid, cx, cy, w, h in gt:
                n_gt += 1
                cname = CLASS_NAMES[cid] if 0 <= cid < 4 else str(cid)
                per_class_gt[cname] = per_class_gt.get(cname, 0) + 1
                hit = False
                for pc, px, py, pw, ph in pred_boxes:
                    if pc == cid and iou_xywh((cx, cy, w, h), (px, py, pw, ph)) >= 0.5:
                        hit = True
                        break
                if hit:
                    n_hit += 1
                    per_class_hit[cname] = per_class_hit.get(cname, 0) + 1
        row_out = {
            "split_label": split_label,
            "split": split,
            "country": str(country),
            "status": "ok",
            "n_images": n_img,
            "n_objects": n_gt,
            "box_recall_iou50": (n_hit / n_gt) if n_gt else None,
            "map50": "Pending",  # single-country official mAP requires subset yaml; Pending unless filled
            "map50_95": "Pending",
            "note": "box_recall is a proxy; official mAP Pending for single-country subset unless computed",
            "country_held_out_protocol": False,
        }
        for c in CLASS_NAMES:
            row_out[f"ap_proxy_{c}"] = (
                (per_class_hit.get(c, 0) / per_class_gt[c]) if per_class_gt.get(c) else None
            )
            row_out[f"n_gt_{c}"] = per_class_gt.get(c, 0)
        rows.append(row_out)

    # Summary note for test=US-only
    if split == "test" and len(rows) == 1:
        rows[0]["note"] = (
            "Main test split contains only this country (grouped 70/15/15). "
            "This is incidental composition, NOT a designed country-held-out OOD protocol. "
            "No new split was created."
        )
        rows[0]["country_held_out_protocol"] = False
    return rows


def plot_country(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    usable = [r for r in rows if r.get("status") == "ok" and r.get("box_recall_iou50") is not None]
    fig, ax = plt.subplots(figsize=(10, 5))
    if not usable:
        ax.text(0.5, 0.5, "Pending / unavailable", ha="center", va="center", transform=ax.transAxes)
    else:
        labels = [f"{r['split_label']}:{r['country']}" for r in usable]
        vals = [float(r["box_recall_iou50"]) for r in usable]
        colors = ["#4C78A8" if r["split_label"] == "Validation" else "#F58518" for r in usable]
        ax.bar(range(len(labels)), vals, color=colors)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Box recall @ IoU 0.5 (proxy)")
        ax.set_title("Week 6 — country performance (Validation vs Test labeled)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def update_protocol_yaml(path: Path, conf: float, test_done: bool) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    data["confidence_threshold"] = float(conf)
    data["conf_selection"] = data.get("conf_selection") or {}
    data["conf_selection"]["status"] = "selected_on_validation"
    data["conf_selection"]["selected_conf"] = float(conf)
    data["test_evaluation_completed"] = bool(test_done)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    protocol_path = Path(args.protocol)
    protocol = load_protocol(protocol_path)
    WEEK6.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)

    if TEST_DONE.is_file() and not args.force_rerun and not args.skip_test:
        print(
            f"Test evaluation already completed ({TEST_DONE}). "
            "Refusing to rerun (protocol: one-shot test). "
            "Pass --force-rerun only to document a breach, or --skip-test for val-only.",
            file=sys.stderr,
        )
        return 2

    # Frozen Week 6 protocol = LEGACY synthetic-fixture evaluation.
    data_yaml = Path(args.data) if args.data else ensure_legacy_fixture_data_yaml()
    if not data_yaml.is_absolute():
        data_yaml = ROOT / data_yaml
    processed = legacy_fixture_processed_dir()
    device = pick_device(args.device)
    gname = gpu_name(device)
    imgsz = int(protocol.get("imgsz", 640))
    iou_nms = float(protocol.get("iou_threshold", 0.7))
    area_thr = float(protocol.get("small_object_area_threshold", 0.02))
    conf_map = float(protocol.get("conf_for_map", 0.001))

    # Verify split hashes vs Week 5
    split_hashes = {
        s: file_sha16(legacy_fixture_splits_dir() / f"{s}.csv") for s in ("train", "val", "test")
    }
    print(f"Device={device} GPU={gname}")
    print(f"Split hashes={split_hashes}")

    from ultralytics import YOLO

    models = model_registry(protocol)
    # Deduplicate identical checkpoints for expensive test pass but keep table rows
    final_ckpt = resolve_ckpt(protocol["final_model"]["checkpoint"])
    if final_ckpt is None:
        print(f"Missing final checkpoint: {protocol['final_model']['checkpoint']}", file=sys.stderr)
        return 1

    # --- A. Confidence selection on VALIDATION (final model only) ---
    if LOCK.is_file() and not args.force_rerun:
        lock = json_load(LOCK)
        selected_conf = float(lock["selected_conf"])
        conf_rows = lock.get("sweep_rows", [])
        print(f"Using locked confidence threshold={selected_conf} from {LOCK}")
    else:
        print("Selecting confidence threshold on VALIDATION only …")
        final_model = YOLO(str(final_ckpt))
        sel = select_confidence_on_val(final_model, data_yaml, protocol, device, imgsz)
        selected_conf = sel["selected_conf"]
        conf_rows = sel["rows"]
        lock_payload = {
            "selected_conf": selected_conf,
            "criterion": protocol.get("conf_selection", {}).get("criterion"),
            "split": "val",
            "split_label": "Validation",
            "test_used": False,
            "sweep_rows": conf_rows,
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(final_ckpt),
        }
        write_json(LOCK, lock_payload)
        write_csv(WEEK6 / "confidence_threshold_val.csv", conf_rows)
        update_protocol_yaml(protocol_path, selected_conf, test_done=False)
        print(f"Locked confidence_threshold={selected_conf} (validation only)")

    protocol = load_protocol(protocol_path)  # refresh
    results_rows = []
    predictions_all = []
    country_all = []
    per_model_detail = {}

    evaluated_ckpts: set[str] = set()

    for spec in models:
        ckpt = resolve_ckpt(spec["checkpoint"])
        mid = spec["model_id"]
        print(f"\n=== {mid}: {spec['display_name']} ===")
        if ckpt is None:
            for split_label, split in (("Validation", "val"), ("Test", "test")):
                results_rows.append(
                    {
                        "model_id": mid,
                        "display_name": spec["display_name"],
                        "role": spec["role"],
                        "split_label": split_label,
                        "split": split,
                        "status": "Pending",
                        "reason": f"missing checkpoint {spec['checkpoint']}",
                        "seed": spec.get("seed"),
                        "imgsz": spec["imgsz"],
                        "confidence_threshold": selected_conf,
                        "iou_threshold": iou_nms,
                    }
                )
            continue

        # Skip duplicate compute for identical_to_final — copy final metrics later
        if spec.get("identical_to_final") and "final_selected" in per_model_detail:
            print("  identical to final_selected — reusing metrics (no extra test pass)")
            detail = dict(per_model_detail["final_selected"])
            detail["model_id"] = mid
            detail["display_name"] = spec["display_name"]
            detail["role"] = spec["role"]
            detail["identical_to_final"] = True
            per_model_detail[mid] = detail
            for row in detail.get("result_rows", []):
                r = dict(row)
                r["model_id"] = mid
                r["display_name"] = spec["display_name"]
                r["role"] = spec["role"]
                r["identical_to_final"] = True
                results_rows.append(r)
            continue

        model = YOLO(str(ckpt))
        detail = {"checkpoint": str(ckpt), "result_rows": [], "predictions": [], "country": []}

        for split_label, split in (("Validation", "val"), ("Test", "test")):
            if split == "test" and args.skip_test:
                results_rows.append(
                    {
                        "model_id": mid,
                        "display_name": spec["display_name"],
                        "role": spec["role"],
                        "split_label": "Test",
                        "split": "test",
                        "status": "Pending",
                        "reason": "--skip-test",
                        "seed": spec.get("seed"),
                        "confidence_threshold": selected_conf,
                        "iou_threshold": iou_nms,
                    }
                )
                continue

            print(f"  Evaluating {split_label} …")
            try:
                res = model.val(
                    data=str(data_yaml.resolve()),
                    split=split,
                    imgsz=spec["imgsz"],
                    device=device,
                    conf=conf_map,
                    iou=iou_nms,
                    plots=True,
                    verbose=False,
                )
                metrics = extract_val_metrics(res)
                soap = small_object_ap(model, processed, split, spec["imgsz"], device, area_thr)
                fpfn, pred_rows = count_fp_fn(
                    model,
                    processed,
                    split,
                    spec["imgsz"],
                    device,
                    selected_conf,
                    iou_thr=0.5,
                    max_images=12 if args.smoke else None,
                )
                # tag predictions
                for pr in pred_rows:
                    pr["model_id"] = mid
                    pr["display_name"] = spec["display_name"]
                predictions_all.extend(pred_rows)
                detail["predictions"].extend(pred_rows)

                # Copy PR / confusion plots for final model test
                if mid == "final_selected" and hasattr(res, "save_dir") and res.save_dir:
                    src = Path(res.save_dir)
                    for name, dest_name in (
                        ("confusion_matrix.png", "confusion_matrix.png"),
                        ("BoxPR_curve.png", "precision_recall_curves.png"),
                    ):
                        if (src / name).is_file():
                            shutil.copy2(src / name, FIGDIR / dest_name)

                row = {
                    "model_id": mid,
                    "display_name": spec["display_name"],
                    "role": spec["role"],
                    "split_label": split_label,
                    "split": split,
                    "status": "completed",
                    "seed": spec.get("seed"),
                    "imgsz": spec["imgsz"],
                    "confidence_threshold_operating": selected_conf,
                    "conf_for_map": conf_map,
                    "iou_threshold": iou_nms,
                    "map50": metrics["map50"],
                    "map50_95": metrics["map50_95"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "small_object_ap": soap,
                    "n_detections": fpfn["n_detections"],
                    "false_positive_count": fpfn["fp"],
                    "false_negative_count": fpfn["fn"],
                    "true_positive_count": fpfn["tp"],
                    "localization_poor_count": fpfn["localization_poor"],
                    "fn_small_count": fpfn["fn_small"],
                    "checkpoint": str(ckpt),
                    "device": device,
                    "gpu": gname,
                    "test_used_for_selection": False,
                }
                for c in CLASS_NAMES:
                    row[f"ap50_{c}"] = metrics["per_class_ap50"].get(c, "Pending")
                    row[f"precision_{c}"] = metrics["per_class_precision"].get(c, "Pending")
                    row[f"recall_{c}"] = metrics["per_class_recall"].get(c, "Pending")
                results_rows.append(row)
                detail["result_rows"].append(row)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED {split_label}: {exc}", file=sys.stderr)
                results_rows.append(
                    {
                        "model_id": mid,
                        "display_name": spec["display_name"],
                        "role": spec["role"],
                        "split_label": split_label,
                        "split": split,
                        "status": "error",
                        "reason": str(exc),
                        "confidence_threshold": selected_conf,
                    }
                )

        if not args.skip_country:
            for split_label, split, csv_name in (
                ("Validation", "val", "val.csv"),
                ("Test", "test", "test.csv"),
            ):
                if split == "test" and args.skip_test:
                    continue
                crow = country_metrics(
                    model,
                    processed,
                    legacy_fixture_splits_dir() / csv_name,
                    split,
                    split_label,
                    spec["imgsz"],
                    device,
                    selected_conf,
                    data_yaml,
                    smoke=args.smoke,
                )
                for r in crow:
                    r["model_id"] = mid
                    r["display_name"] = spec["display_name"]
                country_all.extend(crow)
                detail["country"].extend(crow)

        per_model_detail[mid] = detail
        evaluated_ckpts.add(str(ckpt))

    # Persist
    write_csv(WEEK6 / "final_test_results.csv", results_rows)
    write_json(
        WEEK6 / "final_test_results.json",
        {
            "protocol_id": protocol.get("protocol_id"),
            "confidence_threshold": selected_conf,
            "iou_threshold": iou_nms,
            "conf_selected_on": "Validation",
            "test_used_for_selection": False,
            "split_hashes": split_hashes,
            "rows": results_rows,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    write_csv(WEEK6 / "final_predictions.csv", predictions_all)
    write_csv(WEEK6 / "country_results.csv", country_all)
    write_json(WEEK6 / "country_results.json", {"rows": country_all, "new_split_created": False})
    plot_country([r for r in country_all if r.get("model_id") == "final_selected"], FIGDIR / "country_performance.png")

    test_completed = (not args.skip_test) and any(
        r.get("split_label") == "Test" and r.get("status") == "completed" for r in results_rows
    )
    if test_completed:
        write_json(
            TEST_DONE,
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "models_evaluated": sorted(evaluated_ckpts),
                "confidence_threshold": selected_conf,
                "note": "One-shot test evaluation recorded. Do not retune.",
            },
        )
        update_protocol_yaml(protocol_path, selected_conf, test_done=True)

    manifest = {
        "week": 6,
        "protocol_path": str(protocol_path),
        "protocol_id": protocol.get("protocol_id"),
        "started_finish": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpu": gname,
        "software": software_versions(),
        "confidence_threshold": selected_conf,
        "iou_threshold": iou_nms,
        "split_hashes": split_hashes,
        "test_evaluation_completed": test_completed,
        "test_used_for_selection": False,
        "force_rerun": bool(args.force_rerun),
        "smoke": bool(args.smoke),
        "outputs": {
            "final_test_results.csv": str(WEEK6 / "final_test_results.csv"),
            "final_test_results.json": str(WEEK6 / "final_test_results.json"),
            "final_predictions.csv": str(WEEK6 / "final_predictions.csv"),
            "country_results.csv": str(WEEK6 / "country_results.csv"),
            "confidence_lock": str(LOCK),
        },
    }
    write_json(WEEK6 / "final_run_manifest.json", manifest)
    print(f"\nWrote {WEEK6 / 'final_test_results.csv'} ({len(results_rows)} rows)")
    print(f"Confidence locked at {selected_conf} (validation). Test completed={test_completed}")
    return 0


def json_load(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
