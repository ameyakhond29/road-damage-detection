#!/usr/bin/env python3
"""Week 5 — Aggregate validation metrics, small-object AP, figures, seed summaries.

IMPORTANT: Evaluates the **validation** split only for model comparison / selection.
Test-set evaluation is deferred to Week 6. This script refuses --split test unless
--allow-test is passed (for Week 6 only).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CLASS_NAMES
from src.models.resolve import MODEL_REGISTRY
from src.paths import ensure_legacy_fixture_data_yaml, legacy_fixture_processed_dir
from src.training.baseline import pick_device, to_float, write_json

SMALL_AREA_THR = 0.02  # normalized box area
EXPERIMENTS = list(MODEL_REGISTRY.keys())
SEEDS = [0, 1, 2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate / aggregate Week 5 ablations (val only).")
    p.add_argument("--data", type=str, default="",
                   help="Ultralytics YAML (default: legacy fixture YAML — Week 5 results are fixture artifacts)")
    p.add_argument("--week5-root", type=str, default=str(ROOT / "results" / "week5"))
    p.add_argument("--figures-dir", type=str, default=str(ROOT / "figures" / "week5"))
    p.add_argument("--processed-dir", type=str, default=str(legacy_fixture_processed_dir()))
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument(
        "--allow-test",
        action="store_true",
        help="Permit test-split evaluation (Week 6 only). Default: blocked.",
    )
    p.add_argument("--device", type=str, default="")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--n-examples", type=int, default=8)
    p.add_argument("--skip-predict-examples", action="store_true")
    p.add_argument("--recompute-val", action="store_true", help="Re-run model.val on each ckpt")
    return p.parse_args()


def pending(v):
    return v is None or v == "Pending" or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))


def load_seed_metrics(week5: Path, experiment: str, seed: int) -> dict | None:
    path = week5 / experiment / f"seed_{seed}" / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_ckpt(card: dict, week5: Path, experiment: str, seed: int) -> Path | None:
    ckpt = card.get("best_checkpoint")
    if isinstance(ckpt, str) and Path(ckpt).is_file():
        return Path(ckpt)
    # Fallback Ultralytics layout
    cand = week5 / experiment / f"seed_{seed}" / "train" / "weights" / "best.pt"
    return cand if cand.is_file() else None


def small_object_ap50(
    model,
    processed: Path,
    split: str,
    imgsz: int,
    device: str,
    area_thr: float = SMALL_AREA_THR,
) -> float | str:
    """Approximate small-object AP@0.5 via filtered val labels (area < thr).

    Builds a temporary YOLO data yaml pointing at the same images but only
    keeps GT boxes with normalized area < area_thr. Returns mAP50 or Pending.
    """
    import shutil
    import tempfile

    import yaml
    from ultralytics import YOLO  # noqa: F401 — model already loaded

    img_dir = processed / "images" / split
    lbl_dir = processed / "labels" / split
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        return "Pending"

    tmp = Path(tempfile.mkdtemp(prefix="w5_small_"))
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
                w, h = float(parts[3]), float(parts[4])
                if w * h < area_thr:
                    lines_out.append(line.strip())
                    n_kept += 1
            stem = lbl.stem
            # symlink / copy image if any small boxes OR keep empty for negatives? skip empties with no small
            src_img = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                cand = img_dir / f"{stem}{ext}"
                if cand.is_file():
                    src_img = cand
                    break
            if src_img is None:
                continue
            if not lines_out:
                continue  # only evaluate images that contain ≥1 small GT
            (tlbl / lbl.name).write_text("\n".join(lines_out) + "\n", encoding="utf-8")
            dest = timg / src_img.name
            try:
                dest.symlink_to(src_img.resolve())
            except OSError:
                shutil.copy2(src_img, dest)

        if n_kept == 0:
            return "Pending"

        data = {
            "path": str(tmp),
            "train": f"images/{split}",  # unused
            "val": f"images/{split}",
            "nc": 4,
            "names": {0: "D00", 1: "D10", 2: "D20", 3: "D40"},
        }
        dy = tmp / "data.yaml"
        dy.write_text(yaml.safe_dump(data), encoding="utf-8")
        res = model.val(data=str(dy), split="val", imgsz=imgsz, device=device, plots=False, verbose=False)
        if hasattr(res, "results_dict") and res.results_dict:
            return to_float(res.results_dict.get("metrics/mAP50(B)", "Pending"))
        return "Pending"
    except Exception as exc:  # noqa: BLE001
        print(f"  small-object AP failed: {exc}", file=sys.stderr)
        return "Pending"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def f1_of(p, r):
    if pending(p) or pending(r):
        return "Pending"
    p, r = float(p), float(r)
    if p + r <= 0:
        return 0.0
    return round(2 * p * r / (p + r), 6)


def collect_rows(week5: Path, recompute: bool, data_yaml: Path, processed: Path, device: str, split: str) -> list[dict]:
    rows = []
    for exp in EXPERIMENTS:
        for seed in SEEDS:
            card = load_seed_metrics(week5, exp, seed)
            row = {
                "experiment": exp,
                "model_variant": MODEL_REGISTRY[exp]["label"],
                "seed": seed,
                "input_resolution": MODEL_REGISTRY[exp]["imgsz_default"],
                "status": "Pending",
                "batch_size": "Pending",
                "epochs": "Pending",
                "learning_rate": "Pending",
                "optimizer": "Pending",
                "gpu": "Pending",
                "training_duration_sec": "Pending",
                "parameter_count": "Pending",
                "gflops": "Pending",
                "peak_gpu_memory_mb": "Pending",
                "inference_latency_ms": "Pending",
                "map50": "Pending",
                "map50_95": "Pending",
                "precision": "Pending",
                "recall": "Pending",
                "f1": "Pending",
                "ap_D00": "Pending",
                "ap_D10": "Pending",
                "ap_D20": "Pending",
                "ap_D40": "Pending",
                "small_object_ap": "Pending",
                "best_checkpoint": "Pending",
                "split_evaluated": split,
                "test_used_for_selection": False,
            }
            if card is None:
                rows.append(row)
                continue

            row["status"] = card.get("status", "Pending")
            row["batch_size"] = card.get("batch_size", "Pending")
            row["epochs"] = card.get("epochs", "Pending")
            row["learning_rate"] = card.get("learning_rate", "Pending")
            row["optimizer"] = card.get("optimizer", "Pending")
            row["gpu"] = card.get("gpu", "Pending")
            row["input_resolution"] = card.get("input_resolution", row["input_resolution"])
            row["training_duration_sec"] = card.get("training_duration_sec", "Pending")
            row["parameter_count"] = card.get("parameter_count", "Pending")
            row["gflops"] = card.get("gflops", "Pending")
            row["peak_gpu_memory_mb"] = card.get("peak_gpu_memory_mb", "Pending")
            lat = card.get("inference_latency")
            if isinstance(lat, dict):
                row["inference_latency_ms"] = lat.get("latency_ms_mean", "Pending")
            row["map50"] = card.get("map50", "Pending")
            row["map50_95"] = card.get("map50_95", "Pending")
            row["precision"] = card.get("precision", "Pending")
            row["recall"] = card.get("recall", "Pending")
            row["f1"] = card.get("f1", f1_of(row["precision"], row["recall"]))
            per = card.get("per_class_ap") or {}
            if isinstance(per, dict):
                for c in CLASS_NAMES:
                    row[f"ap_{c}"] = per.get(c, "Pending")
            row["small_object_ap"] = card.get("small_object_ap", "Pending")
            row["best_checkpoint"] = card.get("best_checkpoint", "Pending")

            ckpt = resolve_ckpt(card, week5, exp, seed)
            if recompute and ckpt and ckpt.is_file() and split == "val":
                try:
                    from ultralytics import YOLO

                    model = YOLO(str(ckpt))
                    imgsz = int(row["input_resolution"]) if not pending(row["input_resolution"]) else 640
                    # Prefer requested device; fall back to CPU on MPS/CUDA OOM
                    eval_device = device
                    try:
                        res = model.val(
                            data=str(Path(data_yaml).resolve()),
                            split="val",
                            imgsz=imgsz,
                            device=eval_device,
                            plots=False,
                            verbose=False,
                        )
                    except RuntimeError as oom:
                        if "out of memory" not in str(oom).lower():
                            raise
                        print(f"  OOM on {eval_device} for {exp}; retrying on cpu", file=sys.stderr)
                        eval_device = "cpu"
                        res = model.val(
                            data=str(Path(data_yaml).resolve()),
                            split="val",
                            imgsz=imgsz,
                            device=eval_device,
                            plots=False,
                            verbose=False,
                        )
                    if hasattr(res, "results_dict") and res.results_dict:
                        m = {k: to_float(v) for k, v in res.results_dict.items()}
                        row["map50"] = m.get("metrics/mAP50(B)", row["map50"])
                        row["map50_95"] = m.get("metrics/mAP50-95(B)", row["map50_95"])
                        row["precision"] = m.get("metrics/precision(B)", row["precision"])
                        row["recall"] = m.get("metrics/recall(B)", row["recall"])
                        row["f1"] = f1_of(row["precision"], row["recall"])
                        try:
                            box = res.box
                            if hasattr(box, "ap50") and box.ap50 is not None:
                                for i, c in enumerate(CLASS_NAMES):
                                    if i < len(box.ap50):
                                        row[f"ap_{c}"] = to_float(box.ap50[i])
                        except Exception:  # noqa: BLE001
                            pass
                    soap = small_object_ap50(model, Path(processed), "val", imgsz, eval_device)
                    row["small_object_ap"] = soap
                    # Persist back into seed metrics
                    card["small_object_ap"] = soap
                    card["map50"] = row["map50"]
                    card["map50_95"] = row["map50_95"]
                    card["precision"] = row["precision"]
                    card["recall"] = row["recall"]
                    card["f1"] = row["f1"]
                    card["per_class_ap"] = {c: row[f"ap_{c}"] for c in CLASS_NAMES}
                    write_json(week5 / exp / f"seed_{seed}" / "metrics.json", card)
                except Exception as exc:  # noqa: BLE001
                    print(f"  recompute failed {exp} seed={seed}: {exc}", file=sys.stderr)

            rows.append(row)
    return rows


def seed_summary(rows: list[dict]) -> list[dict]:
    by_exp: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_exp[r["experiment"]].append(r)

    # Baseline means for improvement column
    base_maps = [
        float(r["map50"])
        for r in by_exp.get("yolo11n_640", [])
        if not pending(r["map50"])
    ]
    base_mean = float(np.mean(base_maps)) if base_maps else None

    summaries = []
    metrics = [
        "map50",
        "map50_95",
        "precision",
        "recall",
        "f1",
        "ap_D00",
        "ap_D10",
        "ap_D20",
        "ap_D40",
        "small_object_ap",
        "inference_latency_ms",
        "parameter_count",
        "gflops",
        "training_duration_sec",
    ]
    for exp, rs in by_exp.items():
        summary = {
            "experiment": exp,
            "model_variant": MODEL_REGISTRY[exp]["label"],
            "n_seeds_completed": sum(1 for r in rs if not pending(r["map50"])),
            "improvement_map50_vs_baseline": "Pending",
        }
        for m in metrics:
            vals = [float(r[m]) for r in rs if not pending(r.get(m))]
            if not vals:
                summary[f"{m}_mean"] = "Pending"
                summary[f"{m}_std"] = "Pending"
                summary[f"{m}_best"] = "Pending"
                summary[f"{m}_worst"] = "Pending"
            else:
                summary[f"{m}_mean"] = round(float(np.mean(vals)), 6)
                summary[f"{m}_std"] = round(float(np.std(vals, ddof=0)), 6)
                # For latency/params/gflops/time: "best" = lowest; for AP metrics: highest
                if m in {"inference_latency_ms", "parameter_count", "gflops", "training_duration_sec"}:
                    summary[f"{m}_best"] = round(min(vals), 6)
                    summary[f"{m}_worst"] = round(max(vals), 6)
                else:
                    summary[f"{m}_best"] = round(max(vals), 6)
                    summary[f"{m}_worst"] = round(min(vals), 6)

        if base_mean is not None and not pending(summary["map50_mean"]):
            summary["improvement_map50_vs_baseline"] = round(
                float(summary["map50_mean"]) - base_mean, 6
            )
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_bar(path: Path, labels, values, ylabel, title, baseline_idx=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    colors = ["#4C78A8" if i != baseline_idx else "#F58518" for i in range(len(labels))]
    # Replace Pending with nan
    y = [np.nan if pending(v) else float(v) for v in values]
    ax.bar(x, y, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(summaries: list[dict], rows: list[dict], figdir: Path, week5: Path) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    # Order experiments
    order = EXPERIMENTS
    smap = {s["experiment"]: s for s in summaries}
    labels = [MODEL_REGISTRY[e]["label"] for e in order]

    plot_bar(
        figdir / "map_comparison.png",
        labels,
        [smap.get(e, {}).get("map50_mean", "Pending") for e in order],
        "val mAP@0.5 (mean over seeds)",
        "Week 5 ablation — validation mAP@0.5",
    )

    # Per-class grouped bars
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.18
    x = np.arange(len(CLASS_NAMES))
    for i, e in enumerate(order):
        vals = []
        for c in CLASS_NAMES:
            v = smap.get(e, {}).get(f"ap_{c}_mean", "Pending")
            vals.append(np.nan if pending(v) else float(v))
        ax.bar(x + (i - 2) * width, vals, width, label=MODEL_REGISTRY[e]["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("AP@0.5 (val mean)")
    ax.set_title("Week 5 — per-class AP@0.5")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "per_class_ap_comparison.png", dpi=150)
    plt.close(fig)

    plot_bar(
        figdir / "small_object_ap_comparison.png",
        labels,
        [smap.get(e, {}).get("small_object_ap_mean", "Pending") for e in order],
        "val small-object AP@0.5 (mean)",
        "Week 5 — small-object AP@0.5 (area < 0.02)",
    )

    # Accuracy vs efficiency scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for e in order:
        s = smap.get(e, {})
        m = s.get("map50_mean", "Pending")
        g = s.get("gflops_mean", "Pending")
        if pending(m) or pending(g):
            continue
        ax.scatter(float(g), float(m), s=80, label=MODEL_REGISTRY[e]["label"])
        ax.annotate(e, (float(g), float(m)), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("GFLOPs (mean)")
    ax.set_ylabel("val mAP@0.5 (mean)")
    ax.set_title("Week 5 — accuracy vs efficiency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "accuracy_efficiency_tradeoff.png", dpi=150)
    plt.close(fig)

    # Training curves: overlay results.csv from seed_0 when present
    fig, ax = plt.subplots(figsize=(10, 5))
    any_curve = False
    for e in order:
        csv_path = week5 / e / "seed_0" / "results.csv"
        if not csv_path.is_file():
            # Ultralytics may nest under train/
            alt = week5 / e / "seed_0" / "train" / "results.csv"
            csv_path = alt if alt.is_file() else csv_path
        if not csv_path.is_file():
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            col = None
            for c in df.columns:
                if "mAP50" in c and "95" not in c:
                    col = c
                    break
            if col is None:
                continue
            ax.plot(df.index + 1, df[col], label=MODEL_REGISTRY[e]["label"])
            any_curve = True
        except Exception:  # noqa: BLE001
            continue
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Week 5 — training curves (seed 0)")
    if any_curve:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Pending — no results.csv yet", ha="center", va="center", transform=ax.transAxes)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "training_curves.png", dpi=150)
    plt.close(fig)


def qualitative_examples(week5: Path, figdir: Path, processed: Path, device: str, n: int, conf: float) -> None:
    """Save a few val prediction overlays for each experiment (seed 0) when ckpt exists."""
    from ultralytics import YOLO

    out = figdir / "predictions"
    out.mkdir(parents=True, exist_ok=True)
    val_img_dir = Path(processed) / "images" / "val"
    if not val_img_dir.is_dir():
        return
    images = sorted(val_img_dir.glob("*"))[: max(n, 1)]
    for exp in EXPERIMENTS:
        card = load_seed_metrics(week5, exp, 0)
        if not card:
            continue
        ckpt = resolve_ckpt(card, week5, exp, 0)
        if not ckpt:
            continue
        imgsz = int(card.get("input_resolution") or MODEL_REGISTRY[exp]["imgsz_default"])
        try:
            model = YOLO(str(ckpt))
        except Exception:  # noqa: BLE001
            continue
        exp_dir = out / exp
        exp_dir.mkdir(parents=True, exist_ok=True)
        for i, img_path in enumerate(images[:n]):
            try:
                res = model.predict(
                    source=str(img_path),
                    imgsz=imgsz,
                    conf=conf,
                    device=device,
                    verbose=False,
                )[0]
                plotted = res.plot()
                Image.fromarray(plotted[..., ::-1]).save(exp_dir / f"val_{i:02d}_{img_path.stem}.jpg")
            except Exception as exc:  # noqa: BLE001
                print(f"  predict example failed: {exc}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        print(
            "Refusing test-split evaluation in Week 5. "
            "Use Week 6 (or pass --allow-test explicitly).",
            file=sys.stderr,
        )
        return 2

    week5 = Path(args.week5_root)
    figdir = Path(args.figures_dir)
    device = pick_device(args.device)
    data_yaml = Path(args.data) if args.data else ensure_legacy_fixture_data_yaml()

    rows = collect_rows(
        week5,
        recompute=args.recompute_val,
        data_yaml=data_yaml,
        processed=Path(args.processed_dir),
        device=device,
        split=args.split,
    )
    summaries = seed_summary(rows)

    write_csv(week5 / "ablation_results.csv", rows)
    write_json(
        week5 / "ablation_results.json",
        {
            "split": args.split,
            "test_used_for_selection": False,
            "selection_rule": "best_val_mAP50 per seed; compare mean val mAP50 across seeds",
            "rows": rows,
            "note": "Unfinished runs marked Pending. Do not fabricate metrics.",
        },
    )
    write_csv(week5 / "seed_summary.csv", summaries)
    write_json(week5 / "seed_summary.json", {"summaries": summaries})

    make_figures(summaries, rows, figdir, week5)

    if not args.skip_predict_examples:
        qualitative_examples(
            week5, figdir, Path(args.processed_dir), device, args.n_examples, args.conf
        )

    n_done = sum(1 for r in rows if not pending(r["map50"]))
    print(f"Wrote {week5 / 'ablation_results.csv'} ({n_done}/{len(rows)} completed seed-runs)")
    print(f"Wrote {week5 / 'seed_summary.csv'}")
    print(f"Figures → {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
