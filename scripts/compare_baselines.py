#!/usr/bin/env python3
"""Week 4 — Fair comparison of YOLOv8n vs YOLO11n from saved artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CLASS_NAMES, DEFAULT_SEED
from src.seed import set_seed
from src.training.baseline import (
    count_parameters,
    gpu_name,
    pick_device,
    profile_latency,
    to_float,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare YOLOv8n and YOLO11n baselines.")
    p.add_argument(
        "--v8-metrics",
        type=str,
        default=str(ROOT / "results" / "week3" / "yolov8n_baseline" / "metrics.json"),
    )
    p.add_argument(
        "--v11-metrics",
        type=str,
        default=str(ROOT / "results" / "week4" / "yolo11n_baseline" / "metrics.json"),
    )
    p.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "processed" / "RDD2022_yolo" / "rdd2022_fixture.yaml"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "week4"),
    )
    p.add_argument(
        "--figures-dir",
        type=str,
        default=str(ROOT / "figures" / "week4"),
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", type=str, default="")
    p.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip live latency profiling (use values from metrics JSON if present)",
    )
    return p.parse_args()


def load_metrics(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def extract_row(name: str, card: dict, *, latency_override: dict | None = None) -> dict:
    m = card.get("metrics") if isinstance(card.get("metrics"), dict) else {}
    p = to_float(m.get("metrics/precision(B)", "Pending"))
    r = to_float(m.get("metrics/recall(B)", "Pending"))
    f1 = "Pending"
    if isinstance(p, float) and isinstance(r, float) and (p + r) > 0:
        f1 = round(2 * p * r / (p + r), 6)

    lat = latency_override or card.get("latency")
    lat_ms = lat.get("latency_ms_mean") if isinstance(lat, dict) else "Pending"

    ckpt = card.get("best_checkpoint")
    ckpt_size = card.get("checkpoint_size_mb", "Pending")
    if ckpt_size == "Pending" and isinstance(ckpt, str) and Path(ckpt).is_file():
        ckpt_size = round(Path(ckpt).stat().st_size / (1024**2), 3)

    return {
        "model": name,
        "checkpoint": ckpt,
        "seed": card.get("seed"),
        "imgsz": card.get("imgsz"),
        "epochs": card.get("epochs"),
        "batch": card.get("batch"),
        "smoke": card.get("smoke"),
        "mAP50": m.get("metrics/mAP50(B)", "Pending"),
        "mAP50_95": m.get("metrics/mAP50-95(B)", "Pending"),
        "precision": p,
        "recall": r,
        "f1": f1,
        "n_parameters": card.get("n_parameters", "Pending"),
        "gflops": card.get("gflops", "Pending"),
        "latency_ms": lat_ms,
        "throughput_img_s": lat.get("throughput_img_s") if isinstance(lat, dict) else "Pending",
        "checkpoint_size_mb": ckpt_size,
        "training_time_sec": card.get("training_time_sec", "Pending"),
        "peak_gpu_memory_mb": card.get("peak_gpu_memory_mb", "Pending"),
        "gpu_name": card.get("gpu_name", "Pending"),
        "split_id": "grouped_70_15_15_seed6845",
        "selection_split": "val",
        "test_used_for_selection": False,
        "source_metrics_file": None,
    }


def maybe_profile(card: dict, imgsz: int, device: str) -> dict | None:
    ckpt = card.get("best_checkpoint")
    if not isinstance(ckpt, str) or not Path(ckpt).is_file():
        return None
    from ultralytics import YOLO

    model = YOLO(ckpt)
    n_params = count_parameters(model)
    if n_params is not None:
        card["n_parameters"] = n_params
    # GFLOPs via ultralytics info when possible
    try:
        info = model.info(detailed=False, verbose=False)
        # info often returns (layers, params, gradients, gflops) depending on version
        if isinstance(info, (tuple, list)) and len(info) >= 4:
            card["gflops"] = to_float(info[3])
    except Exception:  # noqa: BLE001
        pass
    return profile_latency(model, imgsz=imgsz, device=device)


def plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = ["mAP50", "mAP50_95", "precision", "recall", "f1"]
    models = df["model"].tolist()
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(models):
        vals = []
        for m in metrics:
            v = df.loc[df["model"] == model, m].iloc[0]
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                vals.append(0.0)
        ax.bar(x + (i - 0.5) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("YOLOv8n vs YOLO11n (validation metrics from saved runs)")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def conclude(df: pd.DataFrame) -> str:
    """Return a cautious textual conclusion from actual mAP50 values."""
    try:
        v8 = float(df.loc[df["model"] == "YOLOv8n", "mAP50"].iloc[0])
        v11 = float(df.loc[df["model"] == "YOLO11n", "mAP50"].iloc[0])
    except (TypeError, ValueError, IndexError):
        return (
            "Insufficient numeric mAP50 values to declare a winner. "
            "Do not claim YOLO11n is better."
        )
    if abs(v11 - v8) < 1e-6:
        return "mAP@0.5 is essentially tied under this run; no clear winner."
    if v11 > v8:
        return (
            f"Under this run, YOLO11n mAP@0.5 ({v11:.6g}) is higher than "
            f"YOLOv8n ({v8:.6g}). Treat as preliminary if smoke/fixture."
        )
    return (
        f"Under this run, YOLOv8n mAP@0.5 ({v8:.6g}) is higher than "
        f"YOLO11n ({v11:.6g}). Do not claim YOLO11n is better."
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        v8 = load_metrics(Path(args.v8_metrics))
        v11 = load_metrics(Path(args.v11_metrics))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Train both baselines first (Week 3 YOLOv8n + Week 4 YOLO11n).",
            file=sys.stderr,
        )
        return 1

    device = pick_device(args.device)
    print(f"Compare device={device} ({gpu_name(device)})")

    lat8 = lat11 = None
    if not args.skip_latency:
        try:
            lat8 = maybe_profile(v8, args.imgsz, device)
        except Exception as exc:  # noqa: BLE001
            print(f"YOLOv8n latency/profile warning: {exc}", file=sys.stderr)
        try:
            lat11 = maybe_profile(v11, args.imgsz, device)
        except Exception as exc:  # noqa: BLE001
            print(f"YOLO11n latency/profile warning: {exc}", file=sys.stderr)

    row8 = extract_row("YOLOv8n", v8, latency_override=lat8)
    row11 = extract_row("YOLO11n", v11, latency_override=lat11)
    row8["source_metrics_file"] = str(Path(args.v8_metrics).resolve())
    row11["source_metrics_file"] = str(Path(args.v11_metrics).resolve())
    # Refresh params from profile side effects
    row8["n_parameters"] = v8.get("n_parameters", row8["n_parameters"])
    row11["n_parameters"] = v11.get("n_parameters", row11["n_parameters"])
    row8["gflops"] = v8.get("gflops", "Pending")
    row11["gflops"] = v11.get("gflops", "Pending")

    df = pd.DataFrame([row8, row11])
    csv_path = out_dir / "baseline_comparison.csv"
    json_path = out_dir / "baseline_comparison.json"
    fig_path = fig_dir / "baseline_comparison.png"

    conclusion = conclude(df)
    payload = {
        "models": df.to_dict(orient="records"),
        "conclusion": conclusion,
        "notes": [
            "Metrics loaded from saved run JSON files only.",
            "Validation metrics are used for comparison of selected checkpoints.",
            "Test-set numbers belong in evaluate_week4.py outputs, not selection.",
            "Do not claim superiority unless conclusion and numbers agree.",
        ],
        "class_names": CLASS_NAMES,
    }
    df.to_csv(csv_path, index=False)
    write_json(json_path, payload)
    plot_comparison(df, fig_path)

    print("\n=== Baseline comparison ===")
    print(df[["model", "mAP50", "mAP50_95", "precision", "recall", "f1", "n_parameters", "latency_ms"]].to_string(index=False))
    print(f"\nConclusion: {conclusion}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
