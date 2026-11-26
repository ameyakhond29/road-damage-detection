#!/usr/bin/env python3
"""Week 4 — Train YOLO11n strong baseline (same protocol as YOLOv8n).

Default data: official RDD2022 (configs/rdd2022.yaml + data/splits/*.csv); the
synthetic fixture requires ``--dataset fixture``. Outputs go to
results/official_rdd2022/week4/yolo11n_baseline/ (fixture: results/fixture_runs/).
Seed 6845, imgsz 640, val-only selection.
Test split is never used for training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import DEFAULT_SEED
from src.paths import default_configs_dir
from src.seed import set_seed
from src.training.baseline import (
    add_dataset_args,
    count_parameters,
    prepare_training_run,
    export_ultralytics_plots,
    gpu_name,
    peak_gpu_memory_mb,
    pick_device,
    profile_latency,
    software_versions,
    to_float,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO11n Week 4 strong baseline.")
    p.add_argument("--config", "--data", dest="config", type=str,
                   default=str(default_configs_dir() / "rdd2022.yaml"),
                   help="Dataset config (official). Ignored with --dataset fixture.")
    add_dataset_args(p)
    p.add_argument("--model", type=str, default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", type=str, default="train")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.batch = min(args.batch, 4)

    set_seed(args.seed)
    try:
        data_info, project = prepare_training_run(
            dataset=args.dataset, config=args.config, data_root=args.data_root,
            output_dir=args.output_dir, default_rel="week4/yolo11n_baseline",
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset not ready: {exc}", file=sys.stderr)
        return 1
    data_yaml = Path(data_info["ultralytics_yaml"])

    import torch

    device = pick_device(args.device)
    amp = device not in {"cpu", "mps"}
    gname = gpu_name(device)

    run_card = {
        "model": args.model,
        "model_family": "YOLO11n",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "epochs": args.epochs,
        "lr0": args.lr0,
        "optimizer": "auto (Ultralytics)",
        "patience": args.patience,
        "seed": args.seed,
        "amp": amp,
        "device": device,
        "gpu_name": gname,
        "data": str(data_yaml.resolve()),
        "dataset": data_info["dataset"],
        "dataset_source": data_info["dataset_source"],
        "dataset_version": data_info.get("dataset_version"),
        "dataset_hash": data_info.get("dataset_hash"),
        "split_hashes": data_info.get("split_hashes"),
        "command": " ".join(sys.argv),
        "software": software_versions(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(args.smoke),
        "checkpoint_selection": "best validation fitness / mAP50 (never test)",
        "test_used_during_training": False,
        "n_parameters": "Pending",
        "peak_gpu_memory_mb": "Pending",
        "latency": "Pending",
        "status": "not_started",
        "metrics": "Pending",
        "training_time_sec": "Pending",
        "best_checkpoint": "Pending",
    }
    write_json(project / "run_config.json", run_card)

    if args.skip_train:
        run_card["status"] = "skipped"
        write_json(project / "metrics.json", run_card)
        print("Training skipped (--skip-train).")
        return 0

    from ultralytics import YOLO

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"Device={device} AMP={amp} GPU={gname}")
    print(
        f"Data={data_yaml} model={args.model} imgsz={args.imgsz} "
        f"epochs={args.epochs} batch={args.batch} seed={args.seed}"
    )

    t0 = time.perf_counter()
    model = YOLO(args.model)
    n_params = count_parameters(model)
    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        amp=amp,
        seed=args.seed,
        project=str(project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        patience=args.patience,
        lr0=args.lr0,
        workers=args.workers,
        plots=True,
        save=True,
        val=True,
    )
    elapsed = time.perf_counter() - t0

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    ckpt = best if best.is_file() else last

    val_metrics: dict = {}
    if hasattr(results, "results_dict") and results.results_dict:
        val_metrics = {k: to_float(v) for k, v in results.results_dict.items()}
    try:
        val_res = model.val(data=str(data_yaml.resolve()), split="val")
        if hasattr(val_res, "results_dict") and val_res.results_dict:
            val_metrics = {k: to_float(v) for k, v in val_res.results_dict.items()}
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: val pass failed: {exc}", file=sys.stderr)

    export_ultralytics_plots(run_dir, project)

    latency = "Pending"
    try:
        latency = profile_latency(model, imgsz=args.imgsz, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: latency profiling failed: {exc}", file=sys.stderr)

    peak_mem = peak_gpu_memory_mb(device)

    run_card.update(
        {
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "training_time_sec": round(elapsed, 2),
            "run_dir": str(run_dir),
            "best_checkpoint": str(ckpt) if ckpt.is_file() else None,
            "metrics": val_metrics if val_metrics else "Pending",
            "n_parameters": n_params if n_params is not None else "Pending",
            "peak_gpu_memory_mb": peak_mem if peak_mem is not None else "Pending",
            "latency": latency,
            "checkpoint_size_mb": round(ckpt.stat().st_size / (1024**2), 3)
            if ckpt.is_file()
            else "Pending",
        }
    )
    write_json(project / "run_config.json", run_card)
    write_json(project / "metrics.json", run_card)

    print("\n=== YOLO11n baseline complete ===")
    print(f"Checkpoint: {run_card['best_checkpoint']}")
    print(f"Params: {run_card['n_parameters']} | time_s={run_card['training_time_sec']}")
    if isinstance(val_metrics, dict):
        for k in sorted(val_metrics):
            if "mAP" in k or "precision" in k.lower() or "recall" in k.lower():
                print(f"  {k}: {val_metrics[k]}")
    print(f"Metrics: {project / 'metrics.json'}")
    return 0 if ckpt.is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
