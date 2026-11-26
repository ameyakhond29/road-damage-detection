#!/usr/bin/env python3
"""Week 3 — Train reproducible YOLOv8n reference baseline.

- Default data: official RDD2022 (configs/rdd2022.yaml + data/splits/*.csv);
  the synthetic fixture requires an explicit ``--dataset fixture``
- Outputs: results/official_rdd2022/week3/yolov8n_baseline/ (fixture: results/fixture_runs/)
- Image size 640, seed 6845
- Mixed precision when CUDA is available
- Checkpoint selection from validation only (Ultralytics never trains on test)
- Test split is not evaluated during training/selection
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
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
from src.training.baseline import add_dataset_args, prepare_training_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8n Week 3 reference baseline.")
    p.add_argument(
        "--config",
        "--data",
        dest="config",
        type=str,
        default=str(default_configs_dir() / "rdd2022.yaml"),
        help="Dataset config (official). Ignored with --dataset fixture.",
    )
    add_dataset_args(p)
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Default 50 for T4/Colab/Kaggle budget; raise for final runs",
    )
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name", type=str, default="train")
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="Empty = auto (cuda → mps → cpu)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Short run (3 epochs, batch 4) for pipeline verification",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Only write a stub run card (no training)",
    )
    return p.parse_args()


def pick_device(requested: str) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def software_versions() -> dict:
    import torch

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import ultralytics

        versions["ultralytics"] = ultralytics.__version__
    except ImportError:
        versions["ultralytics"] = None
    return versions


def gpu_name(device: str) -> str:
    import torch

    if device.isdigit() and torch.cuda.is_available():
        return torch.cuda.get_device_name(int(device))
    if device == "0" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple MPS"
    return "cpu"


def write_run_card(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_curves_and_predictions(run_dir: Path, project: Path) -> None:
    """Copy Ultralytics plots into the Week 3 results folder if present."""
    mapping = {
        "results.png": project / "curves_results.png",
        "BoxPR_curve.png": project / "curves_precision_recall.png",
        "BoxP_curve.png": project / "curves_precision.png",
        "BoxR_curve.png": project / "curves_recall.png",
        "BoxF1_curve.png": project / "curves_f1.png",
        "confusion_matrix.png": project / "confusion_matrix.png",
        "confusion_matrix_normalized.png": project / "confusion_matrix_normalized.png",
    }
    for src_name, dest in mapping.items():
        src = run_dir / src_name
        if src.is_file():
            shutil.copy2(src, dest)

    # Sample validation prediction mosaics
    for name in ("val_batch0_pred.jpg", "val_batch1_pred.jpg", "val_batch0_labels.jpg"):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, project / name)


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.batch = min(args.batch, 4)

    set_seed(args.seed)
    try:
        data_info, project = prepare_training_run(
            dataset=args.dataset, config=args.config, data_root=args.data_root,
            output_dir=args.output_dir, default_rel="week3/yolov8n_baseline",
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset not ready: {exc}", file=sys.stderr)
        return 1
    data_yaml = Path(data_info["ultralytics_yaml"])

    import torch

    device = pick_device(args.device)
    amp = device not in {"cpu", "mps"}
    versions = software_versions()
    gname = gpu_name(device)

    run_card = {
        "model": args.model,
        "model_family": "YOLOv8n",
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
        "software": versions,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(args.smoke),
        "checkpoint_selection": "best validation fitness / mAP50 (never test)",
        "test_used_during_training": False,
        "status": "not_started",
        "metrics": "Not yet available.",
        "training_time_sec": "Not yet available.",
        "best_checkpoint": "Not yet available.",
    }
    write_run_card(project / "run_config.json", run_card)

    if args.skip_train:
        run_card["status"] = "skipped"
        write_run_card(project / "metrics.json", run_card)
        print("Training skipped (--skip-train). Config written.")
        return 0

    from ultralytics import YOLO

    print(f"Device={device} AMP={amp} GPU={gname}")
    print(f"Data={data_yaml} model={args.model} imgsz={args.imgsz} "
          f"epochs={args.epochs} batch={args.batch} seed={args.seed}")

    t0 = time.perf_counter()
    model = YOLO(args.model)
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
        # Explicitly do not pass test for selection; Ultralytics uses val.
    )
    elapsed = time.perf_counter() - t0

    run_dir = Path(results.save_dir)
    weights_best = run_dir / "weights" / "best.pt"
    weights_last = run_dir / "weights" / "last.pt"
    ckpt = weights_best if weights_best.is_file() else weights_last

    val_metrics: dict = {}
    # Prefer results_dict from training object
    if hasattr(results, "results_dict") and results.results_dict:
        val_metrics = {k: _to_jsonable(v) for k, v in results.results_dict.items()}

    # Explicit validation pass on val split only
    try:
        val_res = model.val(data=str(data_yaml.resolve()), split="val")
        if hasattr(val_res, "results_dict") and val_res.results_dict:
            val_metrics = {k: _to_jsonable(v) for k, v in val_res.results_dict.items()}
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: extra val pass failed: {exc}", file=sys.stderr)

    export_curves_and_predictions(run_dir, project)

    # Copy results.csv if present
    src_csv = run_dir / "results.csv"
    if src_csv.is_file():
        shutil.copy2(src_csv, project / "results.csv")

    run_card.update(
        {
            "status": "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "training_time_sec": round(elapsed, 2),
            "run_dir": str(run_dir),
            "best_checkpoint": str(ckpt) if ckpt.is_file() else None,
            "metrics": val_metrics if val_metrics else "Not yet available.",
        }
    )
    write_run_card(project / "run_config.json", run_card)
    write_run_card(project / "metrics.json", run_card)

    print("\n=== YOLOv8n baseline complete ===")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {run_card['best_checkpoint']}")
    print(f"Training time (s): {run_card['training_time_sec']}")
    if isinstance(val_metrics, dict) and val_metrics:
        for k in sorted(val_metrics):
            if "mAP" in k or "precision" in k.lower() or "recall" in k.lower():
                print(f"  {k}: {val_metrics[k]}")
    print(f"Metrics JSON: {project / 'metrics.json'}")
    return 0 if ckpt.is_file() else 1


def _to_jsonable(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
