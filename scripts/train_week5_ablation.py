#!/usr/bin/env python3
"""Week 5 — Train one ablation experiment (val-only checkpoint selection).

Never uses the test split for selection or hyperparameter tuning.
Default data: official RDD2022 (experiment YAML ``data: configs/rdd2022.yaml``); the
synthetic fixture requires ``--dataset fixture``. Saves full config + command under
results/official_rdd2022/week5/<experiment>/seed_<k>/ (fixture: results/fixture_runs/).
"""

from __future__ import annotations

import argparse
import hashlib
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.resolve import MODEL_REGISTRY, resolve_model_spec
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

EXPERIMENT_CONFIGS = {
    "yolo11n_640": "configs/week5_yolo11n_640.yaml",
    "yolo11n_p2_640": "configs/week5_yolo11n_p2_640.yaml",
    "yolo11n_attention_640": "configs/week5_yolo11n_attention_640.yaml",
    "yolo11n_800": "configs/week5_yolo11n_800.yaml",
    "combined_800": "configs/week5_combined_800.yaml",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Week 5 ablation experiment.")
    p.add_argument(
        "--experiment",
        required=True,
        choices=sorted(EXPERIMENT_CONFIGS.keys()),
        help="Ablation experiment name",
    )
    p.add_argument("--config", type=str, default="", help="Override experiment YAML")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--smoke", action="store_true", help="Short run (1–3 epochs, small batch)")
    p.add_argument("--skip-train", action="store_true")
    add_dataset_args(p)
    return p.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def file_sha16(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def gflops_of(model, imgsz: int) -> float | None:
    try:
        # verbose=True returns (layers, params, grads, gflops); verbose=False may return None
        info = model.info(detailed=False, verbose=True)
        if isinstance(info, (tuple, list)) and len(info) >= 4:
            return float(info[3])
    except Exception:  # noqa: BLE001
        pass
    try:
        from ultralytics.utils.torch_utils import get_flops

        return float(get_flops(model.model, imgsz=imgsz))
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    args = parse_args()
    cfg_rel = args.config or EXPERIMENT_CONFIGS[args.experiment]
    cfg_path = Path(cfg_rel)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if not cfg_path.is_file():
        print(f"Missing config: {cfg_path}", file=sys.stderr)
        return 1

    cfg = load_yaml(cfg_path)
    spec = resolve_model_spec(experiment=args.experiment)

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    imgsz = int(args.imgsz if args.imgsz is not None else cfg.get("imgsz", spec["imgsz_default"]))
    batch = int(args.batch if args.batch is not None else cfg.get("batch", 16))
    epochs = int(args.epochs if args.epochs is not None else cfg.get("epochs", 100))
    patience = int(cfg.get("patience", 30))
    lr0 = float(cfg.get("lr0", 0.01))
    workers = int(args.workers if args.workers is not None else cfg.get("workers", 4))
    data_rel = cfg.get("data", "configs/rdd2022.yaml")
    data_cfg = Path(data_rel) if Path(data_rel).is_absolute() else ROOT / data_rel

    if args.smoke:
        epochs = min(epochs, 2)
        # Combined P2@800 needs a small batch on laptop GPUs / MPS
        if args.experiment == "combined_800" or imgsz >= 800:
            batch = min(batch, 2)
        else:
            batch = min(batch, 4)
        patience = min(patience, 2)

    try:
        data_info, out_dir = prepare_training_run(
            dataset=args.dataset, config=str(data_cfg), data_root=args.data_root,
            output_dir=args.output_dir, default_rel=f"week5/{args.experiment}/seed_{seed}",
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Dataset not ready: {exc}", file=sys.stderr)
        return 1
    data_yaml = Path(data_info["ultralytics_yaml"])

    set_seed(seed)
    import torch

    device = pick_device(args.device)
    # AMP only when CUDA is available (protocol)
    amp = bool(cfg.get("amp", True)) and torch.cuda.is_available() and device not in {"cpu", "mps"}
    gname = gpu_name(device)

    argv_cmd = " ".join([shlex.quote(a) for a in sys.argv])
    command = f"python scripts/train_week5_ablation.py {argv_cmd.split('train_week5_ablation.py', 1)[-1].strip()}"
    if "train_week5_ablation.py" not in argv_cmd:
        command = f"python {' '.join(shlex.quote(a) for a in sys.argv)}"

    split_hashes = data_info.get("split_hashes")

    run_card = {
        "experiment": args.experiment,
        "model_variant": spec["label"],
        "architecture": spec["architecture"],
        "factors": spec["factors"],
        "model": spec["model"],
        "pretrained_weights": spec.get("pretrained", cfg.get("pretrained_weights")),
        "input_resolution": imgsz,
        "batch_size": batch,
        "epochs": epochs,
        "learning_rate": lr0,
        "optimizer": cfg.get("optimizer", "auto"),
        "patience": patience,
        "seed": seed,
        "gpu": gname,
        "device": device,
        "amp": amp,
        "data": str(data_yaml.resolve()),
        "config_path": str(cfg_path.resolve()),
        "config_sha16": file_sha16(cfg_path),
        "dataset": data_info["dataset"],
        "dataset_source": data_info["dataset_source"],
        "dataset_version": data_info.get("dataset_version"),
        "dataset_hash": data_info.get("dataset_hash"),
        "dataset_config": str(data_cfg),
        "split_id": data_info.get("split_id", "grouped_70_15_15_seed6845"),
        "split_hashes": split_hashes,
        "class_mapping": cfg.get("class_mapping", {"D00": 0, "D10": 1, "D20": 2, "D40": 3}),
        "checkpoint_selection": "best_val_mAP50",
        "test_used_for_selection": False,
        "test_inspected": False,
        "command": command,
        "full_argv": list(sys.argv),
        "full_config": cfg,
        "software": software_versions(),
        "smoke": bool(args.smoke),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "not_started",
        "training_duration_sec": "Pending",
        "parameter_count": "Pending",
        "gflops": "Pending",
        "peak_gpu_memory_mb": "Pending",
        "inference_latency": "Pending",
        "metrics": "Pending",
        "best_checkpoint": "Pending",
    }
    write_json(out_dir / "run_config.json", run_card)
    write_json(out_dir / "command.json", {"command": command, "argv": list(sys.argv)})

    if args.skip_train:
        run_card["status"] = "skipped"
        write_json(out_dir / "metrics.json", run_card)
        print("Training skipped (--skip-train).")
        return 0

    from ultralytics import YOLO

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(
        f"[Week5] exp={args.experiment} seed={seed} imgsz={imgsz} "
        f"batch={batch} epochs={epochs} device={device} amp={amp}"
    )
    print(f"  model={spec['model']}")
    print(f"  data={data_yaml} split_hashes={split_hashes}")

    t0 = time.perf_counter()
    model = YOLO(spec["model"])
    # Transfer COCO-pretrained weights when starting from custom YAML
    pretrained = spec.get("pretrained") or cfg.get("pretrained_weights")
    if pretrained and str(spec["model"]).endswith((".yaml", ".yml")):
        try:
            model = YOLO(spec["model"]).load(pretrained)
            print(f"  loaded matching weights from {pretrained}")
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: weight transfer failed ({exc}); training from YAML init")

    n_params = count_parameters(model)
    gflops = gflops_of(model, imgsz)

    results = model.train(
        data=str(data_yaml.resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        amp=amp,
        seed=seed,
        project=str(out_dir),
        name="train",
        exist_ok=True,
        pretrained=bool(cfg.get("pretrained", True)),
        patience=patience,
        lr0=lr0,
        workers=workers,
        plots=True,
        save=True,
        val=True,
        hsv_h=float(cfg.get("hsv_h", 0.015)),
        hsv_s=float(cfg.get("hsv_s", 0.7)),
        hsv_v=float(cfg.get("hsv_v", 0.4)),
        degrees=float(cfg.get("degrees", 0.0)),
        translate=float(cfg.get("translate", 0.1)),
        scale=float(cfg.get("scale", 0.5)),
        shear=float(cfg.get("shear", 0.0)),
        perspective=float(cfg.get("perspective", 0.0)),
        flipud=float(cfg.get("flipud", 0.0)),
        fliplr=float(cfg.get("fliplr", 0.5)),
        mosaic=float(cfg.get("mosaic", 1.0)),
        mixup=float(cfg.get("mixup", 0.0)),
        copy_paste=float(cfg.get("copy_paste", 0.0)),
    )
    elapsed = time.perf_counter() - t0

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    ckpt = best if best.is_file() else last

    # Validation-only metrics (never test)
    val_metrics: dict = {}
    if hasattr(results, "results_dict") and results.results_dict:
        val_metrics = {k: to_float(v) for k, v in results.results_dict.items()}
    try:
        val_res = model.val(data=str(data_yaml.resolve()), split="val", imgsz=imgsz, device=device)
        if hasattr(val_res, "results_dict") and val_res.results_dict:
            val_metrics = {k: to_float(v) for k, v in val_res.results_dict.items()}
        # Attach per-class AP50 when available
        try:
            box = val_res.box
            if hasattr(box, "ap50") and box.ap50 is not None:
                names = ["D00", "D10", "D20", "D40"]
                per = {}
                for i, name in enumerate(names):
                    if i < len(box.ap50):
                        per[name] = to_float(box.ap50[i])
                val_metrics["per_class_ap50"] = per
            if hasattr(box, "ap") and box.ap is not None:
                names = ["D00", "D10", "D20", "D40"]
                per = {}
                for i, name in enumerate(names):
                    if i < len(box.ap):
                        per[name] = to_float(box.ap[i])
                val_metrics["per_class_ap"] = per
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: val pass failed: {exc}", file=sys.stderr)

    export_ultralytics_plots(run_dir, out_dir)

    latency: dict | str = "Pending"
    try:
        latency = profile_latency(model, imgsz=imgsz, device=device, warmup=5, iters=20 if args.smoke else 50)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: latency failed: {exc}", file=sys.stderr)

    peak_mem = peak_gpu_memory_mb(device)
    if n_params is None:
        n_params = count_parameters(model)
    if gflops is None:
        gflops = gflops_of(model, imgsz)

    # Derive P/R/F1 from Ultralytics keys when present
    precision = val_metrics.get("metrics/precision(B)", "Pending")
    recall = val_metrics.get("metrics/recall(B)", "Pending")
    f1: float | str = "Pending"
    if isinstance(precision, (int, float)) and isinstance(recall, (int, float)):
        if (precision + recall) > 0:
            f1 = round(2 * precision * recall / (precision + recall), 6)

    run_card.update(
        {
            "status": "completed_smoke" if args.smoke else "completed",
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "training_duration_sec": round(elapsed, 2),
            "run_dir": str(run_dir),
            "best_checkpoint": str(ckpt) if ckpt.is_file() else None,
            "parameter_count": n_params if n_params is not None else "Pending",
            "gflops": gflops if gflops is not None else "Pending",
            "peak_gpu_memory_mb": peak_mem if peak_mem is not None else "Pending",
            "inference_latency": latency,
            "metrics": val_metrics if val_metrics else "Pending",
            "map50": val_metrics.get("metrics/mAP50(B)", "Pending"),
            "map50_95": val_metrics.get("metrics/mAP50-95(B)", "Pending"),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "per_class_ap": val_metrics.get("per_class_ap50", "Pending"),
            "small_object_ap": "Pending",  # filled by evaluate_week5_ablation.py
            "selection_metric": "map50_val",
            "test_used_for_selection": False,
        }
    )
    write_json(out_dir / "run_config.json", run_card)
    write_json(out_dir / "metrics.json", run_card)
    print(f"Saved metrics → {out_dir / 'metrics.json'}")
    print(f"  map50={run_card['map50']} map50-95={run_card['map50_95']} params={n_params} gflops={gflops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
