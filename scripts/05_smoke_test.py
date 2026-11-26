#!/usr/bin/env python3
"""LEGACY (Week 1, synthetic fixture) — Small YOLOv8n smoke test.

Official-data smoke test: scripts/smoke_test_official_rdd2022.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import DEFAULT_SEED
from src.paths import (
    default_figures_dir,
    legacy_fixture_data_yaml,
    legacy_fixture_processed_dir,
    default_results_dir,
    project_root,
)
from src.seed import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a tiny YOLOv8n smoke test.")
    p.add_argument(
        "--data",
        type=str,
        default=str(legacy_fixture_data_yaml()),
        help="Path to YOLO dataset YAML (legacy fixture)",
    )
    p.add_argument(
        "--smoke-config",
        type=str,
        default=str(legacy_fixture_processed_dir() / "rdd2022_smoke.yaml"),
        help="Where to write the reduced smoke-test YAML",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=320, choices=[320, 640])
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-train", type=int, default=300)
    p.add_argument("--max-val", type=int, default=100)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--model", type=str, default="yolov8n.pt")
    p.add_argument(
        "--project",
        type=str,
        default=str(default_results_dir() / "smoke_test"),
    )
    p.add_argument("--name", type=str, default="yolov8n_smoke")
    p.add_argument(
        "--figures-dir",
        type=str,
        default=str(default_figures_dir()),
    )
    p.add_argument(
        "--device",
        type=str,
        default="",
        help="Torch device string (empty = auto)",
    )
    return p.parse_args()


def _resolve_dataset_path(cfg_path: Path, path_value: str) -> Path:
    """Resolve Ultralytics ``path`` (relative to the YAML parent by convention)."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    # Ultralytics resolves relative paths against the YAML directory
    cand = (cfg_path.parent / p).resolve()
    if cand.exists():
        return cand
    # Fallback: project root (older configs)
    cand2 = (project_root() / p).resolve()
    return cand2 if cand2.exists() else cand


def _read_list(list_path: Path, limit: int, *, ds_root: Path | None = None) -> list[str]:
    lines = [
        ln.strip()
        for ln in list_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    lines = lines[:limit]
    if ds_root is None:
        return lines
    # Normalize to absolute paths (handles older relative list files).
    out: list[str] = []
    for ln in lines:
        p = Path(ln)
        if not p.is_absolute():
            p = (ds_root / p).resolve()
        out.append(p.as_posix())
    return out


def build_smoke_yaml(
    full_yaml: Path,
    smoke_yaml: Path,
    *,
    max_train: int,
    max_val: int,
) -> Path:
    import yaml

    cfg = yaml.safe_load(full_yaml.read_text(encoding="utf-8"))
    ds_root = _resolve_dataset_path(full_yaml, cfg["path"])

    train_list = ds_root / cfg["train"]
    val_list = ds_root / cfg["val"]
    if not train_list.is_file():
        raise FileNotFoundError(f"Train list missing: {train_list}")
    if not val_list.is_file():
        raise FileNotFoundError(f"Val list missing: {val_list}")

    smoke_dir = ds_root / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    train_lines = _read_list(train_list, max_train, ds_root=ds_root)
    val_lines = _read_list(val_list, max_val, ds_root=ds_root)
    # Safety net: if val split is empty (e.g. tiny fixture + aggressive phash
    # merging), carve a holdout from train so validation can still run.
    if not val_lines and len(train_lines) >= 20:
        holdout = max(5, min(max_val, len(train_lines) // 5))
        val_lines = train_lines[-holdout:]
        train_lines = train_lines[:-holdout]
        print(
            f"Warning: empty val split — carved {len(val_lines)} images from train "
            "for smoke-test validation only."
        )
    if len(train_lines) < 10:
        raise RuntimeError(
            f"Need at least ~10 train images for a smoke test; found {len(train_lines)}. "
            "Convert more data or lower --max-train only after conversion."
        )

    (smoke_dir / "train.txt").write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    (smoke_dir / "val.txt").write_text("\n".join(val_lines) + "\n", encoding="utf-8")

    # Absolute dataset root — Ultralytics mishandles `..` in relative paths.
    smoke_cfg = {
        "path": ds_root.as_posix(),
        "train": "smoke/train.txt",
        "val": "smoke/val.txt",
        "test": "smoke/val.txt",
        "nc": cfg.get("nc", 4),
        "names": cfg.get("names"),
        "class_ids": cfg.get("class_ids"),
    }
    smoke_yaml.parent.mkdir(parents=True, exist_ok=True)
    header = "# Auto-generated smoke-test subset — do not use for real experiments\n"
    smoke_yaml.write_text(
        header + yaml.safe_dump(smoke_cfg, sort_keys=False), encoding="utf-8"
    )
    print(
        f"Smoke subset: {len(train_lines)} train / {len(val_lines)} val images → {smoke_yaml}"
    )
    return smoke_yaml


def pick_device(requested: str) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        print(
            f"Dataset config not found: {data_yaml}\n"
            "Run scripts 01→04 first (or point --data at an existing YAML).",
            file=sys.stderr,
        )
        return 1

    try:
        smoke_yaml = build_smoke_yaml(
            data_yaml,
            Path(args.smoke_config),
            max_train=args.max_train,
            max_val=args.max_val,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    import torch
    from ultralytics import YOLO

    device = pick_device(args.device)
    amp = device not in {"cpu", "mps"}  # CUDA mixed precision when available
    print(f"Device: {device} | AMP: {amp} | torch {torch.__version__}")
    if device == "0" or (device.isdigit()):
        print(f"CUDA device: {torch.cuda.get_device_name(int(device))}")

    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.train(
        data=str(smoke_yaml.resolve()),
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
        verbose=True,
        plots=True,
        save=True,
        workers=2,
    )

    run_dir = Path(results.save_dir)
    weights = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not weights.is_file() and last.is_file():
        weights = last

    metrics: dict = {}
    if hasattr(results, "results_dict") and results.results_dict:
        metrics = dict(results.results_dict)
    elif hasattr(results, "box") and results.box is not None:
        # Fallback: pull common keys if present
        box = results.box
        for key in ("map50", "map", "mp", "mr"):
            if hasattr(box, key):
                metrics[key] = float(getattr(box, key))

    # Validation pass (explicit)
    val_metrics = model.val(data=str(smoke_yaml.resolve()), split="val")
    val_dict = {}
    if hasattr(val_metrics, "results_dict"):
        val_dict = dict(val_metrics.results_dict)

    # Predictions on a few val images
    import yaml

    cfg = yaml.safe_load(smoke_yaml.read_text(encoding="utf-8"))
    ds_root = _resolve_dataset_path(smoke_yaml, cfg["path"])
    val_file = Path(cfg["val"])
    if not val_file.is_absolute():
        val_file = ds_root / val_file
    val_lines = _read_list(val_file, 8, ds_root=ds_root)
    pred_sources = [p for p in val_lines if Path(p).is_file()]

    pred_dir = project / "predictions"
    if pred_sources:
        pred_results = model.predict(
            source=pred_sources,
            save=True,
            project=str(project),
            name="predictions",
            exist_ok=True,
            imgsz=args.imgsz,
            device=device,
        )
        # Copy a collage / first prediction to figures/
        figures_dir = Path(args.figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        out_fig = figures_dir / "smoke_test_predictions.png"
        # Ultralytics saves annotated images under pred_dir
        saved = list(Path(pred_results[0].save_dir).glob("*.jpg")) + list(
            Path(pred_results[0].save_dir).glob("*.png")
        )
        if saved:
            # Build a simple montage if matplotlib available
            try:
                _save_prediction_montage(saved[:8], out_fig)
            except Exception:
                shutil.copy2(saved[0], out_fig)
            print(f"Predictions figure: {out_fig}")
        else:
            print("Warning: no prediction images were saved.", file=sys.stderr)

    summary = {
        "device": device,
        "amp": amp,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "run_dir": str(run_dir),
        "checkpoint": str(weights) if weights.is_file() else None,
        "train_metrics": {k: _to_float(v) for k, v in metrics.items()},
        "val_metrics": {k: _to_float(v) for k, v in val_dict.items()},
        "smoke_yaml": str(smoke_yaml),
    }
    summary_path = project / "smoke_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== Smoke test summary ===")
    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {summary['checkpoint']}")
    print("Validation metrics:")
    for k, v in summary["val_metrics"].items():
        print(f"  {k}: {v}")
    print(f"Summary JSON: {summary_path}")

    # Verify checklist
    ok_checkpoint = bool(summary["checkpoint"])
    ok_metrics = bool(summary["val_metrics"])
    if not (ok_checkpoint and ok_metrics):
        print("Smoke test incomplete: missing checkpoint or metrics.", file=sys.stderr)
        return 1
    print("Smoke test PASSED (train + val + checkpoint + predictions).")
    return 0


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _save_prediction_montage(image_paths: list[Path], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(image_paths)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.axis("off")
        if i < n:
            ax.imshow(Image.open(image_paths[i]))
            ax.set_title(image_paths[i].name, fontsize=7)
    fig.suptitle("Smoke-test predictions (YOLOv8n)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
