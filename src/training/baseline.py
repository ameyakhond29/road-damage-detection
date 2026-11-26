"""Shared helpers for YOLOv8n / YOLO11n baseline training and profiling."""

from __future__ import annotations

import json
import platform
import shutil
import time
from pathlib import Path


def pick_device(requested: str = "") -> str:
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
        "cuda_available": bool(torch.cuda.is_available()),
    }
    try:
        import ultralytics

        versions["ultralytics"] = ultralytics.__version__
    except ImportError:
        versions["ultralytics"] = None
    return versions


def gpu_name(device: str) -> str:
    import torch

    if (device.isdigit() or device == "0") and torch.cuda.is_available():
        idx = int(device) if device.isdigit() else 0
        return torch.cuda.get_device_name(idx)
    if device == "mps":
        return "Apple MPS"
    return "cpu"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_ultralytics_plots(run_dir: Path, project: Path) -> None:
    mapping = {
        "results.png": "curves_results.png",
        "BoxPR_curve.png": "curves_precision_recall.png",
        "BoxP_curve.png": "curves_precision.png",
        "BoxR_curve.png": "curves_recall.png",
        "BoxF1_curve.png": "curves_f1.png",
        "confusion_matrix.png": "confusion_matrix.png",
        "confusion_matrix_normalized.png": "confusion_matrix_normalized.png",
    }
    project.mkdir(parents=True, exist_ok=True)
    for src_name, dest_name in mapping.items():
        src = run_dir / src_name
        if src.is_file():
            shutil.copy2(src, project / dest_name)
    for name in (
        "val_batch0_pred.jpg",
        "val_batch1_pred.jpg",
        "val_batch0_labels.jpg",
        "results.csv",
    ):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, project / name)


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def count_parameters(model) -> int | None:
    try:
        return int(sum(p.numel() for p in model.model.parameters()))
    except Exception:  # noqa: BLE001
        try:
            info = model.info(detailed=False, verbose=False)
            if isinstance(info, (tuple, list)) and info:
                return int(info[0])
        except Exception:  # noqa: BLE001
            return None
    return None


def profile_latency(
    model,
    *,
    imgsz: int,
    device: str,
    warmup: int = 10,
    iters: int = 50,
) -> dict:
    """Batch-1 inference latency (ms) and throughput (img/s)."""
    import numpy as np
    import torch

    # Synthetic RGB image
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    # Warmup
    for _ in range(warmup):
        model.predict(source=img, imgsz=imgsz, device=device, verbose=False)

    if device.isdigit() or device == "0":
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(source=img, imgsz=imgsz, device=device, verbose=False)
        if device.isdigit() or device == "0":
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    mean_s = float(np.mean(times)) if times else float("nan")
    return {
        "latency_ms_mean": round(mean_s * 1000.0, 3),
        "latency_ms_std": round(float(np.std(times)) * 1000.0, 3) if times else None,
        "throughput_img_s": round(1.0 / mean_s, 3) if mean_s > 0 else None,
        "n_iters": iters,
        "warmup": warmup,
    }


def peak_gpu_memory_mb(device: str) -> float | None:
    import torch

    if not (device.isdigit() or device == "0"):
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**2), 2)


def add_dataset_args(p, *, config_help: str = "Dataset config (default configs/rdd2022.yaml)") -> None:
    """Common dataset/output CLI flags for training scripts."""
    from src.paths import DATASET_CHOICES

    p.add_argument("--dataset", choices=DATASET_CHOICES, default="official",
                   help="official RDD2022 (default) or the synthetic fixture (debugging only)")
    p.add_argument("--data-root", default=None,
                   help="Processed root with images/ and labels/ (default data/processed)")
    p.add_argument("--output-dir", default=None,
                   help="Default results/official_rdd2022/<week>/<run> (fixture: results/fixture_runs/...)")


def prepare_training_run(
    *, dataset: str, config: str | None, data_root: str | None, output_dir: str | None, default_rel: str
) -> tuple[dict, Path]:
    """Resolve the Ultralytics data YAML + provenance and a safe output directory.

    Official runs write under ``results/official_rdd2022/``; fixture runs under
    ``results/fixture_runs/``. Legacy ``results/week*`` folders are never written.
    """
    from src.dataset.official import resolve_training_data
    from src.paths import is_legacy_results_path, official_results_dir, project_root

    info = resolve_training_data(dataset, config=config, data_root=data_root)
    base = official_results_dir() if dataset == "official" else project_root() / "results" / "fixture_runs"
    out = Path(output_dir).expanduser() if output_dir else base / default_rel
    out = out if out.is_absolute() else (Path.cwd() / out)
    if is_legacy_results_path(out):
        raise SystemExit(f"Refusing to write into legacy fixture results: {out}")
    official_root = official_results_dir().resolve()
    inside_official = official_root == out.resolve() or official_root in out.resolve().parents
    if dataset == "fixture" and inside_official:
        raise SystemExit("Fixture runs must not write into results/official_rdd2022/")
    out.mkdir(parents=True, exist_ok=True)
    return info, out


def split_file_hash(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]
