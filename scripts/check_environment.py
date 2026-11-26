#!/usr/bin/env python3
"""Report Python / PyTorch / Ultralytics / CUDA / package versions for Week 7."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "week7"
OUT.mkdir(parents=True, exist_ok=True)

from src.paths import default_processed_dir, default_rdd2022_config  # noqa: E402


def _ver(mod_name: str):
    try:
        mod = __import__(mod_name)
        return getattr(mod, "__version__", str(mod))
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({exc})"


def main() -> int:
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "rdd2022_config": str(default_rdd2022_config()),
        "official_processed_present": (default_processed_dir() / "manifest.csv").is_file(),
    }

    info["numpy_version"] = _ver("numpy")
    info["pandas_version"] = _ver("pandas")
    try:
        import cv2

        info["opencv_version"] = cv2.__version__
    except Exception as exc:  # noqa: BLE001
        info["opencv_version"] = f"MISSING ({exc})"

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            info["gpu_name"] = "Apple MPS"
            info["cuda_available"] = False
        else:
            info["gpu_name"] = "cpu"
    except Exception as exc:  # noqa: BLE001
        info["torch_version"] = f"MISSING ({exc})"
        info["cuda_available"] = False
        info["cuda_version"] = None
        info["gpu_name"] = "unknown"

    info["torchvision_version"] = _ver("torchvision")
    info["ultralytics_version"] = _ver("ultralytics")

    # Print human-readable block
    keys = [
        ("Python version", "python_version"),
        ("PyTorch version", "torch_version"),
        ("Torchvision version", "torchvision_version"),
        ("Ultralytics version", "ultralytics_version"),
        ("CUDA availability", "cuda_available"),
        ("CUDA version", "cuda_version"),
        ("GPU name", "gpu_name"),
        ("OpenCV version", "opencv_version"),
        ("NumPy version", "numpy_version"),
        ("Pandas version", "pandas_version"),
    ]
    print("=== Environment check ===")
    for label, key in keys:
        print(f"{label}: {info.get(key)}")

    out_json = OUT / "environment_check.json"
    out_json.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    # Soft fail if critical deps missing
    critical = ["torch_version", "ultralytics_version", "numpy_version", "pandas_version"]
    bad = [k for k in critical if str(info.get(k, "")).startswith("MISSING")]
    if bad:
        print(f"FAIL: missing packages: {bad}", file=sys.stderr)
        return 1
    print("PASS: core packages importable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
