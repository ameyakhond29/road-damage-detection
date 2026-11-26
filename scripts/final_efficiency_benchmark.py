#!/usr/bin/env python3
"""Week 6 — Efficiency benchmark under the frozen protocol (imgsz / batch from YAML).

Targets NVIDIA T4 when available; otherwise records the actual device (MPS/CPU)
and marks T4-specific fields honestly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.week6_utils import (
    load_protocol,
    model_registry,
    pending,
    resolve_ckpt,
    write_csv,
    write_json,
)
from src.training.baseline import count_parameters, gpu_name, peak_gpu_memory_mb, pick_device

WEEK6 = ROOT / "results" / "week6"
FIGDIR = ROOT / "figures" / "week6"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Week 6 efficiency benchmark.")
    p.add_argument("--protocol", type=str, default=str(ROOT / "configs" / "final_evaluation.yaml"))
    p.add_argument("--device", type=str, default="")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def gflops_of(model, imgsz: int) -> float | str:
    try:
        info = model.info(detailed=False, verbose=True)
        if isinstance(info, (tuple, list)) and len(info) >= 4:
            return float(info[3])
    except Exception:  # noqa: BLE001
        pass
    try:
        from ultralytics.utils.torch_utils import get_flops

        return float(get_flops(model.model, imgsz=imgsz))
    except Exception:  # noqa: BLE001
        return "Pending"


def latency_stats(model, imgsz: int, device: str, warmup: int, iters: int) -> dict:
    img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):
        model.predict(source=img, imgsz=imgsz, device=device, verbose=False)
    import torch

    times = []
    for _ in range(iters):
        if torch.cuda.is_available() and (device.isdigit() or device == "0"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(source=img, imgsz=imgsz, device=device, verbose=False)
        if torch.cuda.is_available() and (device.isdigit() or device == "0"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times) if times else np.array([float("nan")])
    mean = float(np.mean(arr))
    return {
        "latency_ms_mean": round(mean, 3),
        "latency_ms_p50": round(float(np.percentile(arr, 50)), 3),
        "latency_ms_p95": round(float(np.percentile(arr, 95)), 3),
        "images_per_second": round(1000.0 / mean, 3) if mean > 0 else "Pending",
        "n_iters": iters,
        "warmup": warmup,
    }


def training_time_from_metrics(ckpt: Path) -> float | str:
    # Walk up looking for metrics.json
    for parent in [ckpt.parents[2], ckpt.parents[3], ckpt.parents[1]]:
        for name in ("metrics.json", "run_config.json"):
            m = parent / name
            if m.is_file():
                try:
                    card = json.loads(m.read_text(encoding="utf-8"))
                    for key in ("training_duration_sec", "training_time_sec"):
                        if key in card and not pending(card[key]):
                            return float(card[key])
                except Exception:  # noqa: BLE001
                    pass
    return "Pending"


def main() -> int:
    args = parse_args()
    protocol = load_protocol(Path(args.protocol))
    WEEK6.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    gname = gpu_name(device)
    is_t4 = "T4" in gname
    warmup = 3 if args.smoke else args.warmup
    iters = 10 if args.smoke else args.iters
    imgsz = int(protocol.get("imgsz", 640))
    batch_nominal = int(protocol.get("final_model", {}).get("train_batch_size_nominal", 16))

    from ultralytics import YOLO
    import torch

    rows = []
    seen = set()
    for spec in model_registry(protocol):
        ckpt = resolve_ckpt(spec["checkpoint"])
        key = str(ckpt) if ckpt else spec["model_id"]
        if key in seen and spec.get("identical_to_final"):
            # duplicate row referencing final
            base = next((r for r in rows if r["model_id"] == "final_selected"), None)
            if base:
                dup = dict(base)
                dup["model_id"] = spec["model_id"]
                dup["display_name"] = spec["display_name"]
                dup["identical_to_final"] = True
                rows.append(dup)
            continue
        seen.add(key)
        print(f"[efficiency] {spec['model_id']} device={device} imgsz={imgsz}")
        row = {
            "model_id": spec["model_id"],
            "display_name": spec["display_name"],
            "role": spec["role"],
            "checkpoint": str(ckpt) if ckpt else "Pending",
            "imgsz": imgsz,
            "batch_size_protocol": batch_nominal,
            "device": device,
            "gpu": gname,
            "hardware_target": protocol.get("hardware_target", "NVIDIA T4"),
            "is_t4": is_t4,
            "parameter_count": "Pending",
            "gflops": "Pending",
            "model_file_size_mb": "Pending",
            "latency_ms_mean": "Pending",
            "latency_ms_p50": "Pending",
            "latency_ms_p95": "Pending",
            "images_per_second": "Pending",
            "peak_gpu_memory_mb": "Pending",
            "training_time_sec": "Pending",
            "status": "Pending",
        }
        if ckpt is None:
            row["status"] = "missing_checkpoint"
            rows.append(row)
            continue
        try:
            model = YOLO(str(ckpt))
            if torch.cuda.is_available() and (device.isdigit() or device == "0"):
                torch.cuda.reset_peak_memory_stats()
            n_params = count_parameters(model)
            gflops = gflops_of(model, imgsz)
            size_mb = round(ckpt.stat().st_size / (1024**2), 3)
            # peak mem: one forward
            img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            model.predict(source=img, imgsz=imgsz, device=device, verbose=False)
            peak = peak_gpu_memory_mb(device)
            lat = latency_stats(model, imgsz, device, warmup, iters)
            row.update(
                {
                    "parameter_count": n_params if n_params is not None else "Pending",
                    "gflops": gflops,
                    "model_file_size_mb": size_mb,
                    "latency_ms_mean": lat["latency_ms_mean"],
                    "latency_ms_p50": lat["latency_ms_p50"],
                    "latency_ms_p95": lat["latency_ms_p95"],
                    "images_per_second": lat["images_per_second"],
                    "peak_gpu_memory_mb": peak if peak is not None else ("Pending" if not is_t4 else "Pending"),
                    "training_time_sec": training_time_from_metrics(ckpt),
                    "status": "completed",
                    "t4_note": None
                    if is_t4
                    else f"T4 not available; benchmarked on {gname}. Do not claim T4 numbers.",
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["status"] = f"error: {exc}"
            print(f"  FAILED: {exc}", file=sys.stderr)
        rows.append(row)

    write_csv(WEEK6 / "efficiency_benchmark.csv", rows)
    write_json(WEEK6 / "efficiency_benchmark.json", {"rows": rows, "device": device, "gpu": gname})

    # Accuracy-efficiency tradeoff: join test mAP50 if available
    test_path = WEEK6 / "final_test_results.json"
    map_by_id = {}
    if test_path.is_file():
        data = json.loads(test_path.read_text(encoding="utf-8"))
        for r in data.get("rows", []):
            if r.get("split_label") == "Test" and r.get("status") == "completed":
                map_by_id[r["model_id"]] = r.get("map50")

    fig, ax = plt.subplots(figsize=(8, 6))
    plotted = False
    for r in rows:
        if r.get("identical_to_final"):
            continue
        g = r.get("gflops")
        m = map_by_id.get(r["model_id"], "Pending")
        if pending(g) or pending(m):
            continue
        ax.scatter(float(g), float(m), s=90, label=r["display_name"])
        ax.annotate(r["model_id"], (float(g), float(m)), textcoords="offset points", xytext=(5, 5), fontsize=8)
        plotted = True
    ax.set_xlabel("GFLOPs")
    ax.set_ylabel("Test mAP@0.5")
    ax.set_title("Week 6 — accuracy vs efficiency (Test mAP; not Validation)")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "Pending — need Test mAP + GFLOPs", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(FIGDIR / "accuracy_efficiency_tradeoff.png", dpi=150)
    plt.close(fig)

    print(f"Wrote {WEEK6 / 'efficiency_benchmark.csv'}")
    if not is_t4:
        print(f"NOTE: Hardware is {gname}, not T4. Results labeled with actual device.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
