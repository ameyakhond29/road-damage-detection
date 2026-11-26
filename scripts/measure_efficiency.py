#!/usr/bin/env python3
"""Week 5 — Measure parameter count, GFLOPs, latency, peak GPU memory.

Works on:
  - stock checkpoints (.pt)
  - custom architecture YAMLs
  - trained Week 5 best.pt files when present

Writes results/week5/efficiency_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.resolve import MODEL_REGISTRY, resolve_model_spec
from src.training.baseline import (
    count_parameters,
    gpu_name,
    peak_gpu_memory_mb,
    pick_device,
    profile_latency,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure Week 5 model efficiency.")
    p.add_argument("--week5-root", type=str, default=str(ROOT / "results" / "week5"))
    p.add_argument("--device", type=str, default="")
    p.add_argument("--seed", type=int, default=0, help="Prefer this seed's checkpoint if present")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--smoke", action="store_true", help="Fewer latency iterations")
    return p.parse_args()


def gflops_of(model, imgsz: int = 640) -> float | None:
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
        return None


def find_ckpt(week5: Path, experiment: str, seed: int) -> Path | None:
    metrics = week5 / experiment / f"seed_{seed}" / "metrics.json"
    if metrics.is_file():
        card = json.loads(metrics.read_text(encoding="utf-8"))
        ckpt = card.get("best_checkpoint")
        if isinstance(ckpt, str) and Path(ckpt).is_file():
            return Path(ckpt)
    for name in ("best.pt", "last.pt"):
        cand = week5 / experiment / f"seed_{seed}" / "train" / "weights" / name
        if cand.is_file():
            return cand
    return None


def main() -> int:
    args = parse_args()
    week5 = Path(args.week5_root)
    week5.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    gname = gpu_name(device)
    warmup = 3 if args.smoke else args.warmup
    iters = 10 if args.smoke else args.iters

    from ultralytics import YOLO
    import torch
    import numpy as np

    rows = []
    for exp, meta in MODEL_REGISTRY.items():
        spec = resolve_model_spec(experiment=exp)
        imgsz = int(meta["imgsz_default"])
        ckpt = find_ckpt(week5, exp, args.seed)
        source = str(ckpt) if ckpt else spec["model"]
        status = "checkpoint" if ckpt else "architecture_init"

        print(f"[efficiency] {exp} source={source} imgsz={imgsz} device={device}")
        try:
            model = YOLO(source)
            if (
                not ckpt
                and str(spec["model"]).endswith((".yaml", ".yml"))
                and spec.get("pretrained")
            ):
                try:
                    model = YOLO(spec["model"]).load(spec["pretrained"])
                except Exception:  # noqa: BLE001
                    pass

            if torch.cuda.is_available() and (device.isdigit() or device == "0"):
                torch.cuda.reset_peak_memory_stats()

            n_params = count_parameters(model)
            gflops = gflops_of(model, imgsz=imgsz)

            # Peak memory during a short forward/predict
            img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            if torch.cuda.is_available() and (device.isdigit() or device == "0"):
                torch.cuda.reset_peak_memory_stats()
            model.predict(source=img, imgsz=imgsz, device=device, verbose=False)
            peak = peak_gpu_memory_mb(device)

            latency = profile_latency(
                model, imgsz=imgsz, device=device, warmup=warmup, iters=iters
            )

            row = {
                "experiment": exp,
                "model_variant": meta["label"],
                "source": source,
                "status": status,
                "input_resolution": imgsz,
                "parameter_count": n_params if n_params is not None else "Pending",
                "gflops": gflops if gflops is not None else "Pending",
                "peak_gpu_memory_mb": peak if peak is not None else "Pending",
                "inference_latency_ms_mean": latency.get("latency_ms_mean", "Pending"),
                "inference_latency_ms_std": latency.get("latency_ms_std", "Pending"),
                "throughput_img_s": latency.get("throughput_img_s", "Pending"),
                "device": device,
                "gpu": gname,
                "seed_preferred": args.seed,
            }
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {exc}", file=sys.stderr)
            row = {
                "experiment": exp,
                "model_variant": meta["label"],
                "source": source,
                "status": f"error: {exc}",
                "input_resolution": imgsz,
                "parameter_count": "Pending",
                "gflops": "Pending",
                "peak_gpu_memory_mb": "Pending",
                "inference_latency_ms_mean": "Pending",
                "inference_latency_ms_std": "Pending",
                "throughput_img_s": "Pending",
                "device": device,
                "gpu": gname,
                "seed_preferred": args.seed,
            }
        rows.append(row)

    out_csv = week5 / "efficiency_results.csv"
    out_json = week5 / "efficiency_results.json"
    keys = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    write_json(out_json, {"rows": rows, "device": device, "gpu": gname})
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
