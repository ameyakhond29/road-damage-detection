#!/usr/bin/env python3
"""Real-data smoke test on a small subset of the OFFICIAL RDD2022 dataset.

PIPELINE VERIFICATION ONLY — the numbers produced here (1–3 epochs on a few hundred
images) are NOT model performance and must never be reported as results.

* Samples ``--n-train`` images from data/splits/train.csv and ``--n-val`` images from
  data/splits/val.csv (total 100–500). The test split is never touched.
* YOLOv8n, 1–3 epochs, imgsz 320 or 640. CUDA when available, otherwise CPU (warning).
* Writes to results/official_rdd2022/smoke_test/: ground-truth samples, prediction
  samples, label statistics, loss curve/CSV, validation mAP, run metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CLASS_NAMES, DEFAULT_SEED  # noqa: E402
from src.dataset.official import (  # noqa: E402
    dataset_fingerprint,
    load_dataset_config,
    official_paths,
    read_csv,
    write_csv,
    write_json,
)
from src.paths import fixture_raw_dir, legacy_fixture_processed_dir  # noqa: E402

DISCLAIMER = "PIPELINE VERIFICATION ONLY — smoke-test numbers are NOT final model performance."
COLORS = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 200, 0)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Official RDD2022 smoke test (pipeline verification).")
    p.add_argument("--config", default=str(ROOT / "configs" / "rdd2022.yaml"))
    p.add_argument("--data-root", default=None, help="Processed root (default data/processed)")
    p.add_argument("--n-train", type=int, default=300)
    p.add_argument("--n-val", type=int, default=100)
    p.add_argument("--epochs", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--imgsz", type=int, default=320, choices=[320, 640])
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="", help="Default: CUDA if available, else CPU")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--n-samples", type=int, default=6, help="GT / prediction sample images")
    p.add_argument("--output-dir", default=None, help="Default results/official_rdd2022/smoke_test")
    return p.parse_args()


def pick_device(requested: str) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "0"
    print("WARNING: CUDA is not available — running the smoke test on CPU (slow).", file=sys.stderr)
    return "cpu"


def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(rows), size=min(n, len(rows)), replace=False).tolist())
    return [rows[i] for i in idx]


def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    for ln in path.read_text().splitlines():
        if ln.strip():
            c, *v = ln.split()
            out.append((int(c), *(float(x) for x in v)))
    return out


def draw_gt(img_path: Path, lbl_path: Path, dest: Path, max_side: int = 960) -> None:
    from PIL import Image, ImageDraw

    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        d = ImageDraw.Draw(im)
        for c, cx, cy, bw, bh in read_labels(lbl_path):
            box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
            d.rectangle(box, outline=COLORS[c], width=max(2, w // 300))
            d.text((box[0] + 3, max(0, box[1] - 12)), CLASS_NAMES[c], fill=COLORS[c])
        im.thumbnail((max_side, max_side))
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest, quality=90)


def plot_losses(results_csv: Path, dest: Path) -> list[dict]:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with results_csv.open() as f:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(f)]
    keys = [k for k in rows[0] if k.endswith("_loss")] if rows else []
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = [int(float(r["epoch"])) for r in rows]
    for k in keys:
        ax.plot(epochs, [float(r[k]) for r in rows], marker="o", label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Official RDD2022 smoke test losses (pipeline check only)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)
    return rows


def main() -> int:
    args = parse_args()
    if not 100 <= args.n_train + args.n_val <= 500:
        print("n-train + n-val must be between 100 and 500", file=sys.stderr)
        return 1
    cfg = load_dataset_config(args.config)
    paths = official_paths(cfg, args.data_root)
    processed = paths["processed"].resolve()
    out = Path(args.output_dir) if args.output_dir else paths["results"] / "smoke_test"
    out = out.resolve()
    splits = paths["splits"]
    summary_path = splits / "split_summary.json"
    if not summary_path.is_file() or json.loads(summary_path.read_text()).get("dataset") != "official_rdd2022":
        print("Official splits missing. Run scripts/run_official_dataset_pipeline.sh first.", file=sys.stderr)
        return 1

    train_rows = sample_rows(read_csv(splits / "train.csv"), args.n_train, args.seed)
    val_rows = sample_rows(read_csv(splits / "val.csv"), args.n_val, args.seed + 1)
    forbidden = (legacy_fixture_processed_dir().resolve(), fixture_raw_dir().resolve())
    for r in train_rows + val_rows:
        p = (processed / r["image_path"]).resolve()
        if any(f == p or f in p.parents for f in forbidden) or not p.is_file():
            print(f"Refusing non-official or missing image: {p}", file=sys.stderr)
            return 1

    if out.exists():
        shutil.rmtree(out)
    data_dir = out / "data"
    data_dir.mkdir(parents=True)
    lists = {}
    for name, rows in (("train", train_rows), ("val", val_rows)):
        lists[name] = data_dir / f"{name}.txt"
        lists[name].write_text("\n".join(str(processed / r["image_path"]) for r in rows) + "\n")
        write_csv(data_dir / f"{name}_subset.csv", rows)
    import yaml

    data_yaml = data_dir / "smoke_official.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(processed), "train": str(lists["train"]), "val": str(lists["val"]),
        "nc": 4, "names": dict(enumerate(CLASS_NAMES)),
    }, sort_keys=False))

    # label statistics of the subset
    stats_rows = []
    for name, rows in (("train", train_rows), ("val", val_rows)):
        cls = Counter()
        for r in rows:
            for n in CLASS_NAMES:
                cls[n] += int(r[f"n_{n}"])
        stats_rows.append({"subset": name, "n_images": len(rows),
                           "n_negatives": sum(int(r["is_negative"]) for r in rows),
                           "n_objects": sum(int(r["n_objects"]) for r in rows), **dict(cls),
                           "countries": json.dumps(dict(sorted(Counter(r["country"] for r in rows).items())))})
    write_csv(out / "label_stats.csv", stats_rows)

    # ground-truth samples
    by_country: dict[str, list[dict]] = {}
    for r in train_rows:
        if int(r["n_objects"]) > 0:
            by_country.setdefault(r["country"], []).append(r)
    positives = [rows[k] for k in range(args.n_samples) for rows in by_country.values() if k < len(rows)]
    positives = positives[: args.n_samples]
    for r in positives:
        draw_gt(processed / r["image_path"], processed / r["label_path"],
                out / "gt_samples" / f"{r['country']}__{Path(r['image_path']).stem}.jpg")

    import torch
    from ultralytics import YOLO

    from src.training.baseline import software_versions

    device = pick_device(args.device)
    amp = device not in {"cpu", "mps"} and torch.cuda.is_available()
    subset_hash = hashlib.sha256("".join(lists[k].read_text() for k in ("train", "val")).encode()).hexdigest()[:16]
    meta = {
        "purpose": DISCLAIMER,
        "dataset": "official_rdd2022",
        "dataset_version": cfg.get("dataset_version"),
        "dataset_source": cfg.get("source"),
        **dataset_fingerprint(splits),
        "subset_hash": subset_hash,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "test_split_used": False,
        "model": args.model,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "seed": args.seed,
        "device": device,
        "amp": amp,
        "command": " ".join(sys.argv),
        "software": software_versions(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    write_json(out / "run_metadata.json", meta)
    print(DISCLAIMER)
    print(f"device={device} imgsz={args.imgsz} epochs={args.epochs} train={len(train_rows)} val={len(val_rows)}")

    t0 = time.perf_counter()
    model = YOLO(str(ROOT / args.model) if (ROOT / args.model).is_file() else args.model)
    res = model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      device=device, amp=amp, seed=args.seed, deterministic=True, workers=args.workers,
                      project=str(out), name="train", exist_ok=True, plots=True, val=True, verbose=False)
    train_s = time.perf_counter() - t0
    run_dir = Path(res.save_dir) if res is not None and hasattr(res, "save_dir") else out / "train"
    best = run_dir / "weights" / "best.pt"
    ckpt = best if best.is_file() else run_dir / "weights" / "last.pt"

    loss_rows = []
    if (run_dir / "results.csv").is_file():
        shutil.copy2(run_dir / "results.csv", out / "loss_and_metrics_per_epoch.csv")
        loss_rows = plot_losses(run_dir / "results.csv", out / "loss_curve.png")

    eval_model = YOLO(str(ckpt))
    val = eval_model.val(data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch,
                         device=device, plots=False, verbose=False, project=str(out), name="val", exist_ok=True)
    rd = {k: float(v) for k, v in val.results_dict.items()}
    val_metrics = {
        "WARNING": DISCLAIMER,
        "split": "val (smoke subset)",
        "n_images": len(val_rows),
        "mAP50": rd.get("metrics/mAP50(B)"),
        "mAP50_95": rd.get("metrics/mAP50-95(B)"),
        "precision": rd.get("metrics/precision(B)"),
        "recall": rd.get("metrics/recall(B)"),
        "per_class_AP50": {CLASS_NAMES[int(c)]: float(a) for c, a in zip(val.box.ap_class_index, val.box.ap50)},
    }
    write_json(out / "smoke_val_metrics.json", val_metrics)

    pred_dir = out / "pred_samples"
    pred_dir.mkdir(exist_ok=True)
    sample_val = sorted(val_rows, key=lambda r: -int(r["n_objects"]))[: args.n_samples]
    from PIL import Image

    for r in sample_val:
        pr = eval_model.predict(str(processed / r["image_path"]), imgsz=args.imgsz, conf=0.25,
                                device=device, verbose=False)[0]
        im = Image.fromarray(pr.plot()[..., ::-1])
        im.thumbnail((960, 960))
        im.save(pred_dir / f"{r['country']}__{Path(r['image_path']).stem}.jpg", quality=90)
        draw_gt(processed / r["image_path"], processed / r["label_path"],
                pred_dir / f"{r['country']}__{Path(r['image_path']).stem}__gt.jpg")

    meta.update({
        "status": "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "train_seconds": round(train_s, 1),
        "checkpoint": str(ckpt),
        "final_epoch_losses": {k: float(v) for k, v in loss_rows[-1].items() if k.endswith("_loss")} if loss_rows else None,
        "smoke_val_metrics": val_metrics,
    })
    write_json(out / "run_metadata.json", meta)
    (out / "README.md").write_text(
        f"# Official RDD2022 smoke test\n\n**{DISCLAIMER}**\n\n"
        f"- {len(train_rows)} train / {len(val_rows)} val images sampled from the official splits "
        f"(seed {args.seed}); test split not used.\n"
        f"- YOLOv8n, {args.epochs} epoch(s), imgsz {args.imgsz}, device `{device}`.\n"
        "- Files: `gt_samples/`, `pred_samples/`, `label_stats.csv`, `loss_curve.png`, "
        "`loss_and_metrics_per_epoch.csv`, `smoke_val_metrics.json`, `run_metadata.json`.\n",
        encoding="utf-8",
    )
    print(f"Smoke val (NOT performance): mAP50={val_metrics['mAP50']} mAP50-95={val_metrics['mAP50_95']}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
