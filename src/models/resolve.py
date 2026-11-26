"""Resolve Week 5 experiment model identifiers to Ultralytics-loadable paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# experiment_name -> (ultralytics model path or .pt, human label, factors)
MODEL_REGISTRY: dict[str, dict] = {
    "yolo11n_640": {
        "model": "yolo11n.pt",
        "label": "YOLO11n baseline",
        "imgsz_default": 640,
        "factors": ["baseline"],
        "architecture": "stock YOLO11n (P3/P4/P5)",
    },
    "yolo11n_p2_640": {
        "model": str(ROOT / "configs" / "models" / "yolo11n-p2.yaml"),
        "pretrained": "yolo11n.pt",
        "label": "YOLO11n + P2 head",
        "imgsz_default": 640,
        "factors": ["p2"],
        "architecture": "YOLO11n backbone + Detect(P2,P3,P4,P5)",
    },
    "yolo11n_attention_640": {
        "model": str(ROOT / "configs" / "models" / "yolo11n-attn.yaml"),
        "pretrained": "yolo11n.pt",
        "label": "YOLO11n + neck attention (C2PSA@P3)",
        "imgsz_default": 640,
        "factors": ["attention"],
        "architecture": "YOLO11n + C2PSA on P3 neck feature",
    },
    "yolo11n_800": {
        "model": "yolo11n.pt",
        "label": "YOLO11n @ 800px",
        "imgsz_default": 800,
        "factors": ["imgsz_800"],
        "architecture": "stock YOLO11n; input 800",
    },
    "combined_800": {
        "model": str(ROOT / "configs" / "models" / "yolo11n-p2-attn.yaml"),
        "pretrained": "yolo11n.pt",
        "label": "Combined P2+attention@800",
        "imgsz_default": 800,
        "factors": ["p2", "attention", "imgsz_800"],
        "architecture": "P2 Detect + C2PSA@P3; input 800",
    },
}

# Alias stub YAML paths from Week 2 to implemented files
_STUB_ALIASES = {
    "configs/models/yolo11n_p2.yaml": "configs/models/yolo11n-p2.yaml",
    "configs/models/yolo11n_attention.yaml": "configs/models/yolo11n-attn.yaml",
    "configs/models/yolo11n_p2_attention.yaml": "configs/models/yolo11n-p2-attn.yaml",
}


def resolve_model_spec(experiment: str | None = None, model: str | None = None) -> dict:
    """Return a resolved model dict for training/eval.

    Prefer ``experiment`` key from MODEL_REGISTRY. If only ``model`` is given,
    resolve stub aliases to the real Ultralytics YAML.
    """
    if experiment:
        if experiment not in MODEL_REGISTRY:
            raise KeyError(f"Unknown Week 5 experiment: {experiment}")
        spec = dict(MODEL_REGISTRY[experiment])
        spec["experiment"] = experiment
        model_path = Path(spec["model"])
        if model_path.suffix in {".yaml", ".yml"} and not model_path.is_file():
            raise FileNotFoundError(f"Model YAML missing: {model_path}")
        return spec

    if not model:
        raise ValueError("Provide experiment= or model=")

    rel = model
    # Normalize to repo-relative if under ROOT
    p = Path(model)
    if p.is_absolute():
        try:
            rel = str(p.relative_to(ROOT))
        except ValueError:
            rel = str(p)
    if rel in _STUB_ALIASES:
        resolved = ROOT / _STUB_ALIASES[rel]
        return {
            "experiment": None,
            "model": str(resolved),
            "pretrained": "yolo11n.pt",
            "label": rel,
            "imgsz_default": 640,
            "factors": ["custom"],
            "architecture": f"alias {rel} -> {_STUB_ALIASES[rel]}",
        }
    return {
        "experiment": None,
        "model": str(p if p.is_absolute() else (ROOT / p if (ROOT / p).exists() else p)),
        "label": str(model),
        "imgsz_default": 640,
        "factors": [],
        "architecture": "user-specified",
    }
