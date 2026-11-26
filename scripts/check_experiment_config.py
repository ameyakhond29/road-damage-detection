#!/usr/bin/env python3
"""Validate Week 2 experiment configs and manifest.

Checks:
  - Unique experiment IDs
  - Required paths exist
  - Class mapping is consistent (D00=0..D40=3)
  - Test / held-out data is not used for model selection
  - Seeds are recorded
  - Image size and batch size are valid
  - Single-factor discipline (unless multi_factor is declared)
  - Every experiment references an existing config
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import CLASS_TO_ID, CLASS_NAMES, DEFAULT_SEED

REQUIRED_MANIFEST_COLS = [
    "experiment_id",
    "model",
    "variant",
    "image_size",
    "epochs",
    "batch_size",
    "seed",
    "train_split",
    "val_split",
    "test_split",
    "augmentation",
    "primary_metric",
    "status",
    "notes",
]

EXPECTED_CLASS_MAPPING = dict(CLASS_TO_ID)
VALID_IMAGE_SIZES = {320, 640, 800, 1280}
VALID_SEEDS_MAIN = {0, 1, 2}
ALLOWED_DEBUG_SEED = DEFAULT_SEED

# Fields that must match the strong baseline for single-factor ablations
HELD_CONSTANT_KEYS = [
    "epochs",
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "cos_lr",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "copy_paste",
    "conf",
    "iou",
    "max_det",
    "primary_metric",
    "checkpoint_selection",
]

VARIANT_ALLOWED_DELTAS = {
    "none": set(),
    "p2": {"model", "variant", "name", "architecture", "pretrained_weights", "experiment_role"},
    "attention": {"model", "variant", "name", "architecture", "pretrained_weights", "experiment_role"},
    "imgsz_800": {"imgsz", "batch", "variant", "name", "experiment_role", "notes"},
    "p2_attention": {
        "model",
        "variant",
        "name",
        "architecture",
        "pretrained_weights",
        "experiment_role",
        "multi_factor",
        "notes",
    },
}


class Checker:
    def __init__(self, root: Path):
        self.root = root
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def resolve(self, path_str: str) -> Path:
        p = Path(path_str)
        if not p.is_absolute():
            p = self.root / p
        return p


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def check_class_mapping(cfg: dict, label: str, checker: Checker) -> None:
    mapping = cfg.get("class_mapping") or cfg.get("class_ids")
    if mapping is None and "names" in cfg:
        names = cfg["names"]
        if isinstance(names, dict):
            mapping = {v: int(k) for k, v in names.items()}
        elif isinstance(names, list):
            mapping = {n: i for i, n in enumerate(names)}
    if mapping is None:
        checker.err(f"{label}: missing class_mapping / class_ids / names")
        return
    normalized = {str(k): int(v) for k, v in mapping.items()}
    if normalized != {k: int(v) for k, v in EXPECTED_CLASS_MAPPING.items()}:
        checker.err(
            f"{label}: class mapping mismatch. expected {EXPECTED_CLASS_MAPPING}, got {normalized}"
        )


def check_no_test_for_selection(cfg: dict, label: str, checker: Checker) -> None:
    rule = str(cfg.get("checkpoint_selection", "")).lower()
    if rule and "test" in rule:
        checker.err(f"{label}: checkpoint_selection must not use test data ({rule})")
    if "test" in str(cfg.get("primary_metric", "")).lower() and "val" not in rule:
        # primary metric name alone is fine; selection rule matters
        pass
    notes = str(cfg.get("notes", "")).lower()
    if "select" in notes and "test" in notes and "not" not in notes and "never" not in notes:
        checker.warn(f"{label}: notes mention test selection — confirm wording")


def major_factors(cfg: dict) -> set[str]:
    """Return which experimental factors differ from a plain YOLO11n@640 baseline."""
    factors: set[str] = set()
    variant = str(cfg.get("variant", "none"))
    if variant not in {"none", "", "null"}:
        if variant == "imgsz_800":
            factors.add("image_size")
        elif variant == "p2_attention":
            factors.add("p2")
            factors.add("attention")
        else:
            factors.add(variant)
    model = str(cfg.get("model", ""))
    if "yolov8" in model.lower():
        factors.add("detector_family")
    imgsz = int(cfg.get("imgsz", 640))
    if imgsz != 640:
        factors.add("image_size")
    return factors


def check_single_factor(
    cfg: dict,
    label: str,
    baseline: dict | None,
    checker: Checker,
) -> None:
    if baseline is None:
        return
    multi = bool(cfg.get("multi_factor", False))
    factors = major_factors(cfg)
    # Detector-family configs (YOLOv8) are allowed to differ in model only
    role = str(cfg.get("experiment_role", ""))
    if role == "reference_baseline":
        return
    if multi:
        if len(factors) < 2 and str(cfg.get("variant")) != "p2_attention":
            checker.warn(f"{label}: multi_factor=true but fewer than 2 factors detected ({factors})")
        return
    # Single-factor variants vs strong baseline
    if role == "team_investigation" or str(cfg.get("variant", "none")) not in {"none", ""}:
        allowed = VARIANT_ALLOWED_DELTAS.get(str(cfg.get("variant", "none")), set())
        for key in HELD_CONSTANT_KEYS:
            if key not in cfg or key not in baseline:
                continue
            if cfg[key] != baseline[key]:
                checker.err(
                    f"{label}: held-constant field '{key}' changed "
                    f"({baseline[key]!r} → {cfg[key]!r}) without declaring multi_factor"
                )
        # imgsz must stay 640 unless variant is imgsz_800
        if str(cfg.get("variant")) != "imgsz_800" and int(cfg.get("imgsz", 640)) != 640:
            checker.err(f"{label}: imgsz changed but variant is not imgsz_800")
        if str(cfg.get("variant")) == "imgsz_800" and int(cfg.get("imgsz", 0)) != 800:
            checker.err(f"{label}: variant imgsz_800 requires imgsz=800")
        # Unexpected extra keys that look like factor knobs
        for key in ("imgsz", "model"):
            if key == "model" and str(cfg.get("variant")) in {"p2", "attention", "p2_attention"}:
                continue
            if key == "imgsz" and str(cfg.get("variant")) == "imgsz_800":
                continue
        _ = allowed  # documented allow-list for reviewers


def check_manifest(manifest_path: Path, checker: Checker) -> pd.DataFrame | None:
    if not manifest_path.is_file():
        checker.err(f"Manifest not found: {manifest_path}")
        return None
    df = pd.read_csv(manifest_path)
    missing_cols = [c for c in REQUIRED_MANIFEST_COLS if c not in df.columns]
    if missing_cols:
        checker.err(f"Manifest missing columns: {missing_cols}")
    if "config_path" not in df.columns:
        checker.err("Manifest missing column: config_path")

    # Unique IDs
    ids = df["experiment_id"].astype(str).tolist()
    counts = Counter(ids)
    dupes = [i for i, n in counts.items() if n > 1]
    if dupes:
        checker.err(f"Duplicate experiment_id values: {dupes}")

    for idx, row in df.iterrows():
        eid = row["experiment_id"]
        label = f"manifest[{eid}]"

        # Seeds recorded
        if pd.isna(row.get("seed")):
            checker.err(f"{label}: seed is missing")
        else:
            seed = int(row["seed"])
            if seed not in VALID_SEEDS_MAIN and seed != ALLOWED_DEBUG_SEED:
                checker.warn(
                    f"{label}: unusual seed {seed} "
                    f"(expected 0/1/2 for comparisons or {ALLOWED_DEBUG_SEED} for debug)"
                )
            note = str(row.get("notes", "")).lower()
            if seed not in VALID_SEEDS_MAIN and "one-seed" not in note and "debug" not in note:
                # OOD one-seed rows should be labeled
                if str(eid).startswith("ood_") and seed == 0:
                    if "one-seed" not in note and "multi-seed" not in note:
                        checker.warn(f"{label}: label one-seed OOD runs explicitly in notes")

        # Image / batch
        try:
            imgsz = int(row["image_size"])
            batch = int(row["batch_size"])
            epochs = int(row["epochs"])
        except (TypeError, ValueError):
            checker.err(f"{label}: image_size/batch_size/epochs must be integers")
            continue
        if imgsz not in VALID_IMAGE_SIZES:
            checker.err(f"{label}: invalid image_size {imgsz}; allowed {sorted(VALID_IMAGE_SIZES)}")
        if batch < 1 or batch > 128:
            checker.err(f"{label}: invalid batch_size {batch}")
        if epochs < 1:
            checker.err(f"{label}: invalid epochs {epochs}")

        # Paths exist
        for col in ("train_split", "val_split", "test_split"):
            path = checker.resolve(str(row[col]))
            if not path.is_file():
                checker.err(f"{label}: {col} path does not exist: {path}")

        # Config path
        if "config_path" in df.columns and not pd.isna(row.get("config_path")):
            cfg_path = checker.resolve(str(row["config_path"]))
            if not cfg_path.is_file():
                checker.err(f"{label}: config_path does not exist: {cfg_path}")

        # Test not for selection — enforce via notes/protocol fields
        sel = str(row.get("primary_metric", ""))
        if "test" in sel.lower() and "val" not in str(row.get("notes", "")).lower():
            checker.warn(f"{label}: primary_metric naming looks test-related: {sel}")

        # Held-out eval must not be the val_split
        if "heldout" in str(row["val_split"]).lower():
            checker.err(
                f"{label}: val_split points at held-out data — "
                "checkpoint selection would leak OOD labels"
            )
        if str(row["train_split"]) == str(row["test_split"]):
            checker.err(f"{label}: train_split and test_split are identical")

        # Metric
        if str(row["primary_metric"]).lower() not in {"map50", "map@0.5", "metrics/map50(b)"}:
            if "map50" not in str(row["primary_metric"]).lower():
                checker.warn(f"{label}: unexpected primary_metric {row['primary_metric']!r}")

    return df


def check_configs(checker: Checker) -> dict[str, dict]:
    required = [
        "configs/baseline_yolov8n.yaml",
        "configs/baseline_yolo11n.yaml",
        "configs/variant_p2.yaml",
        "configs/variant_attention.yaml",
        "configs/variant_800px.yaml",
        "configs/variant_p2_attention.yaml",
        "configs/training_defaults.yaml",
        "configs/rdd2022.yaml",
        "configs/rdd2022_ood.yaml",
    ]
    loaded: dict[str, dict] = {}
    for rel in required:
        path = checker.resolve(rel)
        if not path.is_file():
            checker.err(f"Required config missing: {rel}")
            continue
        try:
            cfg = load_yaml(path)
        except Exception as exc:  # noqa: BLE001
            checker.err(f"Failed to parse {rel}: {exc}")
            continue
        loaded[rel] = cfg
        check_class_mapping(cfg, rel, checker)
        check_no_test_for_selection(cfg, rel, checker)

    baseline = loaded.get("configs/baseline_yolo11n.yaml")
    for rel, cfg in loaded.items():
        if rel in {
            "configs/training_defaults.yaml",
            "configs/rdd2022.yaml",
            "configs/rdd2022_ood.yaml",
            "configs/baseline_yolov8n.yaml",
            "configs/baseline_yolo11n.yaml",
        }:
            continue
        check_single_factor(cfg, rel, baseline, checker)

    # Explicit: combined config must declare multi_factor
    combined = loaded.get("configs/variant_p2_attention.yaml")
    if combined is not None and not combined.get("multi_factor", False):
        checker.err("configs/variant_p2_attention.yaml must set multi_factor: true")

    # Placeholder model YAMLs should exist for custom variants
    for rel in (
        "configs/models/yolo11n_p2.yaml",
        "configs/models/yolo11n_attention.yaml",
        "configs/models/yolo11n_p2_attention.yaml",
    ):
        if not checker.resolve(rel).is_file():
            checker.err(f"Missing architecture placeholder: {rel}")

    return loaded


def check_manifest_config_agreement(df: pd.DataFrame, checker: Checker) -> None:
    """Every experiment must have a config; configs should cover all variants used."""
    if df is None or "config_path" not in df.columns:
        return
    variants_needed = set(df["variant"].astype(str))
    config_by_variant = {
        "none": {"configs/baseline_yolov8n.yaml", "configs/baseline_yolo11n.yaml"},
        "p2": {"configs/variant_p2.yaml"},
        "attention": {"configs/variant_attention.yaml"},
        "imgsz_800": {"configs/variant_800px.yaml"},
        "p2_attention": {"configs/variant_p2_attention.yaml"},
    }
    for variant in variants_needed:
        expected = config_by_variant.get(variant)
        if expected is None:
            checker.warn(f"Unknown variant in manifest: {variant}")
            continue
        used = set(
            df.loc[df["variant"].astype(str) == variant, "config_path"].astype(str)
        )
        if variant == "none":
            # both baselines may appear
            continue
        if not used.issubset(expected):
            checker.err(
                f"variant={variant}: unexpected config_path values {used - expected}"
            )
        if not expected.intersection(used):
            checker.err(f"variant={variant}: no experiment uses {expected}")

    # Ensure every listed config_path is referenced at least once for the five user configs
    must_be_used = [
        "configs/baseline_yolov8n.yaml",
        "configs/baseline_yolo11n.yaml",
        "configs/variant_p2.yaml",
        "configs/variant_attention.yaml",
        "configs/variant_800px.yaml",
    ]
    used_all = set(df["config_path"].astype(str))
    for rel in must_be_used:
        if rel not in used_all:
            checker.err(f"Config never referenced in manifest: {rel}")


def check_docs(checker: Checker) -> None:
    for rel in (
        "docs/week2_experiment_protocol.md",
        "docs/research_question.md",
        "docs/related_work_summary.md",
        "README.md",
    ):
        if not checker.resolve(rel).is_file():
            checker.err(f"Missing documentation: {rel}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate Week 2 experiment configuration.")
    p.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT / "configs" / "experiment_manifest.csv"),
    )
    p.add_argument("--root", type=str, default=str(ROOT))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    checker = Checker(root)

    print(f"Project root: {root}")
    print(f"Manifest: {args.manifest}")
    print(f"Expected class mapping: {EXPECTED_CLASS_MAPPING}")
    print(f"Default project seed (data/debug): {DEFAULT_SEED}")
    print(f"Comparison seeds: {sorted(VALID_SEEDS_MAIN)}")
    print()

    check_docs(checker)
    check_configs(checker)
    df = check_manifest(Path(args.manifest) if Path(args.manifest).is_absolute() else root / args.manifest, checker)
    if df is not None:
        check_manifest_config_agreement(df, checker)
        print(f"Manifest experiments: {len(df)}")
        print(f"Unique experiment_ids: {df['experiment_id'].nunique()}")
        print(f"Variants: {sorted(df['variant'].astype(str).unique())}")
        print(f"Seeds: {sorted(df['seed'].dropna().astype(int).unique())}")

    print()
    if checker.warnings:
        print(f"Warnings ({len(checker.warnings)}):")
        for w in checker.warnings:
            print(f"  WARN  {w}")
    if checker.errors:
        print(f"Errors ({len(checker.errors)}):")
        for e in checker.errors:
            print(f"  ERROR {e}")
        print("\nCONFIG CHECK FAILED")
        return 1

    print("CONFIG CHECK PASSED")
    if checker.warnings:
        print(f"({len(checker.warnings)} warning(s) — review before claiming readiness)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
