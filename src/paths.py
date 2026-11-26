"""Path helpers and raw-dataset discovery.

Dataset selection rules:
  * ``official`` (default): raw root = CLI path > ``$RDD2022_ROOT`` > ``data/raw/RDD2022``;
    processed root = ``data/processed`` (``images/``, ``labels/``).
  * ``fixture``: synthetic data in ``data/fixture/RDD2022`` — unit tests / debugging only,
    selected exclusively via ``--dataset fixture``.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.constants import (
    DOWNLOAD_INSTRUCTIONS,
    IMAGE_EXTENSIONS,
    KNOWN_COUNTRIES,
    RDD2022_ROOT_ENV,
)

DATASET_CHOICES = ("official", "fixture")


def project_root() -> Path:
    """Return the repository root (parent of ``src/``)."""
    return Path(__file__).resolve().parents[1]


def default_raw_dir() -> Path:
    return project_root() / "data" / "raw" / "RDD2022"


def default_processed_dir() -> Path:
    """Official processed root (``images/<country>``, ``labels/<country>``)."""
    return project_root() / "data" / "processed"


def default_splits_dir() -> Path:
    return project_root() / "data" / "splits"


def official_results_dir() -> Path:
    return project_root() / "results" / "official_rdd2022"


def fixture_raw_dir() -> Path:
    return project_root() / "data" / "fixture" / "RDD2022"


def legacy_fixture_processed_dir() -> Path:
    """Processed synthetic fixture used by all Week 1–7 legacy results."""
    return project_root() / "data" / "processed" / "RDD2022_yolo"


def legacy_fixture_splits_dir() -> Path:
    return project_root() / "data" / "splits" / "legacy_fixture"


def legacy_results_dirs() -> tuple[Path, ...]:
    """Fixture-derived result folders that new runs must never write into."""
    root = project_root() / "results"
    return tuple(root / f"week{i}" for i in range(1, 8))


def is_legacy_results_path(path: str | Path) -> bool:
    p = Path(path).expanduser().resolve()
    for legacy in legacy_results_dirs():
        try:
            p.relative_to(legacy.resolve())
            return True
        except ValueError:
            continue
    return False


def resolve_dataset_root(
    cli_path: str | Path | None = None,
    dataset: str = "official",
    *,
    must_exist: bool = True,
) -> Path:
    """Resolve the *raw* VOC dataset root.

    ``official``: ``cli_path`` > ``$RDD2022_ROOT`` > ``data/raw/RDD2022``; the result is
    searched recursively for the folder holding the country sub-folders.
    ``fixture``: ``data/fixture/RDD2022`` (``cli_path`` may override for tests).
    """
    if dataset not in DATASET_CHOICES:
        raise ValueError(f"dataset must be one of {DATASET_CHOICES}, got {dataset!r}")
    if dataset == "fixture":
        path = Path(cli_path) if cli_path else fixture_raw_dir()
    elif cli_path:
        path = Path(cli_path)
    elif os.environ.get(RDD2022_ROOT_ENV):
        path = Path(os.environ[RDD2022_ROOT_ENV])
    else:
        path = default_raw_dir()
    path = path.expanduser()
    path = (path if path.is_absolute() else Path.cwd() / path).resolve()
    if not must_exist:
        return path
    if not path.exists():
        raise FileNotFoundError(
            f"[{dataset}] dataset root does not exist: {path}\n\n" + DOWNLOAD_INSTRUCTIONS
        )
    found = find_rdd2022_root(path)
    if found is None:
        raise FileNotFoundError(
            f"[{dataset}] no RDD2022 country folders found under: {path}\n\n"
            + DOWNLOAD_INSTRUCTIONS
        )
    return found


def find_rdd2022_root(path: Path, max_depth: int = 4) -> Path | None:
    """Breadth-first search for the directory whose children are country folders."""
    frontier = [Path(path)]
    for _ in range(max_depth + 1):
        nxt: list[Path] = []
        for d in frontier:
            if not d.is_dir():
                continue
            if discover_countries(d):
                return d
            try:
                nxt.extend(c for c in sorted(d.iterdir()) if c.is_dir() and not c.name.startswith("."))
            except PermissionError:
                continue
        frontier = nxt
        if not frontier:
            break
    return None


def resolve_processed_root(
    cli_path: str | Path | None = None, dataset: str = "official"
) -> Path:
    """Processed (YOLO) root: CLI path > dataset default."""
    if cli_path:
        p = Path(cli_path).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p).resolve()
    return default_processed_dir() if dataset == "official" else legacy_fixture_processed_dir()


def default_results_dir() -> Path:
    return project_root() / "results"


def default_figures_dir() -> Path:
    return project_root() / "figures"


def default_configs_dir() -> Path:
    return project_root() / "configs"


def default_rdd2022_config() -> Path:
    return default_configs_dir() / "rdd2022.yaml"


def legacy_fixture_data_yaml() -> Path:
    return legacy_fixture_processed_dir() / "rdd2022_fixture.yaml"


def ensure_legacy_fixture_data_yaml(out: Path | None = None) -> Path:
    """Write the Ultralytics YAML for the *legacy synthetic fixture* (Weeks 1–7).

    Ultralytics resolves relative ``path`` entries against its global datasets
    directory, so ``path`` is absolute and the file lives under the gitignored
    ``data/processed/RDD2022_yolo/``. It never touches ``configs/rdd2022.yaml``
    (the tracked official dataset config).
    """
    processed = legacy_fixture_processed_dir()
    dest = out or legacy_fixture_data_yaml()
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = (
        "# LEGACY SYNTHETIC FIXTURE — unit tests / debugging only. Not RDD2022.\n"
        "# Classes: D00=0, D10=1, D20=2, D40=3\n"
        "# Generated by src.paths.ensure_legacy_fixture_data_yaml()\n"
        "dataset: fixture\n"
        f"path: {processed.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 4\n"
        "names:\n"
        "  0: D00\n"
        "  1: D10\n"
        "  2: D20\n"
        "  3: D40\n"
        "class_ids:\n"
        "  D00: 0\n"
        "  D10: 1\n"
        "  D20: 2\n"
        "  D40: 3\n"
        "imgsz: 640\n"
        "dataset_version: synthetic-fixture-week3\n"
        "split_id: grouped_70_15_15_seed6845\n"
    )
    dest.write_text(text, encoding="utf-8")
    return dest


def resolve_raw_dir(raw_dir: str | Path | None = None) -> Path:
    """Resolve and validate a raw VOC root (official resolution rules)."""
    return resolve_dataset_root(raw_dir, "official")


def discover_countries(raw_dir: Path) -> list[str]:
    """Return sorted country folder names that look like RDD2022 subsets."""
    countries: list[str] = []
    if not raw_dir.is_dir():
        return countries
    for child in sorted(raw_dir.iterdir()):
        if not child.is_dir():
            continue
        # Prefer known names; also accept any folder with train/images.
        train_images = child / "train" / "images"
        images = child / "images"
        if child.name in KNOWN_COUNTRIES or train_images.is_dir() or images.is_dir():
            countries.append(child.name)
    return countries


def country_train_dirs(raw_dir: Path, country: str) -> tuple[Path, Path]:
    """Return (images_dir, annotations_dir) for a country train split.

    Supports both layouts:
      Country/train/images + Country/train/annotations/xmls
      Country/images + Country/annotations/xmls
    """
    base = raw_dir / country
    candidates = [
        (base / "train" / "images", base / "train" / "annotations" / "xmls"),
        (base / "train" / "images", base / "train" / "annotations"),
        (base / "images", base / "annotations" / "xmls"),
        (base / "images", base / "annotations"),
    ]
    for images_dir, ann_dir in candidates:
        if images_dir.is_dir():
            return images_dir, ann_dir
    raise FileNotFoundError(
        f"Could not find images for country '{country}' under {base}.\n"
        + DOWNLOAD_INSTRUCTIONS
    )


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        return []
    files: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(images_dir.glob(f"*{ext}"))
    return sorted(files)


def stem_to_xml(annotations_dir: Path, stem: str) -> Path | None:
    """Locate a VOC XML for an image stem, if present."""
    if not annotations_dir.is_dir():
        return None
    for candidate in (
        annotations_dir / f"{stem}.xml",
        annotations_dir / "xmls" / f"{stem}.xml",
    ):
        if candidate.is_file():
            return candidate
    return None
