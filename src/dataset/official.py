"""Official RDD2022 pipeline: config, raw scan, audit, VOC→YOLO conversion, grouped
splits, validation, Ultralytics runtime YAML and dataset fingerprints.

Raw files (``data/raw/RDD2022``) are only ever read. Processed images are hardlinks
(fallback symlink → copy) under ``data/processed/images/<country>/`` with YOLO labels in
``data/processed/labels/<country>/``. The synthetic fixture is never used here unless a
caller explicitly asks for ``dataset="fixture"``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.constants import (
    CLASS_NAMES,
    CLASS_TO_ID,
    DATASET_VERSION,
    DEFAULT_SEED,
    EXCLUDE_COUNTRIES,
    GROUP_BLOCK_SIZE,
    KNOWN_COUNTRIES,
    RDD2022_ROOT_ENV,
)
from src.dataset.convert import voc_box_to_yolo
from src.dataset.voc import parse_voc_xml
from src.paths import (
    default_rdd2022_config,
    default_splits_dir,
    discover_countries,
    ensure_legacy_fixture_data_yaml,
    legacy_fixture_processed_dir,
    legacy_fixture_splits_dir,
    project_root,
    resolve_dataset_root,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
OFFICIAL_DATASET = "official_rdd2022"
SPLIT_COLUMNS = [
    "image_id", "country", "split", "image_path", "label_path", "source_image",
    "group_id", "n_objects", "n_D00", "n_D10", "n_D20", "n_D40", "is_negative",
    "width", "height", "md5", "phash",
]


# --------------------------------------------------------------------------- config


def repo_path(rel: str | Path) -> Path:
    p = Path(rel).expanduser()
    return p if p.is_absolute() else project_root() / p


def load_dataset_config(config_path: str | Path | None = None) -> dict:
    """Load the tracked official dataset config (``configs/rdd2022.yaml``)."""
    path = Path(config_path) if config_path else default_rdd2022_config()
    if not path.is_absolute():
        path = path if path.is_file() else repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if cfg.get("dataset") != OFFICIAL_DATASET:
        raise ValueError(
            f"{path} is not an official RDD2022 dataset config (dataset={cfg.get('dataset')!r})"
        )
    cfg["_config_path"] = str(path.resolve())
    return cfg


def is_project_dataset_config(path: str | Path) -> bool:
    try:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return False
    return cfg.get("dataset") == OFFICIAL_DATASET and "raw_root" in cfg


def resolve_raw_root(cfg: dict, cli_path: str | Path | None = None, *, must_exist: bool = True) -> Path:
    """CLI path > ``$RDD2022_ROOT`` > ``raw_root`` from the config."""
    env = os.environ.get(cfg.get("root_env", RDD2022_ROOT_ENV))
    cand = cli_path or env or repo_path(cfg.get("raw_root", "data/raw/RDD2022"))
    return resolve_dataset_root(cand, "official", must_exist=must_exist)


def official_paths(cfg: dict, processed_root: str | Path | None = None) -> dict[str, Path]:
    return {
        "processed": repo_path(processed_root) if processed_root else repo_path(cfg.get("processed_root", "data/processed")),
        "splits": repo_path(cfg.get("splits_dir", "data/splits")),
        "results": repo_path(cfg.get("results_dir", "results/official_rdd2022")),
        "ultralytics_yaml": repo_path(cfg.get("ultralytics_yaml", "data/processed/ultralytics/rdd2022_official.yaml")),
    }


def excluded_folders(cfg: dict | None) -> set[str]:
    if cfg and cfg.get("excluded_folders") is not None:
        return set(cfg["excluded_folders"])
    return set(EXCLUDE_COUNTRIES)


# --------------------------------------------------------------------------- helpers


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _list_files(d: Path, suffixes: set[str]) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith("."))


def _rel(p: Path, base: Path) -> str:
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or (list(rows[0].keys()) if rows else [])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2)))


# --------------------------------------------------------------------------- raw scan


def scan_raw(raw_root: Path) -> list[dict]:
    """List every train image/XML pair and every (unlabeled) official test image."""
    items: list[dict] = []
    for country in discover_countries(raw_root):
        base = raw_root / country
        img_dir = base / "train" / "images"
        ann_dir = base / "train" / "annotations" / "xmls"
        if not ann_dir.is_dir():
            ann_dir = base / "train" / "annotations"
        imgs = {p.stem: p for p in _list_files(img_dir, IMAGE_SUFFIXES)}
        xmls = {p.stem: p for p in _list_files(ann_dir, {".xml"})}
        for stem in sorted(set(imgs) | set(xmls)):
            items.append({"country": country, "subset": "train", "stem": stem,
                          "image": imgs.get(stem), "xml": xmls.get(stem)})
        for p in _list_files(base / "test" / "images", IMAGE_SUFFIXES):
            items.append({"country": country, "subset": "test", "stem": p.stem, "image": p, "xml": None})
    return items


def looks_processed(root: Path) -> bool:
    return (root / "images").is_dir() and (root / "labels").is_dir() and not discover_countries(root)


# --------------------------------------------------------------------------- image index


def inspect_image(path_str: str, full_decode: bool = True) -> dict:
    """Size, readability (full decode), md5 and 64-bit pHash for one image."""
    from PIL import Image

    import imagehash

    out = {"width": 0, "height": 0, "readable": False, "error": "", "md5": "", "phash": ""}
    try:
        data = Path(path_str).read_bytes()
        out["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            out["width"], out["height"] = im.size
            if full_decode:
                im.load()
            else:
                im.draft("L", (256, 256))
            g = im.convert("L")
            g.thumbnail((256, 256))
            out["phash"] = str(imagehash.phash(g))
        out["readable"] = True
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def verify_image(path_str: str) -> str:
    """Cheap structural check (PIL verify). Returns '' when readable."""
    from PIL import Image

    try:
        with Image.open(path_str) as im:
            im.verify()
        return ""
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"[:300]


INDEX_COLUMNS = ["rel_path", "size_bytes", "mtime_ns", "width", "height", "readable", "error", "md5", "phash"]


def build_image_index(
    paths: list[Path], base: Path, cache_csv: Path | None, *, workers: int, full_decode: bool = True
) -> dict[str, dict]:
    """Hash/verify images in parallel; reuse rows from ``cache_csv`` when size+mtime match."""
    cache: dict[str, dict] = {}
    if cache_csv and cache_csv.is_file():
        for r in read_csv(cache_csv):
            cache[r["rel_path"]] = r
    index: dict[str, dict] = {}
    todo: list[tuple[str, Path, os.stat_result]] = []
    for p in paths:
        rel = _rel(p, base)
        st = p.stat()
        c = cache.get(rel)
        if c and int(c["size_bytes"]) == st.st_size and int(c["mtime_ns"]) == st.st_mtime_ns:
            c["readable"] = str(c["readable"]) in {"True", "true", "1"}
            c["width"], c["height"] = int(c["width"]), int(c["height"])
            index[rel] = c
        else:
            todo.append((rel, p, st))
    if todo:
        print(f"  hashing/decoding {len(todo)} images with {workers} workers "
              f"({len(index)} reused from cache)", flush=True)
        args = [str(p) for _, p, _ in todo]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(inspect_image, args, [full_decode] * len(args), chunksize=16))
        else:
            results = [inspect_image(a, full_decode) for a in args]
        for (rel, _, st), res in zip(todo, results):
            index[rel] = {"rel_path": rel, "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, **res}
    if cache_csv:
        write_csv(cache_csv, [index[k] for k in sorted(index)], INDEX_COLUMNS)
    return index


# --------------------------------------------------------------------------- duplicates


def phash_pairs(hashes: list[str], max_hamming: int) -> list[tuple[int, int, int]]:
    """All (i, j, distance) with i < j and pHash Hamming distance <= ``max_hamming``."""
    import numpy as np

    valid = [i for i, h in enumerate(hashes) if h]
    if len(valid) < 2:
        return []
    if max_hamming <= 0:
        buckets: dict[str, list[int]] = defaultdict(list)
        for i in valid:
            buckets[hashes[i]].append(i)
        return [(a, b, 0) for members in buckets.values() if len(members) > 1
                for k, a in enumerate(members) for b in members[k + 1:]]
    vals = np.array([int(hashes[i], 16) for i in valid], dtype=np.uint64)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    out: list[tuple[int, int, int]] = []
    n = len(vals)
    for start in range(0, n, 256):
        block = vals[start:start + 256, None] ^ vals[None, :]
        dist = table[block.view(np.uint8)].reshape(block.shape + (8,)).sum(-1, dtype=np.uint16)
        ii, jj = np.nonzero(dist <= max_hamming)
        for a, b in zip(ii.tolist(), jj.tolist()):
            ga = start + a
            if ga < b:
                out.append((valid[ga], valid[b], int(dist[a, b])))
    return out


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def duplicate_clusters(records: list[dict], max_hamming: int) -> list[list[int]]:
    """Index clusters (size > 1) of exact-md5 or near-identical-pHash images.

    md5 matches are global; pHash matches are searched within each country (and
    globally for distance 0).
    """
    uf = _UnionFind()
    by_md5: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        if r.get("md5"):
            by_md5[r["md5"]].append(i)
    for members in by_md5.values():
        for m in members[1:]:
            uf.union(members[0], m)
    hashes = [r.get("phash") or "" for r in records]
    for a, b, _ in phash_pairs(hashes, 0):
        uf.union(a, b)
    if max_hamming > 0:
        by_country: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            by_country[r["country"]].append(i)
        for idxs in by_country.values():
            for a, b, _ in phash_pairs([hashes[i] for i in idxs], max_hamming):
                uf.union(idxs[a], idxs[b])
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(len(records)):
        comps[uf.find(i)].append(i)
    return [sorted(m) for m in comps.values() if len(m) > 1]


# --------------------------------------------------------------------------- audit


def _box_issue(box, w: int, h: int) -> str:
    if box.xmax <= box.xmin or box.ymax <= box.ymin:
        return "invalid_box_order"
    if w <= 0 or h <= 0:
        return "unknown_image_size"
    if box.xmax <= 0 or box.ymax <= 0 or box.xmin >= w or box.ymin >= h:
        return "box_outside_image"
    if box.xmin < 0 or box.ymin < 0 or box.xmax > w or box.ymax > h:
        return "box_partially_out_of_bounds"
    return ""


def _quantiles(vals: list[float]) -> dict:
    import numpy as np

    if not vals:
        return {"mean": "", "p05": "", "p50": "", "p95": "", "min": "", "max": ""}
    a = np.asarray(vals, dtype=float)
    return {"mean": round(float(a.mean()), 3), "p05": round(float(np.quantile(a, 0.05)), 3),
            "p50": round(float(np.quantile(a, 0.5)), 3), "p95": round(float(np.quantile(a, 0.95)), 3),
            "min": round(float(a.min()), 3), "max": round(float(a.max()), 3)}


def run_official_audit(
    raw_root: Path,
    out_dir: Path,
    *,
    exclude: set[str],
    workers: int,
    check_images: bool = True,
    max_hamming: int = 0,
    info_hamming: int = 4,
) -> dict:
    """Audit raw VOC data. Writes summary/class/box/issue/duplicate tables to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = scan_raw(raw_root)
    train = [it for it in items if it["subset"] == "train"]
    test_imgs = [it for it in items if it["subset"] == "test"]

    issues: list[dict] = []
    per_country: dict[str, Counter] = defaultdict(Counter)
    label_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    box_dims: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    index: dict[str, dict] = {}
    if check_images:
        img_paths = [it["image"] for it in train if it["image"] is not None]
        index = build_image_index(img_paths, raw_root, out_dir / "image_index.csv", workers=workers)

    for it in test_imgs:
        per_country[it["country"]]["official_test_images_unlabeled"] += 1

    for it in train:
        c, stem = it["country"], it["stem"]
        pc = per_country[c]
        if it["image"] is None:
            pc["xml_without_image"] += 1
            issues.append({"country": c, "stem": stem, "issue": "xml_without_image", "detail": _rel(it["xml"], raw_root)})
            continue
        pc["train_images"] += 1
        rel_img = _rel(it["image"], raw_root)
        info = index.get(rel_img)
        if info is not None and not info["readable"]:
            pc["corrupt_images"] += 1
            issues.append({"country": c, "stem": stem, "issue": "corrupt_image", "detail": info["error"]})
        if it["xml"] is None:
            pc["image_without_xml"] += 1
            issues.append({"country": c, "stem": stem, "issue": "image_without_xml", "detail": rel_img})
            continue
        pc["train_xmls"] += 1
        ann = parse_voc_xml(it["xml"])
        if ann.parse_error:
            pc["malformed_xml"] += 1
            issues.append({"country": c, "stem": stem, "issue": "malformed_xml", "detail": ann.parse_error})
            continue
        for name in ann.malformed_objects:
            pc["malformed_objects"] += 1
            issues.append({"country": c, "stem": stem, "issue": "malformed_object", "detail": name})
        w, h = ann.width, ann.height
        if info is not None and info["readable"]:
            if (w, h) != (info["width"], info["height"]):
                pc["xml_size_mismatch"] += 1
                issues.append({"country": c, "stem": stem, "issue": "xml_size_mismatch",
                               "detail": f"xml={w}x{h} image={info['width']}x{info['height']}"})
            w, h = info["width"], info["height"]
        n_supported = 0
        for box in ann.boxes:
            label_counts[(c, box.label)]["boxes"] += 1
            supported = box.label in CLASS_TO_ID
            if not supported:
                pc["unsupported_boxes"] += 1
                continue
            problem = _box_issue(box, w, h)
            if problem:
                pc[problem] += 1
                issues.append({"country": c, "stem": stem, "issue": problem,
                               "detail": f"{box.label} {box.xmin},{box.ymin},{box.xmax},{box.ymax} img={w}x{h}"})
                if problem in {"invalid_box_order", "box_outside_image", "unknown_image_size"}:
                    pc["invalid_boxes"] += 1
                    continue
            n_supported += 1
            bw = min(box.xmax, w) - max(box.xmin, 0)
            bh = min(box.ymax, h) - max(box.ymin, 0)
            if c not in exclude:
                box_dims[box.label]["width_px"].append(bw)
                box_dims[box.label]["height_px"].append(bh)
                box_dims[box.label]["rel_area"].append(bw * bh / float(w * h))
                box_dims[box.label]["sqrt_area_px"].append((bw * bh) ** 0.5)
        for lbl in {b.label for b in ann.boxes}:
            label_counts[(c, lbl)]["images"] += 1
        if n_supported == 0:
            pc["negatives_no_supported_boxes"] += 1

    # duplicates over the modeling set (non-excluded labeled train images)
    exact_rows: list[dict] = []
    near_rows: list[dict] = []
    dup_summary: dict = {"checked": bool(check_images)}
    if check_images:
        modeling = [dict(index[_rel(it["image"], raw_root)], country=it["country"])
                    for it in train if it["image"] is not None and it["xml"] is not None
                    and it["country"] not in exclude and _rel(it["image"], raw_root) in index]
        by_md5: dict[str, list[dict]] = defaultdict(list)
        for r in modeling:
            if r["md5"]:
                by_md5[r["md5"]].append(r)
        for md5, members in by_md5.items():
            if len(members) > 1:
                for r in members:
                    exact_rows.append({"md5": md5, "rel_path": r["rel_path"], "country": r["country"],
                                       "cluster_size": len(members)})
        hashes = [r["phash"] for r in modeling]
        n_info_pairs = 0
        by_country: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(modeling):
            by_country[r["country"]].append(i)
        for country, idxs in sorted(by_country.items()):
            for a, b, d in phash_pairs([hashes[i] for i in idxs], max(info_hamming, max_hamming)):
                ra, rb = modeling[idxs[a]], modeling[idxs[b]]
                if d <= info_hamming:
                    n_info_pairs += 1
                if d <= max_hamming:
                    near_rows.append({"country": country, "image_a": ra["rel_path"], "image_b": rb["rel_path"],
                                      "hamming": d, "same_md5": ra["md5"] == rb["md5"]})
        cross = phash_pairs(hashes, 0)
        n_cross_country = sum(1 for a, b, _ in cross if modeling[a]["country"] != modeling[b]["country"])
        clusters = duplicate_clusters(modeling, max_hamming)
        dup_summary.update({
            "modeling_images_hashed": len(modeling),
            "exact_duplicate_clusters": sum(1 for m in by_md5.values() if len(m) > 1),
            "exact_duplicate_images": len(exact_rows),
            "near_duplicate_definition": f"pHash(8x8) Hamming distance <= {max_hamming}",
            "near_duplicate_pairs_within_country": len(near_rows),
            "phash_identical_pairs_cross_country": n_cross_country,
            f"informational_pairs_hamming_le_{info_hamming}_within_country": n_info_pairs,
            "duplicate_clusters_enforced_in_splits": len(clusters),
            "images_in_duplicate_clusters": sum(len(c) for c in clusters),
        })
    write_csv(out_dir / "exact_duplicates.csv", exact_rows, ["md5", "rel_path", "country", "cluster_size"])
    write_csv(out_dir / "near_duplicates.csv", near_rows, ["country", "image_a", "image_b", "hamming", "same_md5"])

    # tables
    countries = sorted(per_country)
    class_rows = []
    for (c, lbl), cnt in sorted(label_counts.items()):
        class_rows.append({"country": c, "label": lbl, "class_id": CLASS_TO_ID.get(lbl, ""),
                           "supported": lbl in CLASS_TO_ID, "excluded_country": c in exclude,
                           "n_boxes": cnt["boxes"], "n_images": cnt["images"]})
    write_csv(out_dir / "class_distribution.csv", class_rows)

    box_rows = []
    for lbl in CLASS_NAMES:
        dims = box_dims.get(lbl, {})
        row = {"class": lbl, "class_id": CLASS_TO_ID[lbl], "n_boxes": len(dims.get("width_px", []))}
        for key in ("width_px", "height_px", "sqrt_area_px", "rel_area"):
            for stat, v in _quantiles(dims.get(key, [])).items():
                row[f"{key}_{stat}"] = v
        sq = dims.get("sqrt_area_px", [])
        row["n_small_lt32px"] = sum(1 for v in sq if v < 32)
        row["n_medium_32_96px"] = sum(1 for v in sq if 32 <= v < 96)
        row["n_large_ge96px"] = sum(1 for v in sq if v >= 96)
        box_rows.append(row)
    write_csv(out_dir / "box_statistics.csv", box_rows)

    country_rows = [{"country": c, "excluded": c in exclude, **dict(per_country[c])} for c in countries]
    keys = sorted({k for r in country_rows for k in r if k not in {"country", "excluded"}})
    write_csv(out_dir / "per_country_summary.csv", country_rows, ["country", "excluded", *keys])
    write_csv(out_dir / "issues.csv", issues, ["country", "stem", "issue", "detail"])

    totals: Counter = Counter()
    modeling_totals: Counter = Counter()
    for c in countries:
        totals.update(per_country[c])
        if c not in exclude:
            modeling_totals.update(per_country[c])
    supported_by_class = {lbl: sum(r["n_boxes"] for r in class_rows if r["label"] == lbl and not r["excluded_country"])
                          for lbl in CLASS_NAMES}
    unsupported = Counter()
    for r in class_rows:
        if not r["supported"] and not r["excluded_country"]:
            unsupported[r["label"]] += r["n_boxes"]
    summary = {
        "generated_at": _now(),
        "data_kind": "raw_official_voc",
        "dataset_version": DATASET_VERSION,
        "data_root": str(raw_root),
        "countries": countries,
        "excluded_countries": sorted(exclude),
        "images_decoded": bool(check_images),
        "totals_all_countries": dict(totals),
        "totals_modeling_set": dict(modeling_totals),
        "class_box_counts_modeling_set": supported_by_class,
        "unsupported_label_counts_modeling_set": dict(sorted(unsupported.items())),
        "per_country": {c: dict(per_country[c]) for c in countries},
        "duplicates": dup_summary,
        "n_issue_rows": len(issues),
        "issue_counts": dict(Counter(i["issue"] for i in issues)),
        "outputs": {n: str(out_dir / n) for n in (
            "audit_summary.json", "class_distribution.csv", "box_statistics.csv", "per_country_summary.csv",
            "issues.csv", "exact_duplicates.csv", "near_duplicates.csv", "image_index.csv")},
    }
    write_json(out_dir / "audit_summary.json", summary)
    return summary


def audit_processed(root: Path, out_dir: Path) -> dict:
    """Lightweight audit when ``--data-root`` points at processed YOLO data."""
    counts: Counter = Counter()
    per_country: dict[str, Counter] = defaultdict(Counter)
    for cdir in sorted(p for p in (root / "images").iterdir() if p.is_dir()):
        for img in _list_files(cdir, IMAGE_SUFFIXES):
            lbl = root / "labels" / cdir.name / f"{img.stem}.txt"
            pc = per_country[cdir.name]
            pc["images"] += 1
            if not lbl.is_file():
                pc["missing_labels"] += 1
                continue
            lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
            pc["objects"] += len(lines)
            pc["negatives"] += int(not lines)
            for ln in lines:
                counts[CLASS_NAMES[int(ln.split()[0])] if ln.split()[0].isdigit() and int(ln.split()[0]) < 4 else "invalid"] += 1
    summary = {"generated_at": _now(), "data_kind": "processed_yolo", "data_root": str(root),
               "per_country": {c: dict(v) for c, v in per_country.items()}, "class_counts": dict(counts)}
    write_json(out_dir / "audit_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- conversion


def _link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if os.path.samefile(src, dst):
                return "existing"
        except OSError:
            pass
        dst.unlink()
    order = {"hardlink": ("hardlink", "symlink", "copy"), "symlink": ("symlink", "copy"), "copy": ("copy",)}[mode]
    for m in order:
        try:
            if m == "hardlink":
                os.link(src, dst)
            elif m == "symlink":
                os.symlink(src.resolve(), dst)
            else:
                shutil.copy2(src, dst)
            return m
        except OSError:
            continue
    raise OSError(f"could not materialize {src} -> {dst}")


def convert_annotation(boxes, w: int, h: int) -> tuple[list[str], Counter, list[str]]:
    """Return (yolo_lines, unsupported_label_counts, rejected_box_reasons)."""
    lines: list[str] = []
    unsupported: Counter = Counter()
    rejected: list[str] = []
    seen: set[tuple] = set()
    for box in boxes:
        if box.label not in CLASS_TO_ID:
            unsupported[box.label] += 1
            continue
        if box.xmax <= box.xmin or box.ymax <= box.ymin:
            rejected.append("invalid_box_order")
            continue
        yolo = voc_box_to_yolo(box.xmin, box.ymin, box.xmax, box.ymax, w, h)
        if yolo is None:
            rejected.append("box_outside_image" if w > 0 and h > 0 else "unknown_image_size")
            continue
        cid = CLASS_TO_ID[box.label]
        key = (cid, *(round(v, 6) for v in yolo))
        if key in seen:
            rejected.append("duplicate_box")
            continue
        seen.add(key)
        cx, cy, nw, nh = yolo
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines, unsupported, rejected


MANIFEST_COLUMNS = [
    "image_id", "country", "image_path", "label_path", "source_image", "source_xml",
    "width", "height", "n_objects", "n_D00", "n_D10", "n_D20", "n_D40", "is_negative",
    "n_unsupported", "n_rejected_boxes", "link_mode",
]


def convert_official(
    raw_root: Path,
    processed_root: Path,
    results_dir: Path,
    *,
    exclude: set[str],
    link_mode: str = "hardlink",
    countries: list[str] | None = None,
    max_images_per_country: int | None = None,
) -> dict:
    """Convert labeled train data to YOLO. Never writes inside ``raw_root``."""
    from PIL import Image

    raw_root, processed_root = raw_root.resolve(), processed_root.resolve()
    if processed_root == raw_root or raw_root in processed_root.parents:
        raise ValueError("processed root must not be inside the raw dataset root")
    items = [it for it in scan_raw(raw_root) if it["subset"] == "train"]
    wanted = set(countries) if countries else None

    manifest: list[dict] = []
    unsupported_rows: Counter = Counter()
    issue_rows: list[dict] = []
    counters: Counter = Counter()
    per_country_n: Counter = Counter()
    seen_country: set[str] = set()

    for it in items:
        c = it["country"]
        if c in exclude:
            counters[f"skipped_excluded_{c}"] += 1
            continue
        if wanted and c not in wanted:
            continue
        if max_images_per_country and per_country_n[c] >= max_images_per_country:
            continue
        seen_country.add(c)
        if it["image"] is None:
            counters["xml_without_image"] += 1
            issue_rows.append({"country": c, "stem": it["stem"], "issue": "xml_without_image", "detail": ""})
            continue
        if it["xml"] is None:
            counters["unlabeled_image_excluded"] += 1
            issue_rows.append({"country": c, "stem": it["stem"], "issue": "image_without_xml_excluded", "detail": ""})
            continue
        ann = parse_voc_xml(it["xml"])
        if ann.parse_error:
            counters["malformed_xml_excluded"] += 1
            issue_rows.append({"country": c, "stem": it["stem"], "issue": "malformed_xml_excluded", "detail": ann.parse_error})
            continue
        try:
            with Image.open(it["image"]) as im:
                w, h = im.size
        except Exception as exc:  # noqa: BLE001
            counters["unreadable_image_excluded"] += 1
            issue_rows.append({"country": c, "stem": it["stem"], "issue": "unreadable_image_excluded", "detail": str(exc)[:200]})
            continue
        if (ann.width, ann.height) != (w, h):
            counters["xml_size_mismatch_used_image_size"] += 1
        lines, unsupported, rejected = convert_annotation(ann.boxes, w, h)
        counters["malformed_objects_skipped"] += len(ann.malformed_objects)
        for lbl, n in unsupported.items():
            unsupported_rows[(c, lbl)] += n
        for reason in rejected:
            counters[f"rejected_{reason}"] += 1
            issue_rows.append({"country": c, "stem": it["stem"], "issue": f"rejected_{reason}", "detail": ""})

        image_rel = f"images/{c}/{it['image'].name}"
        label_rel = f"labels/{c}/{it['stem']}.txt"
        lbl_path = processed_root / label_rel
        lbl_path.parent.mkdir(parents=True, exist_ok=True)
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        used = _link_or_copy(it["image"], processed_root / image_rel, link_mode)
        counters[f"image_{used}"] += 1
        per_class = Counter(int(ln.split()[0]) for ln in lines)
        manifest.append({
            "image_id": f"{c}/{it['stem']}", "country": c, "image_path": image_rel, "label_path": label_rel,
            "source_image": _rel(it["image"], raw_root), "source_xml": _rel(it["xml"], raw_root),
            "width": w, "height": h, "n_objects": len(lines),
            **{f"n_{n}": per_class.get(i, 0) for i, n in enumerate(CLASS_NAMES)},
            "is_negative": int(not lines), "n_unsupported": sum(unsupported.values()),
            "n_rejected_boxes": len(rejected), "link_mode": used,
        })
        per_country_n[c] += 1

    # remove stale generated files for the countries processed in this run
    keep_imgs = {m["image_path"] for m in manifest}
    keep_lbls = {m["label_path"] for m in manifest}
    stale = 0
    for c in seen_country:
        for sub, keep in (("images", keep_imgs), ("labels", keep_lbls)):
            d = processed_root / sub / c
            if d.is_dir():
                for p in d.iterdir():
                    if p.is_file() and f"{sub}/{c}/{p.name}" not in keep:
                        p.unlink()
                        stale += 1
    for cache in (processed_root / "labels").glob("*.cache"):
        cache.unlink()

    manifest.sort(key=lambda r: r["image_id"])
    write_csv(processed_root / "manifest.csv", manifest, MANIFEST_COLUMNS)
    results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(results_dir / "conversion_manifest.csv", manifest, MANIFEST_COLUMNS)
    unsup = [{"country": c, "label": lbl, "count": n} for (c, lbl), n in sorted(unsupported_rows.items())]
    write_csv(results_dir / "unsupported_labels.csv", unsup, ["country", "label", "count"])
    write_csv(results_dir / "conversion_issues.csv", issue_rows, ["country", "stem", "issue", "detail"])

    class_counts = {n: sum(int(m[f"n_{n}"]) for m in manifest) for n in CLASS_NAMES}
    by_country = defaultdict(lambda: {"images": 0, "negatives": 0, "objects": 0})
    for m in manifest:
        bc = by_country[m["country"]]
        bc["images"] += 1
        bc["negatives"] += int(m["is_negative"])
        bc["objects"] += int(m["n_objects"])
    unsupported_total = Counter()
    for (_, lbl), n in unsupported_rows.items():
        unsupported_total[lbl] += n
    summary = {
        "generated_at": _now(),
        "dataset": OFFICIAL_DATASET,
        "dataset_version": DATASET_VERSION,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "class_mapping": CLASS_TO_ID,
        "excluded_countries": sorted(exclude),
        "link_mode_requested": link_mode,
        "n_images_converted": len(manifest),
        "n_label_files_written": len(manifest),
        "n_negatives_empty_labels": sum(int(m["is_negative"]) for m in manifest),
        "n_objects": sum(int(m["n_objects"]) for m in manifest),
        "class_counts": class_counts,
        "per_country": dict(sorted(by_country.items())),
        "unsupported_labels_dropped": dict(sorted(unsupported_total.items())),
        "counters": dict(sorted(counters.items())),
        "stale_generated_files_removed": stale,
        "manifest_csv": str(processed_root / "manifest.csv"),
        "manifest_sha256": sha256_file(processed_root / "manifest.csv"),
        "originals_modified": False,
    }
    write_json(results_dir / "conversion_manifest.json", summary)
    return summary


# --------------------------------------------------------------------------- splits


def _assign_country(comps: list[dict], ratios: tuple[float, float, float], rng, n_candidates: int):
    """Pick, among seeded random orderings, the 3-way cut closest to the ratios for
    images, negatives and each class's box count."""
    import numpy as np

    feats = np.array([[c["n_images"], c["n_neg"], *c["classes"]] for c in comps], dtype=float)
    totals = feats.sum(0)
    target = np.array(ratios)
    best = None
    n = len(comps)
    for _ in range(max(1, n_candidates)):
        order = rng.permutation(n)
        cum = np.cumsum(feats[order, 0])
        total_imgs = cum[-1]
        cut1 = int(np.searchsorted(cum, ratios[0] * total_imgs, side="left")) + 1
        cut2 = int(np.searchsorted(cum, (ratios[0] + ratios[1]) * total_imgs, side="left")) + 1
        if n >= 3:
            cut1 = min(max(cut1, 1), n - 2)
            cut2 = min(max(cut2, cut1 + 1), n - 1)
        assign = np.empty(n, dtype=int)
        assign[order[:cut1]] = 0
        assign[order[cut1:cut2]] = 1
        assign[order[cut2:]] = 2
        score = 0.0
        for s in range(3):
            share = feats[assign == s].sum(0) / np.where(totals > 0, totals, 1)
            weight = np.where(totals > 0, 1.0, 0.0)
            weight[0] = 3.0
            score += float((np.abs(share - target[s]) * weight).sum())
        if best is None or score < best[0]:
            best = (score, assign.copy())
    return best


def backup_legacy_fixture_splits(splits_dir: Path) -> list[str]:
    """Copy fixture split files to ``data/splits/legacy_fixture/`` once, before overwrite."""
    dest = legacy_fixture_splits_dir()
    summary = splits_dir / "split_summary.json"
    copied: list[str] = []
    is_official = False
    if summary.is_file():
        try:
            is_official = json.loads(summary.read_text()).get("dataset") == OFFICIAL_DATASET
        except json.JSONDecodeError:
            pass
    if is_official:
        return copied
    dest.mkdir(parents=True, exist_ok=True)
    sources = [splits_dir / f for f in ("train.csv", "val.csv", "test.csv", "split_summary.json")]
    sources += [project_root() / "data" / f"{s}_split.csv" for s in SPLITS]
    for src in sources:
        if src.is_file() and not (dest / src.name).exists():
            shutil.copy2(src, dest / src.name)
            copied.append(str(dest / src.name))
    if copied:
        (dest / "README.md").write_text(
            "# LEGACY synthetic-fixture splits\n\nCopied here before the official RDD2022 splits "
            "replaced `data/splits/*.csv`. They index `data/processed/RDD2022_yolo` (synthetic "
            "fixture) and are kept only to reproduce legacy Week 1–7 artifacts.\n",
            encoding="utf-8",
        )
    return copied


def _near_pairs_within_country(rows: list[dict], hamming: int) -> list[tuple[int, int, int]]:
    by_country: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_country[r["country"]].append(i)
    out = []
    for idxs in by_country.values():
        for a, b, d in phash_pairs([rows[i].get("phash") or "" for i in idxs], hamming):
            out.append((idxs[a], idxs[b], d))
    return out


def purge_cross_split_near_duplicates(rows: list[dict], hamming: int) -> list[dict]:
    """Drop the val/test member of every within-country pHash pair (distance <= ``hamming``)
    that spans two splits. Removal is non-transitive, so groups and balance are kept.
    Returns the purged rows (with the neighbour that triggered the purge)."""
    if hamming <= 0:
        return []
    rank = {s: k for k, s in enumerate(SPLITS)}
    nbrs: dict[int, set[int]] = defaultdict(set)
    for a, b, _ in _near_pairs_within_country(rows, hamming):
        nbrs[a].add(b)
        nbrs[b].add(a)
    dropped: dict[int, int] = {}
    for i in sorted(nbrs, key=lambda i: (-rank[rows[i]["split"]], rows[i]["image_id"])):
        if rows[i]["split"] == "train":
            continue
        trig = [j for j in sorted(nbrs[i]) if j not in dropped and rank[rows[j]["split"]] < rank[rows[i]["split"]]]
        if trig:
            dropped[i] = trig[0]
    return [{**rows[i], "purged_because_of": rows[j]["image_id"], "neighbour_split": rows[j]["split"]}
            for i, j in sorted(dropped.items(), key=lambda kv: rows[kv[0]]["image_id"])]


def leakage_report(rows: list[dict], max_hamming: int = 0, purge_hamming: int = 0) -> dict:
    """Check that no image id, group, exact duplicate or near-duplicate spans two splits."""
    split_of_id: dict[str, set[str]] = defaultdict(set)
    split_of_group: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        split_of_id[r["image_id"]].add(r["split"])
        split_of_group[r["group_id"]].add(r["split"])
    clusters = duplicate_clusters(rows, max_hamming)
    crossing_clusters = [c for c in clusters if len({rows[i]["split"] for i in c}) > 1]
    md5_cross = 0
    by_md5: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("md5"):
            by_md5[r["md5"]].add(r["split"])
    md5_cross = sum(1 for s in by_md5.values() if len(s) > 1)
    report = {
        "n_images": len(rows),
        "image_ids_in_multiple_splits": sum(1 for s in split_of_id.values() if len(s) > 1),
        "groups_in_multiple_splits": sum(1 for s in split_of_group.values() if len(s) > 1),
        "near_duplicate_definition": f"exact md5 or pHash(8x8) Hamming distance <= {max_hamming}",
        "duplicate_clusters": len(clusters),
        "images_in_duplicate_clusters": sum(len(c) for c in clusters),
        "duplicate_clusters_crossing_splits": len(crossing_clusters),
        "exact_md5_values_crossing_splits": md5_cross,
        "examples_crossing": [[rows[i]["image_id"] for i in c][:5] for c in crossing_clusters[:10]],
    }
    near_cross = 0
    info_cross: Counter = Counter()
    for a, b, d in _near_pairs_within_country(rows, max(purge_hamming, 6)):
        if rows[a]["split"] != rows[b]["split"]:
            info_cross[d] += 1
            if d <= purge_hamming:
                near_cross += 1
    report[f"near_duplicate_pairs_le_{purge_hamming}_crossing_splits"] = near_cross
    report["informational_crossing_pairs_by_phash_distance"] = {str(k): v for k, v in sorted(info_cross.items())}
    report["passed"] = (report["image_ids_in_multiple_splits"] == 0 and report["groups_in_multiple_splits"] == 0
                        and report["duplicate_clusters_crossing_splits"] == 0 and md5_cross == 0
                        and near_cross == 0)
    return report


def create_official_splits(
    processed_root: Path,
    splits_dir: Path,
    results_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    block_size: int = GROUP_BLOCK_SIZE,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    max_hamming: int = 0,
    purge_hamming: int = 0,
    n_candidates: int = 200,
    image_index_csv: Path | None = None,
    workers: int = 4,
) -> dict:
    """Deterministic grouped split of the converted official data."""
    import numpy as np

    manifest_path = processed_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{manifest_path} missing — run scripts/convert_voc_to_yolo.py first")
    rows = read_csv(manifest_path)
    if not rows:
        raise ValueError("conversion manifest is empty")

    # duplicate hashes: reuse the audit index (keyed by raw-relative path), else hash now
    cached: dict[str, dict] = {}
    if image_index_csv and image_index_csv.is_file():
        cached = {r["rel_path"]: r for r in read_csv(image_index_csv)}
    missing = [r for r in rows if r["source_image"] not in cached]
    if missing:
        idx = build_image_index([processed_root / r["image_path"] for r in missing], processed_root,
                                results_dir / "split_image_index.csv", workers=workers, full_decode=False)
        for r in missing:
            cached[r["source_image"]] = idx[r["image_path"]]
    for r in rows:
        info = cached[r["source_image"]]
        r["md5"], r["phash"] = info["md5"], info["phash"]

    # block-of-N groups per country (sorted filenames)
    by_country: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_country[r["country"]].append(i)
    for c, idxs in by_country.items():
        idxs.sort(key=lambda i: Path(rows[i]["image_path"]).name)
        for k, i in enumerate(idxs):
            rows[i]["group_id"] = f"{c}_g{k // block_size:04d}"

    # merge groups linked by duplicates
    uf = _UnionFind()
    for r in rows:
        uf.find(r["group_id"])
    clusters = duplicate_clusters(rows, max_hamming)
    for cl in clusters:
        for i in cl[1:]:
            uf.union(rows[cl[0]]["group_id"], rows[i]["group_id"])
    comps: dict[str, dict] = {}
    for i, r in enumerate(rows):
        key = uf.find(r["group_id"])
        comp = comps.setdefault(key, {"key": key, "members": [], "countries": Counter()})
        comp["members"].append(i)
        comp["countries"][r["country"]] += 1
    merged_groups = sum(1 for g in {r["group_id"] for r in rows} if uf.find(g) != g)

    comps_by_country: dict[str, list[dict]] = defaultdict(list)
    for comp in comps.values():
        idxs = comp["members"]
        comp["n_images"] = len(idxs)
        comp["n_neg"] = sum(int(rows[i]["is_negative"]) for i in idxs)
        comp["classes"] = [sum(int(rows[i][f"n_{n}"]) for i in idxs) for n in CLASS_NAMES]
        comps_by_country[comp["countries"].most_common(1)[0][0]].append(comp)

    split_names = list(SPLITS)
    selection: dict[str, dict] = {}
    for country in sorted(comps_by_country):
        cl = sorted(comps_by_country[country], key=lambda c: c["key"])
        rng = np.random.default_rng([seed, zlib.crc32(country.encode())])
        score, assign = _assign_country(cl, ratios, rng, n_candidates)
        for comp, s in zip(cl, assign):
            for i in comp["members"]:
                rows[i]["split"] = split_names[int(s)]
        selection[country] = {"n_components": len(cl), "selection_score": round(score, 6)}

    purged = purge_cross_split_near_duplicates(rows, purge_hamming)
    purged_ids = {p["image_id"] for p in purged}
    rows = [r for r in rows if r["image_id"] not in purged_ids]
    report = leakage_report(rows, max_hamming, purge_hamming)
    report.update({"generated_at": _now(), "seed": seed, "block_size": block_size,
                   "groups_merged_by_duplicates": merged_groups,
                   "near_duplicate_purge_hamming": purge_hamming,
                   "n_purged_near_duplicates": len(purged),
                   "purged_by_split": dict(Counter(p["split"] for p in purged))})

    is_repo_splits = splits_dir.resolve() == default_splits_dir().resolve()
    copied = backup_legacy_fixture_splits(splits_dir) if is_repo_splits else []
    splits_dir.mkdir(parents=True, exist_ok=True)
    out_rows = {s: sorted((r for r in rows if r["split"] == s), key=lambda r: r["image_id"]) for s in SPLITS}
    for s in SPLITS:
        write_csv(splits_dir / f"{s}.csv", out_rows[s], SPLIT_COLUMNS)
    purge_cols = [*SPLIT_COLUMNS, "purged_because_of", "neighbour_split"]
    write_csv(splits_dir / "near_duplicate_purged.csv", purged, purge_cols)
    write_csv(results_dir / "near_duplicate_purged.csv", purged, purge_cols)
    write_json(results_dir / "split_leakage_report.json", report)

    def _stats(rs: list[dict]) -> dict:
        return {
            "n_images": len(rs),
            "n_objects": sum(int(r["n_objects"]) for r in rs),
            "n_negatives": sum(int(r["is_negative"]) for r in rs),
            "class_counts": {n: sum(int(r[f"n_{n}"]) for r in rs) for n in CLASS_NAMES},
            "countries": dict(sorted(Counter(r["country"] for r in rs).items())),
        }

    summary = {
        "generated_at": _now(),
        "dataset": OFFICIAL_DATASET,
        "dataset_version": DATASET_VERSION,
        "split_id": f"grouped_{int(ratios[0]*100)}_{int(ratios[1]*100)}_{int(ratios[2]*100)}_seed{seed}",
        "seed": seed,
        "block_size": block_size,
        "ratios": list(ratios),
        "n_candidates_per_country": n_candidates,
        "near_duplicate_max_hamming": max_hamming,
        "near_duplicate_purge_hamming": purge_hamming,
        "n_purged_near_duplicates": len(purged),
        "purged": {"by_split": dict(Counter(p["split"] for p in purged)),
                   "negatives": sum(int(p["is_negative"]) for p in purged),
                   "objects": sum(int(p["n_objects"]) for p in purged)},
        "selection": selection,
        "splits": {s: _stats(out_rows[s]) for s in SPLITS},
        "total": _stats(rows),
        "n_converted_images": len(rows) + len(purged),
        "leakage_passed": report["passed"],
        "csv_sha256": {s: sha256_file(splits_dir / f"{s}.csv") for s in SPLITS},
        "legacy_fixture_backup": copied,
        "official_test_images_used": False,
    }
    write_json(splits_dir / "split_summary.json", summary)
    write_json(results_dir / "split_summary.json", summary)
    return {"summary": summary, "leakage": report}


# --------------------------------------------------------------------------- validation


def _check_label_file(path: Path) -> dict:
    res = {"n_boxes": 0, "bad_format": 0, "bad_class": 0, "not_normalized": 0,
           "nonpositive_wh": 0, "out_of_bounds": 0}
    eps = 1e-6
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        parts = ln.split()
        if len(parts) != 5:
            res["bad_format"] += 1
            continue
        try:
            cid = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            res["bad_format"] += 1
            continue
        res["n_boxes"] += 1
        if cid not in range(len(CLASS_NAMES)) or parts[0] != str(cid):
            res["bad_class"] += 1
        if not all(-eps <= v <= 1 + eps for v in (cx, cy, w, h)):
            res["not_normalized"] += 1
        if w <= 0 or h <= 0:
            res["nonpositive_wh"] += 1
        if cx - w / 2 < -eps or cy - h / 2 < -eps or cx + w / 2 > 1 + eps or cy + h / 2 > 1 + eps:
            res["out_of_bounds"] += 1
    return res


def validate_official(
    processed_root: Path,
    splits_dir: Path,
    *,
    exclude: set[str],
    workers: int,
    check_images: bool = True,
    require_splits: bool = False,
    max_hamming: int = 0,
    purge_hamming: int = 0,
) -> dict:
    images_root, labels_root = processed_root / "images", processed_root / "labels"
    if not images_root.is_dir():
        raise FileNotFoundError(f"{images_root} missing — run scripts/convert_voc_to_yolo.py first")
    images: list[tuple[str, Path]] = []
    for cdir in sorted(p for p in images_root.iterdir() if p.is_dir()):
        images += [(cdir.name, p) for p in _list_files(cdir, IMAGE_SUFFIXES)]
    errors: list[str] = []
    if check_images and images:
        args = [str(p) for _, p in images]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                errors = list(ex.map(verify_image, args, chunksize=64))
        else:
            errors = [verify_image(a) for a in args]
    rows: list[dict] = []
    tot: Counter = Counter()
    class_counts: Counter = Counter()
    image_keys = set()
    for k, (country, img) in enumerate(images):
        lbl = labels_root / country / f"{img.stem}.txt"
        image_keys.add(f"images/{country}/{img.name}")
        row = {"image_path": f"images/{country}/{img.name}", "country": country,
               "label_exists": lbl.is_file(), "readable": (errors[k] == "") if check_images else "not_checked",
               "read_error": errors[k] if check_images else "",
               "country_valid": country in KNOWN_COUNTRIES and country not in exclude}
        res = _check_label_file(lbl) if lbl.is_file() else {}
        row.update(res)
        if lbl.is_file():
            for ln in lbl.read_text().splitlines():
                if ln.strip() and ln.split()[0].isdigit():
                    class_counts[int(ln.split()[0])] += 1
        problems = []
        if not row["label_exists"]:
            problems.append("missing_label")
        for key in ("bad_format", "bad_class", "not_normalized", "nonpositive_wh", "out_of_bounds"):
            if res.get(key):
                problems.append(key)
        if check_images and errors[k]:
            problems.append("unreadable_image")
        if not row["country_valid"]:
            problems.append("invalid_country")
        row["issues"] = ";".join(problems)
        row["ok"] = not problems
        tot["images"] += 1
        tot["negatives"] += int(lbl.is_file() and res.get("n_boxes", 0) == 0 and not res.get("bad_format"))
        tot["boxes"] += res.get("n_boxes", 0)
        for p in problems:
            tot[p] += 1
        rows.append(row)
    orphan = 0
    if labels_root.is_dir():
        for cdir in (p for p in labels_root.iterdir() if p.is_dir()):
            for lf in cdir.glob("*.txt"):
                if not any((images_root / cdir.name / f"{lf.stem}{ext}").exists() for ext in (".jpg", ".jpeg", ".png")):
                    orphan += 1

    # split checks
    split_check: dict = {"status": "not_checked", "reason": "split CSVs not found"}
    csvs = [splits_dir / f"{s}.csv" for s in SPLITS]
    summary_path = splits_dir / "split_summary.json"
    if all(p.is_file() for p in csvs):
        dataset = json.loads(summary_path.read_text()).get("dataset") if summary_path.is_file() else None
        if dataset != OFFICIAL_DATASET:
            split_check = {"status": "not_checked", "reason": f"split CSVs belong to dataset={dataset!r}, not official"}
        else:
            srows = [r for p in csvs for r in read_csv(p)]
            purged_csv = splits_dir / "near_duplicate_purged.csv"
            purged = {r["image_path"] for r in read_csv(purged_csv)} if purged_csv.is_file() else set()
            listed = Counter(r["image_path"] for r in srows)
            missing = [k for k in listed if k not in image_keys]
            unassigned = [k for k in image_keys if k not in listed and k not in purged]
            leak = leakage_report(srows, max_hamming, purge_hamming)
            split_check = {
                "status": "checked",
                "n_rows": len(srows),
                "n_purged_near_duplicates": len(purged),
                "rows_missing_processed_image": len(missing),
                "processed_images_not_in_any_split": len(unassigned),
                "images_listed_more_than_once": sum(1 for v in listed.values() if v > 1),
                "missing_country": sum(1 for r in srows if not r.get("country")),
                "leakage": leak,
                "passed": leak["passed"] and not missing and not unassigned
                and all(v == 1 for v in listed.values()),
            }
    checks = {
        "every_image_has_label": tot["missing_label"] == 0,
        "five_values_per_line": tot["bad_format"] == 0,
        "class_ids_0_to_3": tot["bad_class"] == 0,
        "coordinates_normalized": tot["not_normalized"] == 0,
        "width_height_positive": tot["nonpositive_wh"] == 0,
        "boxes_within_bounds": tot["out_of_bounds"] == 0,
        "images_readable": tot["unreadable_image"] == 0 if check_images else "not_checked",
        "country_present": tot["invalid_country"] == 0,
        "no_orphan_labels": orphan == 0,
        "no_split_leakage": split_check.get("passed") if split_check["status"] == "checked" else "not_checked",
    }
    if require_splits and split_check["status"] != "checked":
        checks["no_split_leakage"] = False
    passed = all(v is True or v == "not_checked" for v in checks.values())
    summary = {
        "generated_at": _now(),
        "dataset": OFFICIAL_DATASET,
        "processed_root": str(processed_root),
        "n_images": tot["images"],
        "n_boxes": tot["boxes"],
        "n_negatives_empty_labels": tot["negatives"],
        "n_orphan_labels": orphan,
        "class_counts": {CLASS_NAMES[i]: class_counts.get(i, 0) for i in range(len(CLASS_NAMES))},
        "issue_counts": {k: v for k, v in tot.items() if k not in {"images", "negatives", "boxes"}},
        "checks": checks,
        "split_check": split_check,
        "passed": passed,
    }
    return {"summary": summary, "rows": rows}


# --------------------------------------------------------------------------- training data


def dataset_fingerprint(splits_dir: Path) -> dict:
    h = hashlib.sha256()
    per = {}
    for s in SPLITS:
        p = splits_dir / f"{s}.csv"
        if not p.is_file():
            per[s] = None
            continue
        data = p.read_bytes()
        h.update(s.encode() + b"\0" + data)
        per[s] = hashlib.sha256(data).hexdigest()[:16]
    return {"dataset_hash": h.hexdigest()[:16], "split_hashes": per}


def write_ultralytics_yaml(
    cfg: dict, *, processed_root: str | Path | None = None, out: Path | None = None
) -> Path:
    """Generate the per-machine Ultralytics YAML (absolute paths) from the split CSVs."""
    paths = official_paths(cfg, processed_root)
    processed, splits_dir = paths["processed"].resolve(), paths["splits"]
    if processed == legacy_fixture_processed_dir().resolve():
        raise ValueError("official training must not point at the legacy fixture processed dir")
    summary_path = splits_dir / "split_summary.json"
    if not summary_path.is_file() or json.loads(summary_path.read_text()).get("dataset") != OFFICIAL_DATASET:
        raise FileNotFoundError(
            f"Official split CSVs not found in {splits_dir}.\n"
            "Run: bash scripts/run_official_dataset_pipeline.sh  (or create_week3_splits.py --config configs/rdd2022.yaml)"
        )
    out = out or paths["ultralytics_yaml"]
    out.parent.mkdir(parents=True, exist_ok=True)
    lists = {}
    for s in SPLITS:
        srows = read_csv(splits_dir / f"{s}.csv")
        if not srows and s in ("train", "val"):
            raise ValueError(f"official {s} split is empty ({splits_dir / f'{s}.csv'})")
        imgs = [processed / r["image_path"] for r in srows]
        missing = [p for p in imgs if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} {s} images missing under {processed} (e.g. {missing[0]})")
        lst = out.parent / f"{s}.txt"
        lst.write_text("\n".join(str(p) for p in imgs) + "\n", encoding="utf-8")
        lists[s] = lst
    fp = dataset_fingerprint(splits_dir)
    names = cfg.get("names") or dict(enumerate(CLASS_NAMES))
    payload = {
        "path": str(processed),
        "train": str(lists["train"]),
        "val": str(lists["val"]),
        "test": str(lists["test"]),
        "nc": int(cfg.get("nc", 4)),
        "names": {int(k): v for k, v in names.items()},
    }
    header = (
        "# GENERATED per machine from configs/rdd2022.yaml + data/splits/*.csv — do not commit.\n"
        f"# dataset: {OFFICIAL_DATASET} ({cfg.get('dataset_version')}) dataset_hash={fp['dataset_hash']}\n"
    )
    out.write_text(header + yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def resolve_training_data(
    dataset: str = "official",
    *,
    config: str | Path | None = None,
    data_root: str | Path | None = None,
) -> dict:
    """Return the Ultralytics data YAML plus provenance for a run manifest.

    ``data_root`` (official): processed root containing ``images/`` and ``labels/``.
    """
    if dataset == "fixture":
        yaml_path = ensure_legacy_fixture_data_yaml()
        legacy = legacy_fixture_splits_dir()
        fp = dataset_fingerprint(legacy) if legacy.is_dir() else {"dataset_hash": None, "split_hashes": {}}
        return {
            "dataset": "fixture",
            "dataset_source": "SYNTHETIC FIXTURE (data/fixture/RDD2022) — debugging only, not RDD2022",
            "dataset_version": "synthetic-fixture",
            "ultralytics_yaml": str(yaml_path),
            "processed_root": str(legacy_fixture_processed_dir()),
            **fp,
        }
    if dataset != "official":
        raise ValueError(f"unknown dataset {dataset!r}")
    cfg = load_dataset_config(config)
    yaml_path = write_ultralytics_yaml(cfg, processed_root=data_root)
    paths = official_paths(cfg, data_root)
    fp = dataset_fingerprint(paths["splits"])
    return {
        "dataset": OFFICIAL_DATASET,
        "dataset_source": cfg.get("source"),
        "dataset_version": cfg.get("dataset_version"),
        "dataset_config": cfg["_config_path"],
        "ultralytics_yaml": str(yaml_path),
        "processed_root": str(paths["processed"].resolve()),
        "split_id": cfg.get("split_id"),
        **fp,
    }
