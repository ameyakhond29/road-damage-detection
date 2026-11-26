"""Perceptual-hash near-duplicate detection."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def _phash(path: Path, hash_size: int = 8):
    import imagehash

    with Image.open(path) as img:
        img = img.convert("RGB")
        return imagehash.phash(img, hash_size=hash_size)


def find_near_duplicates(
    records: list[dict],
    *,
    max_hamming: int = 0,
    hash_size: int = 8,
) -> pd.DataFrame:
    """Detect near-duplicate images via perceptual hash.

    ``records`` items must include keys: ``image_id``, ``path``, ``country``,
    ``group_id`` (optional but recommended).

    Returns a DataFrame of duplicate pairs with Hamming distance.
    When ``max_hamming == 0``, only exact phash matches are reported
    (efficient via hash bucketing). For ``max_hamming > 0``, pairwise
    comparison within country is used (slower).
    """
    import imagehash

    hashed: list[dict] = []
    for rec in tqdm(records, desc="Computing perceptual hashes"):
        path = Path(rec["path"])
        try:
            h = _phash(path, hash_size=hash_size)
        except Exception as exc:  # noqa: BLE001 — log and skip unreadable
            hashed.append(
                {
                    **rec,
                    "phash": None,
                    "error": str(exc),
                }
            )
            continue
        hashed.append({**rec, "phash": str(h), "phash_obj": h, "error": None})

    pairs: list[dict] = []

    if max_hamming == 0:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for item in hashed:
            if item.get("phash") is None:
                continue
            buckets[item["phash"]].append(item)
        for ph, items in buckets.items():
            if len(items) < 2:
                continue
            # Connect all to the first as canonical representative
            canon = items[0]
            for other in items[1:]:
                pairs.append(
                    {
                        "image_id_a": canon["image_id"],
                        "image_id_b": other["image_id"],
                        "path_a": canon["path"],
                        "path_b": other["path"],
                        "country_a": canon.get("country"),
                        "country_b": other.get("country"),
                        "group_id_a": canon.get("group_id"),
                        "group_id_b": other.get("group_id"),
                        "phash": ph,
                        "hamming": 0,
                    }
                )
    else:
        # Group by country to limit comparisons
        by_country: dict[str, list[dict]] = defaultdict(list)
        for item in hashed:
            if item.get("phash_obj") is None:
                continue
            by_country[str(item.get("country", "unknown"))].append(item)

        for country, items in by_country.items():
            n = len(items)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = items[i]["phash_obj"] - items[j]["phash_obj"]
                    if dist <= max_hamming:
                        pairs.append(
                            {
                                "image_id_a": items[i]["image_id"],
                                "image_id_b": items[j]["image_id"],
                                "path_a": items[i]["path"],
                                "path_b": items[j]["path"],
                                "country_a": items[i].get("country"),
                                "country_b": items[j].get("country"),
                                "group_id_a": items[i].get("group_id"),
                                "group_id_b": items[j].get("group_id"),
                                "phash": items[i]["phash"],
                                "hamming": int(dist),
                            }
                        )

    return pd.DataFrame(pairs)


def build_duplicate_components(dup_df: pd.DataFrame) -> dict[str, str]:
    """Union-find: map each image_id to a canonical component representative."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if dup_df.empty:
        return {}

    for _, row in dup_df.iterrows():
        union(str(row["image_id_a"]), str(row["image_id_b"]))

    return {k: find(k) for k in parent}
