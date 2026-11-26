#!/usr/bin/env python3
"""Download and extract the official RDD2022 dataset (sekilab/RoadDamageDetector).

* Downloads only when no valid archive is present (resumable; S3 URL first, then the
  official Figshare mirror linked from the sekilab README).
* The release archive contains one zip per country (``RDD2022/<Country>.zip``). Inner
  zips are stored uncompressed, so they are extracted directly from the outer archive
  without writing a second copy to disk.
* Original JPG/XML files are written once and never modified afterwards.

Examples::

    python scripts/download_rdd2022.py
    python scripts/download_rdd2022.py --skip-download --archive /path/to/RDD2022.zip
"""

from __future__ import annotations

import argparse
import io
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    ARCHIVE_NAMES,
    OFFICIAL_ARCHIVE_URL,
    OFFICIAL_MIRROR_URL,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download + extract official RDD2022.")
    p.add_argument("--output-dir", default=str(ROOT / "data" / "raw"),
                   help="Directory holding the downloaded archive (default data/raw)")
    p.add_argument("--extract-dir", default=str(ROOT / "data" / "raw" / "RDD2022"),
                   help="Extraction target; country folders land directly inside it")
    p.add_argument("--archive", default="",
                   help="Explicit archive path (default: first existing known name in --output-dir)")
    p.add_argument("--skip-download", action="store_true",
                   help="Never download; fail if no valid archive/extraction exists")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--url", default=OFFICIAL_ARCHIVE_URL)
    p.add_argument("--mirror-url", default=OFFICIAL_MIRROR_URL)
    p.add_argument("--train-only", action="store_true",
                   help="Skip the unlabeled official test/ images during extraction")
    p.add_argument("--force-extract", action="store_true",
                   help="Re-extract even if files already exist (existing files are kept)")
    return p.parse_args()


def find_archive(output_dir: Path, explicit: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    for name in ARCHIVE_NAMES:
        cand = output_dir / name
        if cand.is_file():
            return cand.resolve()
    return (output_dir / "RDD2022.zip").resolve()


def archive_is_valid(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return False, "missing"
    if path.stat().st_size < 1_000_000:
        return False, f"too small ({path.stat().st_size} bytes)"
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
    except zipfile.BadZipFile as exc:
        return False, f"bad zip ({exc})"
    if not any("RDD2022" in n or n.endswith(".zip") for n in names):
        return False, "zip does not look like RDD2022"
    return True, f"ok ({len(names)} members, {path.stat().st_size / 1e9:.2f} GB)"


def _download(url: str, dest: Path) -> None:
    """Resumable download with retries (HTTP Range)."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, 11):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "rdd2022-downloader"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = resp.headers.get("Content-Length")
                total = int(total) + have if total else None
                mode = "ab" if have and resp.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                t0 = time.time()
                last = t0
                with tmp.open(mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 10:
                            last = time.time()
                            pct = f"{100 * have / total:.1f}%" if total else "?"
                            print(f"  {have / 1e9:.2f} GB ({pct})", flush=True)
            if total is None or have >= total:
                tmp.replace(dest)
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                raise
            print(f"  attempt {attempt} failed: {exc}; retrying in 10s", file=sys.stderr)
            time.sleep(10)
    raise RuntimeError(f"download did not complete: {url}")


class _Slice(io.RawIOBase):
    """Seekable read-only view of ``length`` bytes at ``offset`` in a file."""

    def __init__(self, fh, offset: int, length: int):
        self._fh, self._off, self._len, self._pos = fh, offset, length, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, pos: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._pos, 2: self._len}[whence]
        self._pos = max(0, base + pos)
        return self._pos

    def readinto(self, buf) -> int:
        n = min(len(buf), self._len - self._pos)
        if n <= 0:
            return 0
        self._fh.seek(self._off + self._pos)
        data = self._fh.read(n)
        buf[: len(data)] = data
        self._pos += len(data)
        return len(data)


def _member_data_offset(archive: Path, info: zipfile.ZipInfo) -> int:
    with archive.open("rb") as f:
        f.seek(info.header_offset)
        header = f.read(30)
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    return info.header_offset + 30 + name_len + extra_len


def _extract_zip(zf: zipfile.ZipFile, extract_dir: Path, *, train_only: bool, force: bool) -> Counter:
    counts: Counter = Counter()
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = Path(info.filename).parts
        if parts and parts[0] == "RDD2022":
            parts = parts[1:]
        if not parts or ".." in parts:
            continue
        if train_only and len(parts) > 1 and parts[1] == "test":
            continue
        dest = extract_dir.joinpath(*parts)
        kind = "xml" if dest.suffix.lower() == ".xml" else "jpg" if dest.suffix.lower() in {".jpg", ".jpeg", ".png"} else "other"
        if dest.exists() and not force and dest.stat().st_size == info.file_size:
            counts[f"{kind}_existing"] += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
        counts[f"{kind}_written"] += 1
    return counts


def extract(archive: Path, extract_dir: Path, *, train_only: bool, force: bool) -> Counter:
    extract_dir.mkdir(parents=True, exist_ok=True)
    total: Counter = Counter()
    with zipfile.ZipFile(archive) as outer:
        inner_zips = [i for i in outer.infolist() if i.filename.lower().endswith(".zip")]
        if not inner_zips:
            return _extract_zip(outer, extract_dir, train_only=train_only, force=force)
        for info in inner_zips:
            print(f"  extracting {info.filename} ({info.file_size / 1e9:.2f} GB)", flush=True)
            if info.compress_type == zipfile.ZIP_STORED:
                with archive.open("rb") as fh:
                    view = io.BufferedReader(_Slice(fh, _member_data_offset(archive, info), info.file_size), 1 << 20)
                    with zipfile.ZipFile(view) as inner:
                        total += _extract_zip(inner, extract_dir, train_only=train_only, force=force)
            else:
                tmp = extract_dir / f".tmp_{Path(info.filename).name}"
                with outer.open(info) as src, tmp.open("wb") as out:
                    shutil.copyfileobj(src, out, 1 << 20)
                try:
                    with zipfile.ZipFile(tmp) as inner:
                        total += _extract_zip(inner, extract_dir, train_only=train_only, force=force)
                finally:
                    tmp.unlink(missing_ok=True)
    return total


def count_extracted(extract_dir: Path) -> dict:
    out: dict = {}
    for country in sorted(p for p in extract_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        row = {}
        for split in ("train", "test"):
            img_dir = country / split / "images"
            row[f"{split}_images"] = sum(1 for f in img_dir.iterdir() if f.is_file()) if img_dir.is_dir() else 0
        xml_dir = country / "train" / "annotations" / "xmls"
        row["train_xmls"] = sum(1 for f in xml_dir.iterdir() if f.suffix.lower() == ".xml") if xml_dir.is_dir() else 0
        out[country.name] = row
    return out


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    extract_dir = Path(args.extract_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = find_archive(output_dir, args.archive)

    ok, why = archive_is_valid(archive)
    print(f"Archive path:    {archive}")
    print(f"Archive status:  {why}")
    if not ok:
        already = extract_dir.is_dir() and any(extract_dir.glob("*/train/images"))
        if args.skip_download:
            if already:
                print("--skip-download: no valid archive, but an extraction already exists; verifying it.")
            else:
                print("ERROR: --skip-download given and no valid archive or extraction found.", file=sys.stderr)
                return 1
        else:
            for url in (args.url, args.mirror_url):
                print(f"Downloading: {url}")
                try:
                    _download(url, archive)
                    ok, why = archive_is_valid(archive)
                    print(f"Archive status:  {why}")
                    if ok:
                        break
                except Exception as exc:  # noqa: BLE001
                    print(f"  failed: {exc}", file=sys.stderr)
            if not ok:
                print("ERROR: could not obtain a valid RDD2022 archive.", file=sys.stderr)
                return 1

    if ok and not args.skip_extract:
        print(f"Extracting into: {extract_dir}")
        counts = extract(archive, extract_dir, train_only=args.train_only, force=args.force_extract)
        print(f"  files written={sum(v for k, v in counts.items() if k.endswith('_written'))} "
              f"already present={sum(v for k, v in counts.items() if k.endswith('_existing'))}")

    if not extract_dir.is_dir():
        print(f"ERROR: extraction dir missing: {extract_dir}", file=sys.stderr)
        return 1
    print(f"Extraction path: {extract_dir}")
    stats = count_extracted(extract_dir)
    print(f"{'country':<18}{'train_images':>14}{'train_xmls':>12}{'test_images(unlabeled)':>24}")
    tot = Counter()
    for country, row in stats.items():
        print(f"{country:<18}{row['train_images']:>14}{row['train_xmls']:>12}{row['test_images']:>24}")
        tot.update(row)
    print(f"{'TOTAL':<18}{tot['train_images']:>14}{tot['train_xmls']:>12}{tot['test_images']:>24}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
