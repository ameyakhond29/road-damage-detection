"""Unit tests for the official RDD2022 data pipeline.

Tests build tiny synthetic VOC trees in temporary directories (or use the synthetic
fixture explicitly). Nothing here reads or writes the real official dataset or
results/official_rdd2022/.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]

from src.constants import CLASS_NAMES, CLASS_TO_ID, RDD2022_ROOT_ENV  # noqa: E402
from src.dataset import official as off  # noqa: E402
from src.dataset.voc import VocBox, parse_voc_xml  # noqa: E402
from src.paths import (  # noqa: E402
    default_processed_dir,
    fixture_raw_dir,
    legacy_fixture_processed_dir,
    resolve_dataset_root,
    resolve_processed_root,
)


def _xml(path: Path, w: int, h: int, objects: list[tuple[str, int, int, int, int]]) -> None:
    objs = "".join(
        f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin>"
        f"<xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
        for n, a, b, c, d in objects
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"<annotation><size><width>{w}</width><height>{h}</height><depth>3</depth></size>{objs}</annotation>",
        encoding="utf-8",
    )


def _image(path: Path, seed: int, w: int = 64, h: int = 48) -> None:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(h // 8, w // 8, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).resize((w, h)).save(path, quality=95)


def make_raw_tree(root: Path, countries: dict[str, int]) -> Path:
    """RDD2022/<Country>/train/{images,annotations/xmls} with mixed boxes/negatives."""
    raw = root / "RDD2022"
    seed = 0
    for country, n in countries.items():
        for i in range(n):
            stem = f"{country}_{i:06d}"
            _image(raw / country / "train" / "images" / f"{stem}.jpg", seed)
            seed += 1
            if i % 4 == 0:
                objs = [("D44", 1, 1, 10, 10)]  # unsupported only -> negative
            else:
                objs = [(CLASS_NAMES[i % 4], 5, 5, 30, 20), ("D50", 2, 2, 8, 8)]
            _xml(raw / country / "train" / "annotations" / "xmls" / f"{stem}.xml", 64, 48, objs)
        (raw / country / "test" / "images").mkdir(parents=True, exist_ok=True)
        _image(raw / country / "test" / "images" / f"{country}_test_0.jpg", 999)
    return raw


class TestClassMapping(unittest.TestCase):
    def test_exact_mapping(self):
        self.assertEqual(CLASS_TO_ID, {"D00": 0, "D10": 1, "D20": 2, "D40": 3})
        self.assertEqual(CLASS_NAMES, ["D00", "D10", "D20", "D40"])

    def test_config_mapping_and_exclusion(self):
        cfg = off.load_dataset_config(ROOT / "configs" / "rdd2022.yaml")
        self.assertEqual({int(k): v for k, v in cfg["names"].items()}, {0: "D00", 1: "D10", 2: "D20", 3: "D40"})
        self.assertEqual(cfg["nc"], 4)
        self.assertIn("China_Drone", cfg["excluded_folders"])
        self.assertEqual(cfg["split_seed"], 6845)

    def test_tracked_config_has_no_machine_paths(self):
        text = (ROOT / "configs" / "rdd2022.yaml").read_text()
        self.assertNotIn("/Users/", text)
        self.assertNotIn("fixture", yaml.safe_load(text)["raw_root"])


class TestVocToYolo(unittest.TestCase):
    def test_conversion_values(self):
        lines, unsupported, rejected = off.convert_annotation([VocBox("D10", 10, 20, 30, 60)], 100, 200)
        self.assertEqual(lines, ["1 0.200000 0.200000 0.200000 0.200000"])
        self.assertEqual(unsupported, {})
        self.assertEqual(rejected, [])

    def test_parse_xml_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "a.xml"
            _xml(p, 640, 480, [("D40", 0, 0, 64, 48)])
            ann = parse_voc_xml(p)
            lines, _, _ = off.convert_annotation(ann.boxes, ann.width, ann.height)
            self.assertEqual(lines, ["3 0.050000 0.050000 0.100000 0.100000"])

    def test_unsupported_labels_counted_and_negative_is_empty(self):
        lines, unsupported, rejected = off.convert_annotation(
            [VocBox("D44", 1, 1, 5, 5), VocBox("Repair", 1, 1, 5, 5)], 64, 48)
        self.assertEqual(lines, [])
        self.assertEqual(unsupported, {"D44": 1, "Repair": 1})
        self.assertEqual(rejected, [])

    def test_invalid_boxes_rejected(self):
        boxes = [VocBox("D00", 30, 5, 10, 20),     # xmax <= xmin
                 VocBox("D00", 5, 30, 20, 30),     # ymax == ymin
                 VocBox("D00", 100, 100, 120, 120),  # fully outside 64x48
                 VocBox("D00", -5, -5, 10, 10)]    # partially outside -> clipped, kept
        lines, _, rejected = off.convert_annotation(boxes, 64, 48)
        self.assertEqual(sorted(rejected), ["box_outside_image", "invalid_box_order", "invalid_box_order"])
        self.assertEqual(len(lines), 1)
        cid, cx, cy, w, h = lines[0].split()
        self.assertTrue(0 < float(w) <= 1 and 0 < float(h) <= 1)


class TestPathResolution(unittest.TestCase):
    def test_env_var_and_cli_override(self):
        with tempfile.TemporaryDirectory() as td:
            env_raw = make_raw_tree(Path(td) / "env", {"Japan": 1})
            cli_raw = make_raw_tree(Path(td) / "cli", {"Czech": 1})
            with mock.patch.dict(os.environ, {RDD2022_ROOT_ENV: str(env_raw.parent)}):
                self.assertEqual(resolve_dataset_root(None, "official"), env_raw.resolve())
                self.assertEqual(resolve_dataset_root(str(cli_raw), "official"), cli_raw.resolve())

    def test_recursive_search(self):
        with tempfile.TemporaryDirectory() as td:
            raw = make_raw_tree(Path(td) / "a" / "b", {"India": 1})
            self.assertEqual(resolve_dataset_root(Path(td), "official"), raw.resolve())

    def test_missing_data_message(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError) as ctx:
                resolve_dataset_root(Path(td) / "nope", "official")
            self.assertIn("download_rdd2022.py", str(ctx.exception))

    def test_official_is_default_and_not_fixture(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(RDD2022_ROOT_ENV, None)
            p = resolve_dataset_root(None, "official", must_exist=False)
        self.assertNotIn("fixture", p.as_posix())
        self.assertEqual(resolve_processed_root(None, "official"), default_processed_dir())
        self.assertNotEqual(default_processed_dir(), legacy_fixture_processed_dir())

    def test_fixture_mode_is_explicit(self):
        self.assertEqual(resolve_dataset_root(None, "fixture", must_exist=False), fixture_raw_dir())
        self.assertEqual(resolve_processed_root(None, "fixture"), legacy_fixture_processed_dir())
        with self.assertRaises(ValueError):
            resolve_dataset_root(None, "synthetic")


class _TmpDataset(unittest.TestCase):
    """Builds raw -> processed -> splits once per test class in a temp dir."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.raw = make_raw_tree(cls.tmp, {"Japan": 120, "India": 60, "China_Drone": 5})
        # exact duplicate placed in a different 10-block group of the same country
        shutil.copy2(cls.raw / "Japan/train/images/Japan_000005.jpg",
                     cls.raw / "Japan/train/images/Japan_000110.jpg")
        cls.processed = cls.tmp / "processed"
        cls.results = cls.tmp / "results"
        cls.splits = cls.tmp / "splits"
        cls.conv = off.convert_official(cls.raw, cls.processed, cls.results, exclude={"China_Drone"})
        cls.split_res = off.create_official_splits(cls.processed, cls.splits, cls.results, seed=6845,
                                                   block_size=10, n_candidates=20, workers=1)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestConversionAndSplits(_TmpDataset):
    def test_exclusion_and_counts(self):
        self.assertEqual(self.conv["n_images_converted"], 180)
        self.assertFalse((self.processed / "images" / "China_Drone").exists())
        self.assertEqual(self.conv["unsupported_labels_dropped"]["D44"], 45)
        self.assertFalse(self.conv["originals_modified"])

    def test_negatives_get_empty_label_files(self):
        lbl = self.processed / "labels" / "Japan" / "Japan_000000.txt"
        self.assertTrue(lbl.is_file())
        self.assertEqual(lbl.read_text(), "")
        self.assertEqual(self.conv["n_negatives_empty_labels"], 45)

    def test_originals_untouched_and_test_images_unused(self):
        xml = (self.raw / "Japan/train/annotations/xmls/Japan_000001.xml").read_text()
        self.assertIn("<name>D10</name>", xml)
        rows = off.read_csv(self.processed / "manifest.csv")
        self.assertFalse(any("test" in r["source_image"].split("/") for r in rows))

    def test_split_leakage_and_determinism(self):
        leak = self.split_res["leakage"]
        self.assertTrue(leak["passed"])
        self.assertGreaterEqual(leak["duplicate_clusters"], 1)
        rows = {r["image_id"]: r for s in off.SPLITS for r in off.read_csv(self.splits / f"{s}.csv")}
        self.assertEqual(len(rows), 180)
        self.assertEqual(rows["Japan/Japan_000005"]["split"], rows["Japan/Japan_000110"]["split"])
        first = {s: (self.splits / f"{s}.csv").read_bytes() for s in off.SPLITS}
        off.create_official_splits(self.processed, self.splits, self.results, seed=6845,
                                   block_size=10, n_candidates=20, workers=1)
        self.assertEqual(first, {s: (self.splits / f"{s}.csv").read_bytes() for s in off.SPLITS})

    def test_near_duplicate_purge_is_non_transitive(self):
        base = {"country": "J", "md5": "", "n_objects": "0", "is_negative": "1"}
        rows = [{**base, "image_id": "a", "group_id": "g1", "split": "train", "phash": "0000000000000000"},
                {**base, "image_id": "b", "group_id": "g2", "split": "test", "phash": "0000000000000003"},
                {**base, "image_id": "c", "group_id": "g3", "split": "val", "phash": "ffffffffffffffff"}]
        self.assertFalse(off.leakage_report(rows, 0, 4)["passed"])
        purged = off.purge_cross_split_near_duplicates(rows, 4)
        self.assertEqual([p["image_id"] for p in purged], ["b"])
        kept = [r for r in rows if r["image_id"] != "b"]
        self.assertTrue(off.leakage_report(kept, 0, 4)["passed"])

    def test_leakage_report_detects_crossing(self):
        rows = [{"image_id": "a", "group_id": "g1", "split": "train", "md5": "x", "phash": "", "country": "J"},
                {"image_id": "b", "group_id": "g2", "split": "test", "md5": "x", "phash": "", "country": "J"}]
        self.assertFalse(off.leakage_report(rows)["passed"])

    def test_ultralytics_yaml_generation(self):
        cfg = {"dataset": "official_rdd2022", "dataset_version": "t", "nc": 4,
               "names": {0: "D00", 1: "D10", 2: "D20", 3: "D40"},
               "processed_root": str(self.processed), "splits_dir": str(self.splits),
               "ultralytics_yaml": str(self.tmp / "ul" / "rdd.yaml")}
        out = off.write_ultralytics_yaml(cfg)
        data = yaml.safe_load(out.read_text())
        self.assertTrue(Path(data["path"]).is_absolute())
        self.assertEqual(data["nc"], 4)
        self.assertEqual(data["names"], {0: "D00", 1: "D10", 2: "D20", 3: "D40"})
        for s in off.SPLITS:
            imgs = Path(data[s]).read_text().splitlines()
            self.assertTrue(imgs and all(Path(p).is_file() and "/images/" in p for p in imgs), s)

    def test_official_yaml_refuses_fixture_processed_dir(self):
        cfg = {"dataset": "official_rdd2022", "processed_root": str(legacy_fixture_processed_dir()),
               "splits_dir": str(self.splits)}
        with self.assertRaises(ValueError):
            off.write_ultralytics_yaml(cfg)

    def test_validation_and_image_label_matching(self):
        res = off.validate_official(self.processed, self.splits, exclude={"China_Drone"}, workers=1)
        self.assertTrue(res["summary"]["passed"], res["summary"]["checks"])
        self.assertEqual(res["summary"]["split_check"]["status"], "checked")
        lbl = self.processed / "labels" / "India" / "India_000001.txt"
        backup = lbl.read_text()
        try:
            lbl.unlink()
            bad = off.validate_official(self.processed, self.splits, exclude={"China_Drone"}, workers=1)
            self.assertFalse(bad["summary"]["checks"]["every_image_has_label"])
            lbl.write_text("7 0.5 0.5 0.2 0.2\n0 0.5 0.5 1.5 0.2\n")
            bad = off.validate_official(self.processed, self.splits, exclude={"China_Drone"}, workers=1)
            self.assertFalse(bad["summary"]["checks"]["class_ids_0_to_3"])
            self.assertFalse(bad["summary"]["checks"]["boxes_within_bounds"])
        finally:
            lbl.write_text(backup)


class TestFixtureIsolation(unittest.TestCase):
    def test_fixture_training_data_is_labelled_fixture(self):
        if not legacy_fixture_processed_dir().is_dir():
            self.skipTest("legacy fixture not present")
        info = off.resolve_training_data("fixture")
        self.assertEqual(info["dataset"], "fixture")
        self.assertIn("RDD2022_yolo", info["ultralytics_yaml"])
        self.assertIn("SYNTHETIC", info["dataset_source"])

    def test_fixture_runs_cannot_write_official_results(self):
        from src.training.baseline import prepare_training_run

        if not legacy_fixture_processed_dir().is_dir():
            self.skipTest("legacy fixture not present")
        with self.assertRaises(SystemExit):
            prepare_training_run(dataset="fixture", config=None, data_root=None,
                                 output_dir=str(ROOT / "results" / "official_rdd2022" / "x"),
                                 default_rel="x")
        with self.assertRaises(SystemExit):
            prepare_training_run(dataset="fixture", config=None, data_root=None,
                                 output_dir=str(ROOT / "results" / "week3" / "x"), default_rel="x")


if __name__ == "__main__":
    unittest.main()
