"""Unit tests for Week 5 YOLO model YAMLs and registry."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


class TestWeek5ModelYAMLs(unittest.TestCase):
    FILES = {
        "p2": ROOT / "configs" / "models" / "yolo11n-p2.yaml",
        "attn": ROOT / "configs" / "models" / "yolo11n-attn.yaml",
        "combined": ROOT / "configs" / "models" / "yolo11n-p2-attn.yaml",
    }

    def test_yaml_files_exist_and_parse(self):
        for name, path in self.FILES.items():
            self.assertTrue(path.is_file(), f"missing {path}")
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(data.get("nc"), 4, name)
            self.assertIn("backbone", data, name)
            self.assertIn("head", data, name)
            self.assertNotEqual(data.get("placeholder"), True, name)

    def test_detect_layer_shapes(self):
        """P2 models must Detect 4 scales; attn-only Detect 3 scales."""
        p2 = yaml.safe_load(self.FILES["p2"].read_text(encoding="utf-8"))
        attn = yaml.safe_load(self.FILES["attn"].read_text(encoding="utf-8"))
        comb = yaml.safe_load(self.FILES["combined"].read_text(encoding="utf-8"))

        def detect_from(head):
            for row in head:
                if row[2] == "Detect":
                    return row[0]
            return None

        self.assertEqual(len(detect_from(p2["head"])), 4)
        self.assertEqual(len(detect_from(attn["head"])), 3)
        self.assertEqual(len(detect_from(comb["head"])), 4)

    def test_attention_uses_c2psa(self):
        attn = yaml.safe_load(self.FILES["attn"].read_text(encoding="utf-8"))
        modules = [row[2] for row in attn["head"]]
        self.assertIn("C2PSA", modules)

    def test_models_build_and_forward(self):
        from ultralytics import YOLO

        for name, path in self.FILES.items():
            with self.subTest(model=name):
                model = YOLO(str(path))
                n = sum(p.numel() for p in model.model.parameters())
                self.assertGreater(n, 1_000_000)
                img = np.zeros((640, 640, 3), dtype=np.uint8)
                out = model.predict(source=img, imgsz=640, device="cpu", verbose=False)
                self.assertTrue(len(out) >= 1)

    def test_p2_has_more_params_than_attn_only(self):
        from ultralytics import YOLO

        p2 = sum(p.numel() for p in YOLO(str(self.FILES["p2"])).model.parameters())
        attn = sum(p.numel() for p in YOLO(str(self.FILES["attn"])).model.parameters())
        # P2 adds an extra Detect scale → typically more params than attn-only neck insert
        self.assertGreater(p2, attn)


class TestWeek5Registry(unittest.TestCase):
    def test_registry_keys(self):
        from src.models.resolve import MODEL_REGISTRY, resolve_model_spec

        expected = {
            "yolo11n_640",
            "yolo11n_p2_640",
            "yolo11n_attention_640",
            "yolo11n_800",
            "combined_800",
        }
        self.assertEqual(set(MODEL_REGISTRY.keys()), expected)
        for e in expected:
            spec = resolve_model_spec(experiment=e)
            self.assertIn("model", spec)
            if str(spec["model"]).endswith((".yaml", ".yml")):
                self.assertTrue(Path(spec["model"]).is_file(), spec["model"])

    def test_stub_aliases(self):
        from src.models.resolve import resolve_model_spec

        spec = resolve_model_spec(model="configs/models/yolo11n_p2.yaml")
        self.assertTrue(Path(spec["model"]).is_file())
        self.assertIn("yolo11n-p2.yaml", spec["model"])


class TestWeek5Configs(unittest.TestCase):
    def test_week5_configs_share_split(self):
        configs = list((ROOT / "configs").glob("week5_*.yaml"))
        self.assertEqual(len(configs), 5)
        for path in configs:
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(cfg.get("data"), "configs/rdd2022.yaml", path.name)
            self.assertIs(cfg.get("test_used_for_selection"), False, path.name)
            self.assertEqual(cfg.get("checkpoint_selection"), "best_val_mAP50", path.name)
            mapping = cfg.get("class_mapping") or {}
            self.assertEqual(mapping.get("D00"), 0)
            self.assertEqual(mapping.get("D40"), 3)


if __name__ == "__main__":
    unittest.main()
