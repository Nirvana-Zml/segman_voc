from __future__ import annotations

import unittest
from pathlib import Path

import yaml


APP_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = APP_ROOT / "configs"


class ConfigContractTest(unittest.TestCase):
    def test_all_configs_preserve_voc_contract(self) -> None:
        paths = sorted(CONFIG_DIR.glob("*.yaml"))
        self.assertGreaterEqual(len(paths), 4)
        for path in paths:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(config["model"]["num_classes"], 21)
                self.assertEqual(config["data"]["ignore_index"], 255)
                self.assertFalse(config["data"]["reduce_zero_label"])
                self.assertEqual(config["data"]["val_split"], "val")
                self.assertNotEqual(
                    config["data"]["train_split"], config["data"]["val_split"]
                )
                self.assertEqual(config["training"]["save_best_metric"], "miou")

    def test_b2_target_requires_memory_profile(self) -> None:
        path = CONFIG_DIR / "gcb_seg_b2_512.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertTrue(config["experiment"]["requires_memory_profile"])


if __name__ == "__main__":
    unittest.main()
