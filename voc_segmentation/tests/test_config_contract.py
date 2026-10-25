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

    def test_deeplab_uses_explicit_provenance_weight_path(self) -> None:
        path = CONFIG_DIR / "deeplabv3plus_r50.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["model"]["encoder_pretrained"], "imagenet1k_v2")
        self.assertEqual(
            config["model"]["encoder_weights_path"],
            "artifacts/weights/resnet50_imagenet1k_v2/resnet50-11ad3fa6.pth",
        )
        training = config["training"]
        self.assertGreater(training["max_iterations"], 0)
        self.assertGreater(training["validate_every"], 0)
        self.assertLess(training["warmup_iterations"], training["max_iterations"])
        self.assertGreater(training["poly_power"], 0)
        self.assertGreater(training["max_consecutive_amp_overflows"], 0)
        self.assertGreaterEqual(
            training["max_total_amp_overflows"],
            training["max_consecutive_amp_overflows"],
        )
        self.assertLess(
            training["encoder_lr_multiplier"], training["decoder_lr_multiplier"]
        )


if __name__ == "__main__":
    unittest.main()
