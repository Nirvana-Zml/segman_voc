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

    def test_segformer_uses_resource_safe_profile_and_explicit_weight_path(self) -> None:
        path = CONFIG_DIR / "segformer_b1.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["model"]["encoder"], "mit_b1")
        self.assertEqual(
            config["model"]["encoder_weights_path"],
            "artifacts/weights/mit_b1_imagenet1k/mit_b1.pth",
        )
        self.assertEqual(config["data"]["crop_size"], [384, 384])
        self.assertEqual(config["training"]["effective_batch_size"], 8)
        training = config["training"]
        self.assertEqual(
            training["gradient_accumulation_steps"]
            * training["physical_batch_size"],
            training["effective_batch_size"],
        )
        self.assertEqual(training["optimizer"], "adamw")
        self.assertLess(training["encoder_lr"], training["decoder_lr"])
        self.assertGreater(training["gradient_clip_norm"], 0)
        self.assertLess(training["warmup_iterations"], training["max_iterations"])
        self.assertGreater(training["max_total_amp_overflows"], 0)

    def test_gcb_uses_verified_b1_profile_and_planned_components(self) -> None:
        path = CONFIG_DIR / "gcb_seg_b1_384.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = config["model"]
        self.assertEqual(model["encoder"], "mit_b1")
        self.assertEqual(
            model["encoder_weights_path"],
            "artifacts/weights/mit_b1_imagenet1k/mit_b1.pth",
        )
        self.assertEqual(model["global_context_dilations"], [1, 3, 6])
        self.assertTrue(model["use_global_pooling"])
        self.assertTrue(model["use_gated_fusion"])
        self.assertTrue(model["use_boundary_branch"])
        self.assertTrue(model["auxiliary_head"])
        self.assertEqual(config["data"]["crop_size"], [384, 384])
        training = config["training"]
        self.assertEqual(
            training["physical_batch_size"]
            * training["gradient_accumulation_steps"],
            training["effective_batch_size"],
        )


if __name__ == "__main__":
    unittest.main()
