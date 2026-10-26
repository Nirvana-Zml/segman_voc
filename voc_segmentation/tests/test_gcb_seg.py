from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from models import build_gcb_seg  # noqa: E402
from models.modules import GlobalContextModule  # noqa: E402


class GCBSegTest(unittest.TestCase):
    def test_training_and_inference_output_contract(self) -> None:
        model = build_gcb_seg(encoder_weights_path=None)
        image = torch.randn(1, 3, 64, 96)

        model.train()
        outputs = model(image)
        self.assertIsInstance(outputs, dict)
        self.assertEqual(set(outputs), {"logits", "aux_logits", "edge_logits"})
        self.assertEqual(outputs["logits"].shape, (1, 21, 64, 96))
        self.assertEqual(outputs["aux_logits"].shape, (1, 21, 64, 96))
        self.assertEqual(outputs["edge_logits"].shape, (1, 1, 64, 96))
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))

        model.eval()
        with torch.no_grad():
            logits = model(image)
        self.assertEqual(logits.shape, (1, 21, 64, 96))
        self.assertTrue(torch.isfinite(logits).all())

    def test_ablation_switches_preserve_main_output(self) -> None:
        model = build_gcb_seg(
            use_global_pooling=False,
            use_gated_fusion=False,
            use_boundary_branch=False,
            auxiliary_head=False,
        )
        model.train()
        outputs = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(outputs["logits"].shape, (1, 21, 64, 64))
        self.assertIsNone(outputs["aux_logits"])
        self.assertIsNone(outputs["edge_logits"])

    def test_new_modules_use_group_norm_not_batch_norm(self) -> None:
        model = build_gcb_seg()
        decoder_modules = (
            model.global_context,
            model.feature_fusion,
            model.boundary_branch,
            model.main_head,
            model.auxiliary_head,
        )
        self.assertFalse(
            any(
                isinstance(module, nn.BatchNorm2d)
                for root in decoder_modules
                if root is not None
                for module in root.modules()
            )
        )
        self.assertTrue(
            any(
                isinstance(module, nn.GroupNorm)
                for root in decoder_modules
                if root is not None
                for module in root.modules()
            )
        )

    def test_context_module_rejects_unplanned_dilations(self) -> None:
        with self.assertRaises(ValueError):
            GlobalContextModule(512, dilations=(1, 2, 4))


if __name__ == "__main__":
    unittest.main()
