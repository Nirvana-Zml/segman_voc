from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from models.baselines import build_deeplabv3plus  # noqa: E402


class DeepLabV3PlusTest(unittest.TestCase):
    def test_forward_preserves_spatial_shape_and_class_count(self) -> None:
        model = build_deeplabv3plus(num_classes=21, encoder_pretrained=False)
        model.eval()

        with torch.no_grad():
            logits = model(torch.randn(1, 3, 64, 80))

        self.assertEqual(tuple(logits.shape), (1, 21, 64, 80))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
