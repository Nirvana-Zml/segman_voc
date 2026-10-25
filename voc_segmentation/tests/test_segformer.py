from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from models.baselines import build_segformer  # noqa: E402


class SegFormerTest(unittest.TestCase):
    def test_rejects_non_resource_safe_encoder(self) -> None:
        with self.assertRaisesRegex(ValueError, "mit_b1"):
            build_segformer(encoder="mit_b2")

    def test_encoder_feature_contract(self) -> None:
        model = build_segformer(num_classes=21)
        model.eval()

        with torch.no_grad():
            features = model.encoder(torch.randn(1, 3, 64, 96))

        self.assertEqual([feature.shape[1] for feature in features], [64, 128, 320, 512])
        self.assertEqual(
            [tuple(feature.shape[-2:]) for feature in features],
            [(16, 24), (8, 12), (4, 6), (2, 3)],
        )
        self.assertTrue(all(torch.isfinite(feature).all() for feature in features))

    def test_forward_preserves_spatial_shape_and_class_count(self) -> None:
        model = build_segformer(num_classes=21)
        model.eval()

        with torch.no_grad():
            logits = model(torch.randn(1, 3, 64, 80))

        self.assertEqual(tuple(logits.shape), (1, 21, 64, 80))
        self.assertTrue(torch.isfinite(logits).all())

    def test_explicit_encoder_checkpoint_round_trip(self) -> None:
        source = build_segformer(num_classes=21)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mit_b1.pth"
            torch.save(source.encoder.state_dict(), path)
            loaded = build_segformer(num_classes=21, encoder_weights_path=path)

        expected = next(source.encoder.parameters()).detach()
        actual = next(loaded.encoder.parameters()).detach()
        self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
