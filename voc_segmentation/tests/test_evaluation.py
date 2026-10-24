from __future__ import annotations

import unittest

import torch
from torch import nn

from engine import evaluate_segmentation


class FixedLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(image.shape[0], -1, -1, -1)


class EvaluationTest(unittest.TestCase):
    def test_global_metrics_and_loss_ignore_255(self) -> None:
        logits = torch.full((1, 3, 2, 2), -4.0)
        logits[0, 0, 0, 0] = 4.0
        logits[0, 1, 0, 1] = 4.0
        logits[0, 2, 1, 0] = 4.0
        logits[0, 2, 1, 1] = 4.0
        target = torch.tensor([[[0, 2], [2, 255]]])
        loader = [{"image": torch.zeros(1, 3, 2, 2), "mask": target}]
        model = FixedLogitModel(logits)
        model.train()

        result = evaluate_segmentation(
            model,
            loader,
            torch.device("cpu"),
            num_classes=3,
            ignore_index=255,
            amp=False,
        )

        self.assertTrue(model.training)
        self.assertEqual(result["valid_pixel_count"], 3)
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["batches"], 1)
        self.assertTrue(result["full_dataset"])
        self.assertAlmostEqual(result["pixel_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["miou"], (1.0 + 0.0 + 0.5) / 3)

    def test_rejects_nonpositive_limit(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_segmentation(
                nn.Identity(), [], torch.device("cpu"), max_batches=0
            )


if __name__ == "__main__":
    unittest.main()
