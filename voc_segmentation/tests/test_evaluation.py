from __future__ import annotations

import unittest

import torch
from torch import nn

from engine import evaluate_segmentation, evaluate_segmentation_tta, predict_tta_logits


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

    def test_tta_averages_scales_and_flip_back_at_native_size(self) -> None:
        class CoordinateModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shapes: list[tuple[int, int]] = []

            def forward(self, image: torch.Tensor) -> torch.Tensor:
                self.shapes.append(tuple(image.shape[-2:]))
                return torch.cat((image[:, :1], -image[:, :1]), dim=1)

        model = CoordinateModel()
        image = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
        logits = predict_tta_logits(
            model,
            image,
            scales=(0.5, 1.0),
            horizontal_flip=True,
            amp_enabled=False,
        )

        self.assertEqual(logits.shape, (1, 2, 4, 6))
        self.assertEqual(model.shapes, [(2, 3), (2, 3), (4, 6), (4, 6)])
        self.assertTrue(torch.isfinite(logits).all())

    def test_tta_evaluation_records_logit_aggregation_contract(self) -> None:
        image = torch.zeros(1, 3, 2, 2)
        image[:, 0, 0, 0] = 2.0
        target = torch.tensor([[[0, 1], [1, 255]]])

        class BinaryModel(nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                first = value[:, :1]
                return torch.cat((first, 1.0 - first), dim=1)

        model = BinaryModel()
        model.train()
        result = evaluate_segmentation_tta(
            model,
            [{"image": image, "mask": target}],
            torch.device("cpu"),
            scales=(1.0,),
            horizontal_flip=True,
            num_classes=2,
            amp=False,
        )

        self.assertTrue(model.training)
        self.assertEqual(result["samples"], 1)
        self.assertEqual(result["valid_pixel_count"], 3)
        self.assertEqual(result["tta"]["predictions_per_image"], 2)
        self.assertEqual(result["tta"]["aggregation"], "mean_logits")

    def test_tta_rejects_empty_or_duplicate_scales(self) -> None:
        image = torch.zeros(1, 3, 2, 2)
        with self.assertRaises(ValueError):
            predict_tta_logits(
                nn.Identity(), image, scales=(), horizontal_flip=False, amp_enabled=False
            )
        with self.assertRaises(ValueError):
            predict_tta_logits(
                nn.Identity(),
                image,
                scales=(1.0, 1.0),
                horizontal_flip=False,
                amp_enabled=False,
            )


if __name__ == "__main__":
    unittest.main()
