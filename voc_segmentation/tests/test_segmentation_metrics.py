from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from metrics import SegmentationMetrics  # noqa: E402


class SegmentationMetricsTest(unittest.TestCase):
    def test_accumulates_global_confusion_and_ignores_255(self) -> None:
        metric = SegmentationMetrics(num_classes=3, ignore_index=255)
        target = torch.tensor([[[0, 1], [2, 255]]])
        prediction = torch.tensor([[[0, 2], [2, 1]]])

        metric.update(prediction, target)
        result = metric.compute()

        self.assertEqual(result["valid_pixel_count"], 3)
        self.assertEqual(result["confusion_matrix"], [[1, 0, 0], [0, 0, 1], [0, 0, 1]])
        self.assertAlmostEqual(result["pixel_accuracy"], 2 / 3)
        self.assertAlmostEqual(result["per_class_iou"][0], 1.0)
        self.assertAlmostEqual(result["per_class_iou"][1], 0.0)
        self.assertAlmostEqual(result["per_class_iou"][2], 0.5)
        self.assertAlmostEqual(result["miou"], 0.5)


if __name__ == "__main__":
    unittest.main()
