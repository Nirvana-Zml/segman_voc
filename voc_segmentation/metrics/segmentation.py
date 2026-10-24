"""Dataset-level semantic-segmentation metrics from a confusion matrix."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


class SegmentationMetrics:
    def __init__(self, num_classes: int = 21, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes), dtype=torch.int64
        )

    def reset(self) -> None:
        self.confusion_matrix.zero_()

    @torch.no_grad()
    def update(self, prediction: Tensor, target: Tensor) -> None:
        if prediction.ndim == target.ndim + 1:
            prediction = prediction.argmax(dim=1)
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction/target shape mismatch: {prediction.shape} != {target.shape}"
            )
        prediction = prediction.detach().to(torch.int64).reshape(-1).cpu()
        target = target.detach().to(torch.int64).reshape(-1).cpu()
        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
            & (prediction >= 0)
            & (prediction < self.num_classes)
        )
        indices = target[valid] * self.num_classes + prediction[valid]
        self.confusion_matrix += torch.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, Any]:
        matrix = self.confusion_matrix.to(torch.float64)
        true_positive = matrix.diag()
        target_count = matrix.sum(dim=1)
        prediction_count = matrix.sum(dim=0)
        union = target_count + prediction_count - true_positive
        iou = torch.full((self.num_classes,), float("nan"), dtype=torch.float64)
        present_iou = union > 0
        iou[present_iou] = true_positive[present_iou] / union[present_iou]
        class_accuracy = torch.full_like(iou, float("nan"))
        present_target = target_count > 0
        class_accuracy[present_target] = true_positive[present_target] / target_count[present_target]
        total = matrix.sum()
        return {
            "miou": float(torch.nanmean(iou)),
            "foreground_miou": float(torch.nanmean(iou[1:])),
            "pixel_accuracy": float(true_positive.sum() / total) if total else float("nan"),
            "mean_pixel_accuracy": float(torch.nanmean(class_accuracy)),
            "per_class_iou": [None if torch.isnan(value) else float(value) for value in iou],
            "confusion_matrix": self.confusion_matrix.tolist(),
            "valid_pixel_count": int(total),
        }
