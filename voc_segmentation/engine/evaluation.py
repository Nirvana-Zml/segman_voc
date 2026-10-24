"""Full-resolution semantic-segmentation validation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from metrics import SegmentationMetrics


@torch.no_grad()
def evaluate_segmentation(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    num_classes: int = 21,
    ignore_index: int = 255,
    amp: bool = True,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate native-size images using one dataset-level confusion matrix."""

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    was_training = model.training
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    loss_sum = 0.0
    valid_pixel_count = 0
    sample_count = 0
    batch_count = 0
    amp_enabled = bool(amp and device.type == "cuda")

    try:
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits: Tensor = model(images)
                batch_loss_sum = F.cross_entropy(
                    logits,
                    masks,
                    ignore_index=ignore_index,
                    reduction="sum",
                )
            if not torch.isfinite(batch_loss_sum):
                raise RuntimeError("validation produced a non-finite loss")
            valid_pixels = int((masks != ignore_index).sum())
            loss_sum += float(batch_loss_sum)
            valid_pixel_count += valid_pixels
            sample_count += int(images.shape[0])
            batch_count += 1
            metrics.update(logits, masks)
            if max_batches is not None and batch_count >= max_batches:
                break
    finally:
        model.train(was_training)

    if batch_count == 0 or valid_pixel_count == 0:
        raise RuntimeError("validation loader produced no valid pixels")
    result = metrics.compute()
    validation_loss = loss_sum / valid_pixel_count
    if not math.isfinite(validation_loss):
        raise RuntimeError("validation mean loss is non-finite")
    result.update(
        {
            "loss": validation_loss,
            "samples": sample_count,
            "batches": batch_count,
            "full_dataset": max_batches is None,
        }
    )
    return result
