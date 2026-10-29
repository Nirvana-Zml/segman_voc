"""Full-resolution semantic-segmentation validation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from metrics import SegmentationMetrics


def _validate_tta_scales(scales: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(scale) for scale in scales)
    if not values:
        raise ValueError("TTA scales must not be empty")
    if any(not math.isfinite(scale) or scale <= 0 for scale in values):
        raise ValueError("TTA scales must be finite and positive")
    if len(values) != len(set(values)):
        raise ValueError("TTA scales must be unique")
    return values


def _main_logits(output: Tensor | dict[str, Tensor | None]) -> Tensor:
    if isinstance(output, Tensor):
        return output
    logits = output.get("logits")
    if not isinstance(logits, Tensor):
        raise TypeError("model output does not contain main logits")
    return logits


@torch.no_grad()
def predict_tta_logits(
    model: nn.Module,
    images: Tensor,
    *,
    scales: Sequence[float],
    horizontal_flip: bool,
    amp_enabled: bool,
) -> Tensor:
    """Average native-size logits over scales and optional horizontal flips."""

    scale_values = _validate_tta_scales(scales)
    if images.ndim != 4:
        raise ValueError("images must have shape [N, C, H, W]")
    original_size = images.shape[-2:]
    logits_sum: Tensor | None = None
    prediction_count = 0
    for scale in scale_values:
        scaled_size = (
            max(1, round(original_size[0] * scale)),
            max(1, round(original_size[1] * scale)),
        )
        scaled_images = (
            images
            if scaled_size == original_size
            else F.interpolate(
                images,
                size=scaled_size,
                mode="bilinear",
                align_corners=False,
            )
        )
        variants = ((scaled_images, False),)
        if horizontal_flip:
            variants += ((torch.flip(scaled_images, dims=(-1,)), True),)
        for variant, flipped in variants:
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = _main_logits(model(variant))
            if flipped:
                logits = torch.flip(logits, dims=(-1,))
            if logits.shape[-2:] != original_size:
                logits = F.interpolate(
                    logits,
                    size=original_size,
                    mode="bilinear",
                    align_corners=False,
                )
            logits = logits.float()
            logits_sum = logits if logits_sum is None else logits_sum + logits
            prediction_count += 1
    if logits_sum is None or prediction_count == 0:
        raise RuntimeError("TTA produced no logits")
    return logits_sum / prediction_count


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


@torch.no_grad()
def evaluate_segmentation_tta(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    scales: Sequence[float],
    horizontal_flip: bool,
    num_classes: int = 21,
    ignore_index: int = 255,
    amp: bool = True,
    max_batches: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Evaluate logit-averaged multi-scale/flip TTA at native resolution."""

    scale_values = _validate_tta_scales(scales)
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
            logits = predict_tta_logits(
                model,
                images,
                scales=scale_values,
                horizontal_flip=horizontal_flip,
                amp_enabled=amp_enabled,
            )
            batch_loss_sum = F.cross_entropy(
                logits,
                masks,
                ignore_index=ignore_index,
                reduction="sum",
            )
            if not torch.isfinite(batch_loss_sum):
                raise RuntimeError("TTA validation produced a non-finite loss")
            valid_pixels = int((masks != ignore_index).sum())
            loss_sum += float(batch_loss_sum)
            valid_pixel_count += valid_pixels
            sample_count += int(images.shape[0])
            batch_count += 1
            metrics.update(logits, masks)
            if progress_callback is not None:
                progress_callback(batch_count, sample_count)
            if max_batches is not None and batch_count >= max_batches:
                break
    finally:
        model.train(was_training)

    if batch_count == 0 or valid_pixel_count == 0:
        raise RuntimeError("TTA validation loader produced no valid pixels")
    result = metrics.compute()
    validation_loss = loss_sum / valid_pixel_count
    if not math.isfinite(validation_loss):
        raise RuntimeError("TTA validation mean loss is non-finite")
    result.update(
        {
            "loss": validation_loss,
            "samples": sample_count,
            "batches": batch_count,
            "full_dataset": max_batches is None,
            "tta": {
                "scales": list(scale_values),
                "horizontal_flip": bool(horizontal_flip),
                "aggregation": "mean_logits",
                "predictions_per_image": len(scale_values)
                * (2 if horizontal_flip else 1),
                "resize_mode": "bilinear",
                "align_corners": False,
            },
        }
    )
    return result
