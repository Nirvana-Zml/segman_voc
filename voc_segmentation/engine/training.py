"""Small, testable helpers shared by formal segmentation trainers."""

from __future__ import annotations

import math

from torch.optim import Optimizer


def poly_lr_factor(
    step: int,
    max_steps: int,
    *,
    warmup_steps: int = 0,
    power: float = 0.9,
) -> float:
    """Return linear-warmup then polynomial-decay learning-rate multiplier."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if step < 1 or step > max_steps:
        raise ValueError("step must be in [1, max_steps]")
    if warmup_steps < 0 or warmup_steps >= max_steps:
        raise ValueError("warmup_steps must be in [0, max_steps)")
    if power <= 0:
        raise ValueError("power must be positive")
    if max_steps == 1:
        return 1.0
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    decay_steps = max_steps - warmup_steps
    decay_position = step - warmup_steps
    return math.pow(max(0.0, 1.0 - decay_position / decay_steps), power)


def set_learning_rate(optimizer: Optimizer, factor: float) -> list[float]:
    """Apply a multiplier to each group's immutable ``initial_lr``."""

    if factor < 0 or not math.isfinite(factor):
        raise ValueError("learning-rate factor must be finite and non-negative")
    values: list[float] = []
    for group in optimizer.param_groups:
        initial_lr = float(group.setdefault("initial_lr", group["lr"]))
        group["lr"] = initial_lr * factor
        values.append(float(group["lr"]))
    return values


def is_recoverable_amp_overflow(
    gradient_norm: float,
    *,
    amp_enabled: bool,
    scale_before: float,
    scale_after: float,
) -> bool:
    """Classify a non-finite gradient as a verified GradScaler backoff event."""

    if math.isfinite(gradient_norm):
        return False
    if not amp_enabled:
        raise RuntimeError("non-finite gradient without AMP")
    if not math.isfinite(scale_before) or not math.isfinite(scale_after):
        raise RuntimeError("AMP scale is non-finite")
    if scale_before <= 0 or scale_after <= 0:
        raise RuntimeError("AMP scale must remain positive")
    if scale_after >= scale_before:
        raise RuntimeError(
            "non-finite gradient was not accompanied by GradScaler backoff"
        )
    return True
