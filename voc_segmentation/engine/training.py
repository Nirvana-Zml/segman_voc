"""Small, testable helpers shared by formal segmentation trainers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from torch.optim import Optimizer


def reconcile_history_for_resume(
    history_path: Path | str,
    *,
    start_step: int,
    archive_path: Path | str,
) -> dict[str, Any]:
    """Archive history and remove records beyond a restart checkpoint."""

    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    source = Path(history_path)
    archive = Path(archive_path)
    if not source.is_file():
        return {
            "history_existed": False,
            "original_records": 0,
            "kept_records": 0,
            "dropped_records": 0,
            "archive": None,
        }

    original_text = source.read_text(encoding="utf-8")
    lines = [line for line in original_text.splitlines() if line.strip()]
    kept_lines: list[str] = []
    seen_completed: set[tuple[str, int]] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
            record_type = str(record["type"])
            step = int(record["step"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"invalid history record at line {line_number}: {error}"
            ) from error
        if step > start_step:
            continue
        if record_type in {"train", "validation"}:
            key = (record_type, step)
            if key in seen_completed:
                raise ValueError(
                    f"duplicate completed history record before checkpoint: {key}"
                )
            seen_completed.add(key)
        kept_lines.append(line)

    dropped_records = len(lines) - len(kept_lines)
    archive_value: str | None = None
    if dropped_records:
        if archive.exists():
            raise FileExistsError(f"refusing to overwrite history archive: {archive}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(original_text, encoding="utf-8", newline="\n")
        temporary = source.with_suffix(source.suffix + ".reconcile.tmp")
        temporary.write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else ""),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(source)
        archive_value = str(archive.resolve())

    return {
        "history_existed": True,
        "original_records": len(lines),
        "kept_records": len(kept_lines),
        "dropped_records": dropped_records,
        "archive": archive_value,
    }


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
