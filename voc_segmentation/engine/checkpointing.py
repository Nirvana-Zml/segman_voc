"""Checkpoint save/load helpers with hashes and diagnostic metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    step: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    diagnostic_only: bool,
) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    payload = {
        "format_version": 1,
        "step": int(step),
        "diagnostic_only": bool(diagnostic_only),
        "config": config,
        "metrics": metrics,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }
    torch.save(payload, temporary)
    temporary.replace(target)
    return {
        "path": str(target.resolve()),
        "bytes": target.stat().st_size,
        "sha256": file_sha256(target),
        "step": int(step),
        "diagnostic_only": bool(diagnostic_only),
    }


def load_checkpoint(
    path: Path | str,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location)
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format_version')}")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    return payload
