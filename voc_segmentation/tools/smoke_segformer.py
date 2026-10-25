#!/usr/bin/env python3
"""Run one diagnostic pretrained SegFormer optimization step on VOC train."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from datasets import SegmentationTrainTransform, VOCSegmentationDataset  # noqa: E402
from engine import file_sha256  # noqa: E402
from models.baselines import build_segformer  # noqa: E402


CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/segformer_b1.yaml"))
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path("/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/transformer_baseline/smoke.json"),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_app_path(path: Path) -> Path:
    return path if path.is_absolute() else APP_ROOT / path


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["training"]["seed"])
    set_seed(seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)

    weight_path = resolve_app_path(Path(config["model"]["encoder_weights_path"]))
    provenance_path = weight_path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    weight_sha256 = file_sha256(weight_path)
    if provenance.get("ok") is not True:
        raise RuntimeError("MiT-B1 provenance is not accepted")
    if weight_sha256 != provenance.get("converted_sha256"):
        raise RuntimeError("MiT-B1 checkpoint hash does not match provenance")

    crop_size = tuple(int(value) for value in config["data"]["crop_size"])
    transform = SegmentationTrainTransform(
        crop_size=crop_size,
        scale_range=tuple(float(value) for value in config["data"]["scale_range"]),
        horizontal_flip_probability=float(
            config["data"]["horizontal_flip_probability"]
        ),
    )
    dataset = VOCSegmentationDataset(
        args.voc_root, config["data"]["train_split"], transform
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["physical_batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )

    model = build_segformer(
        num_classes=int(config["model"]["num_classes"]),
        encoder=str(config["model"]["encoder"]),
        decoder_channels=int(config["model"]["decoder_channels"]),
        dropout=float(config["model"]["dropout"]),
        encoder_weights_path=weight_path,
    ).to(device)
    model.train()
    encoder_parameters = list(model.encoder.parameters())
    decoder_parameters = list(model.decode_head.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_ids = {id(parameter) for parameter in decoder_parameters}
    if encoder_ids & decoder_ids:
        raise RuntimeError("encoder and decoder optimizer parameter groups overlap")
    if encoder_ids | decoder_ids != {id(parameter) for parameter in model.parameters()}:
        raise RuntimeError("optimizer parameter groups do not cover the model exactly")
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "encoder",
                "params": encoder_parameters,
                "lr": float(config["training"]["encoder_lr"]),
            },
            {
                "name": "decoder",
                "params": decoder_parameters,
                "lr": float(config["training"]["decoder_lr"]),
            },
        ],
        weight_decay=float(config["training"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=int(config["data"]["ignore_index"]))
    amp_enabled = bool(config["training"]["amp"] and device.type == "cuda")
    amp_initial_scale = 1024.0
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled, init_scale=amp_initial_scale
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    batch = next(iter(loader))
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        logits = model(images)
        loss = criterion(logits, masks)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite smoke loss: {float(loss)}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["training"]["gradient_clip_norm"])
    )
    gradient_norm_value = float(gradient_norm)
    if not math.isfinite(gradient_norm_value):
        raise RuntimeError(f"non-finite gradient norm: {gradient_norm_value}")
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started

    prediction = logits.detach().argmax(dim=1)
    valid_mask = masks != int(config["data"]["ignore_index"])
    result: dict[str, Any] = {
        "ok": True,
        "diagnostic_only": True,
        "timestamp": datetime.now(CHINA_TIME).isoformat(timespec="seconds"),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "seed": seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "dataset_count": len(dataset),
        "sample_ids": list(batch["id"]),
        "input_shape": list(images.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach()),
        "gradient_norm": gradient_norm_value,
        "valid_pixel_count": int(valid_mask.sum()),
        "prediction_class_ids": sorted(int(value) for value in prediction.unique()),
        "target_class_ids": sorted(int(value) for value in masks[valid_mask].unique()),
        "amp": amp_enabled,
        "amp_initial_scale": amp_initial_scale if amp_enabled else None,
        "amp_final_scale": float(scaler.get_scale()) if amp_enabled else None,
        "encoder_pretrained": True,
        "encoder_weight_path": str(weight_path),
        "encoder_weight_sha256": weight_sha256,
        "optimizer_groups": [
            {"name": group["name"], "lr": float(group["lr"])}
            for group in optimizer.param_groups
        ],
        "optimizer_step_completed": True,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
