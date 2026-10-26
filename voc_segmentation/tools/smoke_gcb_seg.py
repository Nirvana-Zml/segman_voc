#!/usr/bin/env python3
"""Run one diagnostic pretrained GCB-Seg optimization step on VOC train."""

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
from torch.utils.data import DataLoader


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from datasets import SegmentationTrainTransform, VOCSegmentationDataset  # noqa: E402
from engine import file_sha256  # noqa: E402
from losses import GCBSegLoss, semantic_boundary_targets  # noqa: E402
from models import build_gcb_seg  # noqa: E402


CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")
EXPECTED_WEIGHT_SHA256 = (
    "8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gcb_seg_b1_384.yaml")
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path("/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/gcb_diagnostics/smoke.json"),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_app_path(path: Path) -> Path:
    return path if path.is_absolute() else APP_ROOT / path


def main() -> int:
    args = parse_args()
    config_path = resolve_app_path(args.config).resolve()
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
    if weight_sha256 != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"unexpected MiT-B1 checkpoint SHA-256: {weight_sha256}")
    if weight_sha256 != provenance.get("converted_sha256"):
        raise RuntimeError("MiT-B1 checkpoint hash does not match provenance")

    crop_size = tuple(int(value) for value in config["data"]["crop_size"])
    transform = SegmentationTrainTransform(
        crop_size=crop_size,
        scale_range=tuple(float(value) for value in config["data"]["random_scale"]),
        horizontal_flip_probability=float(config["data"]["horizontal_flip"]),
    )
    dataset = VOCSegmentationDataset(
        args.voc_root, config["data"]["train_split"], transform
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["physical_batch_size"]),
        shuffle=True,
        num_workers=int(config["training"]["num_workers"]),
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )

    model_config = config["model"]
    model = build_gcb_seg(
        num_classes=int(model_config["num_classes"]),
        encoder=str(model_config["encoder"]),
        decoder_channels=int(model_config["decoder_channels"]),
        context_branch_channels=int(model_config["context_branch_channels"]),
        boundary_channels=int(model_config["boundary_channels"]),
        global_context_dilations=tuple(
            int(value) for value in model_config["global_context_dilations"]
        ),
        use_global_pooling=bool(model_config["use_global_pooling"]),
        use_gated_fusion=bool(model_config["use_gated_fusion"]),
        feature_fusion_mode=str(model_config.get("feature_fusion_mode", "gated")),
        use_boundary_branch=bool(model_config["use_boundary_branch"]),
        auxiliary_head=bool(model_config["auxiliary_head"]),
        dropout=float(model_config["dropout"]),
        encoder_weights_path=weight_path,
    ).to(device)
    model.train()
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in encoder_ids
    ]
    decoder_ids = {id(parameter) for parameter in decoder_parameters}
    model_ids = {id(parameter) for parameter in model.parameters()}
    if encoder_ids & decoder_ids:
        raise RuntimeError("encoder and decoder optimizer parameter groups overlap")
    if encoder_ids | decoder_ids != model_ids:
        raise RuntimeError("optimizer parameter groups do not cover the model exactly")
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "encoder",
                "params": encoder_parameters,
                "lr": float(optimizer_config["encoder_lr"]),
            },
            {
                "name": "decoder",
                "params": decoder_parameters,
                "lr": float(optimizer_config["decoder_lr"]),
            },
        ],
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    loss_config = config["loss"]
    criterion = GCBSegLoss(
        ignore_index=int(config["data"]["ignore_index"]),
        ce_weight=float(loss_config["ce_weight"]),
        lovasz_weight=float(loss_config["lovasz_weight"]),
        edge_weight=float(loss_config["edge_weight"]),
        aux_weight=float(loss_config["aux_weight"]),
        label_smoothing=float(loss_config["label_smoothing"]),
    )
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
        outputs = model(images)
        if not isinstance(outputs, dict):
            raise RuntimeError("training-mode GCB-Seg did not return an output mapping")
        losses = criterion(outputs, masks)
    if not all(torch.isfinite(value) for value in losses.values()):
        raise RuntimeError("GCB-Seg smoke produced a non-finite loss component")
    scaler.scale(losses["total"]).backward()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(optimizer_config["grad_clip"])
    )
    gradient_norm_value = float(gradient_norm)
    if not math.isfinite(gradient_norm_value):
        raise RuntimeError(f"non-finite gradient norm: {gradient_norm_value}")
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started

    logits = outputs["logits"]
    aux_logits = outputs["aux_logits"]
    edge_logits = outputs["edge_logits"]
    if logits is None or aux_logits is None or edge_logits is None:
        raise RuntimeError("full GCB-Seg smoke requires main, auxiliary, and edge heads")
    prediction = logits.detach().argmax(dim=1)
    ignore_index = int(config["data"]["ignore_index"])
    valid_mask = masks != ignore_index
    edge_targets, edge_valid = semantic_boundary_targets(
        masks, ignore_index=ignore_index
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
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
        "output_shapes": {
            "logits": list(logits.shape),
            "aux_logits": list(aux_logits.shape),
            "edge_logits": list(edge_logits.shape),
        },
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "gradient_norm": gradient_norm_value,
        "valid_pixel_count": int(valid_mask.sum()),
        "boundary_valid_pixel_count": int(edge_valid.sum()),
        "boundary_positive_pixel_count": int(edge_targets.sum()),
        "prediction_class_ids": sorted(int(value) for value in prediction.unique()),
        "target_class_ids": sorted(int(value) for value in masks[valid_mask].unique()),
        "amp": amp_enabled,
        "amp_initial_scale": amp_initial_scale if amp_enabled else None,
        "amp_final_scale": float(scaler.get_scale()) if amp_enabled else None,
        "encoder_pretrained": True,
        "encoder_weight_path": str(weight_path),
        "encoder_weight_sha256": weight_sha256,
        "encoder_weight_provenance": str(provenance_path),
        "encoder_weight_provenance_sha256": file_sha256(provenance_path),
        "parameter_count": parameter_count,
        "encoder_parameter_count": sum(
            parameter.numel() for parameter in encoder_parameters
        ),
        "decoder_parameter_count": sum(
            parameter.numel() for parameter in decoder_parameters
        ),
        "optimizer_groups": [
            {
                "name": group["name"],
                "lr": float(group["lr"]),
                "parameter_count": sum(
                    parameter.numel() for parameter in group["params"]
                ),
            }
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
    output_path = resolve_app_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
