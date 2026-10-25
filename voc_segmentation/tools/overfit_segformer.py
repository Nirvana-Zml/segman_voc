#!/usr/bin/env python3
"""Overfit pretrained SegFormer on two fixed VOC samples as a diagnostic gate."""

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
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from datasets import SegmentationFixedResizeTransform, VOCSegmentationDataset  # noqa: E402
from engine import file_sha256, save_checkpoint  # noqa: E402
from metrics import SegmentationMetrics  # noqa: E402
from models.baselines import build_segformer  # noqa: E402


CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")
DEFAULT_SAMPLE_IDS = ("2008_005679", "2008_002175")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/segformer_b1.yaml"))
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path("/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/transformer_baseline/overfit_two_samples"),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-4)
    parser.add_argument("--decoder-learning-rate", type=float, default=1e-3)
    parser.add_argument("--required-loss-ratio", type=float, default=0.5)
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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    if args.steps < 2:
        raise ValueError("steps must be at least 2")
    if args.size < 32:
        raise ValueError("size must be at least 32")
    if args.encoder_learning_rate <= 0.0 or args.decoder_learning_rate <= 0.0:
        raise ValueError("learning rates must be positive")
    if not 0.0 < args.required_loss_ratio < 1.0:
        raise ValueError("required-loss-ratio must be between 0 and 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["training"]["seed"])
    set_seed(seed)
    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now(CHINA_TIME)
    tensorboard_dir = output_dir / "tensorboard" / run_timestamp.strftime(
        "%Y%m%dT%H%M%S%z"
    )
    history_path = output_dir / "history.jsonl"
    checkpoint_path = output_dir / "diagnostic.pt"
    result_path = output_dir / "result.json"

    weight_path = resolve_app_path(Path(config["model"]["encoder_weights_path"]))
    provenance_path = weight_path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    weight_sha256 = file_sha256(weight_path)
    if provenance.get("ok") is not True:
        raise RuntimeError("MiT-B1 provenance is not accepted")
    if weight_sha256 != provenance.get("converted_sha256"):
        raise RuntimeError("MiT-B1 checkpoint hash does not match provenance")

    transform = SegmentationFixedResizeTransform((args.size, args.size))
    dataset = VOCSegmentationDataset(args.voc_root, "train", transform)
    index_by_id = {sample.sample_id: index for index, sample in enumerate(dataset.samples)}
    missing = [sample_id for sample_id in DEFAULT_SAMPLE_IDS if sample_id not in index_by_id]
    if missing:
        raise ValueError(f"missing diagnostic sample IDs: {missing}")
    subset = Subset(dataset, [index_by_id[sample_id] for sample_id in DEFAULT_SAMPLE_IDS])
    loader = DataLoader(subset, batch_size=2, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

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
                "lr": args.encoder_learning_rate,
            },
            {
                "name": "decoder",
                "params": decoder_parameters,
                "lr": args.decoder_learning_rate,
            },
        ],
        weight_decay=0.0,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=int(config["data"]["ignore_index"]))
    amp_enabled = bool(config["training"]["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled, init_scale=1024.0)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    history: list[dict[str, float | int]] = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with history_path.open("w", encoding="utf-8", newline="\n") as history_stream:
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {float(loss)}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            gradient_norm_value = float(gradient_norm)
            if not math.isfinite(gradient_norm_value):
                raise RuntimeError(
                    f"non-finite gradient norm at step {step}: {gradient_norm_value}"
                )
            scaler.step(optimizer)
            scaler.update()

            entry: dict[str, float | int] = {
                "step": step,
                "loss": float(loss.detach()),
                "gradient_norm": gradient_norm_value,
                "amp_scale": float(scaler.get_scale()),
                "encoder_learning_rate": float(optimizer.param_groups[0]["lr"]),
                "decoder_learning_rate": float(optimizer.param_groups[1]["lr"]),
            }
            history.append(entry)
            history_stream.write(json.dumps(entry) + "\n")
            writer.add_scalar("Loss/train_ce", entry["loss"], step)
            writer.add_scalar("System/gradient_norm", entry["gradient_norm"], step)
            writer.add_scalar(
                "LearningRate/encoder", entry["encoder_learning_rate"], step
            )
            writer.add_scalar(
                "LearningRate/decoder", entry["decoder_learning_rate"], step
            )
            if step == 1 or step % 10 == 0 or step == args.steps:
                print(
                    f"step={step}/{args.steps} loss={entry['loss']:.6f} "
                    f"grad={entry['gradient_norm']:.6f} scale={entry['amp_scale']:.1f}",
                    flush=True,
                )
    writer.flush()
    writer.close()

    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
        final_logits = model(images)
        final_loss = criterion(final_logits, masks)
    metric = SegmentationMetrics(
        num_classes=int(config["model"]["num_classes"]),
        ignore_index=int(config["data"]["ignore_index"]),
    )
    metric.update(final_logits, masks)
    metric_result = metric.compute()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started

    initial_loss = float(history[0]["loss"])
    final_loss_value = float(final_loss)
    loss_ratio = final_loss_value / initial_loss
    finite_history = all(
        math.isfinite(float(item["loss"]))
        and math.isfinite(float(item["gradient_norm"]))
        for item in history
    )
    ok = bool(finite_history and loss_ratio <= args.required_loss_ratio)
    checkpoint_info = save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        step=args.steps,
        config=config,
        metrics={
            "initial_loss": initial_loss,
            "final_loss": final_loss_value,
            "loss_ratio": loss_ratio,
            **metric_result,
        },
        diagnostic_only=True,
    )

    event_files = sorted(tensorboard_dir.glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise RuntimeError(f"expected one TensorBoard event file, found {len(event_files)}")
    accumulator = EventAccumulator(str(tensorboard_dir))
    accumulator.Reload()
    scalar_tags = sorted(accumulator.Tags().get("scalars", []))
    result: dict[str, Any] = {
        "ok": ok,
        "diagnostic_only": True,
        "timestamp": run_timestamp.isoformat(timespec="seconds"),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "seed": seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "sample_ids": list(DEFAULT_SAMPLE_IDS),
        "input_shape": list(images.shape),
        "steps": args.steps,
        "optimizer": "AdamW",
        "optimizer_groups": [
            {"name": group["name"], "lr": float(group["lr"])}
            for group in optimizer.param_groups
        ],
        "weight_decay": 0.0,
        "amp": amp_enabled,
        "encoder_pretrained": True,
        "encoder_weight_path": str(weight_path),
        "encoder_weight_sha256": weight_sha256,
        "encoder_weight_provenance": str(provenance_path),
        "encoder_weight_provenance_sha256": file_sha256(provenance_path),
        "initial_loss": initial_loss,
        "final_loss": final_loss_value,
        "loss_ratio": loss_ratio,
        "required_loss_ratio": args.required_loss_ratio,
        "metrics": metric_result,
        "history": str(history_path),
        "history_sha256": file_sha256(history_path),
        "checkpoint": checkpoint_info,
        "tensorboard_event": {
            "path": str(event_files[0]),
            "bytes": event_files[0].stat().st_size,
            "sha256": file_sha256(event_files[0]),
            "scalar_tags": scalar_tags,
        },
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }
    write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
