#!/usr/bin/env python3
"""Train DeepLabV3+ and select checkpoints on official full-resolution val."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from datasets import (  # noqa: E402
    SegmentationEvalTransform,
    SegmentationTrainTransform,
    VOCSegmentationDataset,
)
from engine import (  # noqa: E402
    evaluate_segmentation,
    file_sha256,
    load_checkpoint,
    poly_lr_factor,
    save_checkpoint,
    set_learning_rate,
)
from models.baselines import build_deeplabv3plus  # noqa: E402


CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")
EXPECTED_WEIGHT_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/deeplabv3plus_r50.yaml")
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path("/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/deeplabv3plus_r50_voc_train")
    )
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--validate-every", type=int)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--git-commit")
    parser.add_argument("--git-worktree-clean", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def resolve_app_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (APP_ROOT / path).resolve()


def verify_pretrained_weight(config: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    weight_path = resolve_app_path(Path(config["model"]["encoder_weights_path"]))
    provenance_path = weight_path.parent / "provenance.json"
    if not weight_path.is_file() or not provenance_path.is_file():
        raise FileNotFoundError(
            "pretrained weight or provenance is missing; run "
            "tools/acquire_resnet50_weights.py first"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    actual_sha256 = file_sha256(weight_path)
    if not provenance.get("ok"):
        raise RuntimeError("pretrained weight provenance is not marked ok")
    if actual_sha256 != EXPECTED_WEIGHT_SHA256:
        raise RuntimeError(f"unexpected pretrained weight SHA-256: {actual_sha256}")
    if provenance.get("sha256") != actual_sha256:
        raise RuntimeError("pretrained weight does not match provenance SHA-256")
    return weight_path, provenance


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.validation_limit is not None and not args.diagnostic_only:
        raise ValueError("--validation-limit is allowed only with --diagnostic-only")
    if args.validation_limit is not None and args.validation_limit <= 0:
        raise ValueError("--validation-limit must be positive")

    config_path = resolve_app_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    max_iterations = int(
        training["max_iterations"]
        if args.max_iterations is None
        else args.max_iterations
    )
    validate_every = int(
        training["validate_every"]
        if args.validate_every is None
        else args.validate_every
    )
    if max_iterations <= 0 or validate_every <= 0:
        raise ValueError("iteration counts must be positive")
    accumulation_steps = int(training["gradient_accumulation_steps"])
    physical_batch_size = int(training["physical_batch_size"])
    if accumulation_steps <= 0 or physical_batch_size <= 0:
        raise ValueError("batch size and accumulation steps must be positive")
    if physical_batch_size * accumulation_steps != int(training["effective_batch_size"]):
        raise ValueError("configured effective batch size is inconsistent")
    if args.git_commit is not None and not re.fullmatch(
        r"[0-9a-fA-F]{40}", args.git_commit
    ):
        raise ValueError("--git-commit must be a full 40-character hexadecimal SHA")
    if not args.diagnostic_only and (
        args.git_commit is None or not args.git_worktree_clean
    ):
        raise RuntimeError(
            "reportable training requires --git-commit and --git-worktree-clean; "
            "verify both on the host before docker exec"
        )
    git_state = {
        "commit": args.git_commit.lower() if args.git_commit is not None else None,
        "worktree_clean_asserted": bool(args.git_worktree_clean),
        "capture_method": "explicit host injection",
    }

    output_dir = resolve_app_path(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists() and args.resume is None:
        raise FileExistsError(
            f"refusing to overwrite completed run: {result_path}; choose a new output"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    run_timestamp = datetime.now(CHINA_TIME)
    tensorboard_dir = output_dir / "tensorboard" / run_timestamp.strftime(
        "%Y%m%dT%H%M%S%z"
    )
    history_path = output_dir / "history.jsonl"
    resolved_config_path = output_dir / "resolved_config.yaml"

    resolved_config = config.copy()
    resolved_config["runtime"] = {
        "max_iterations": max_iterations,
        "validate_every": validate_every,
        "validation_limit": args.validation_limit,
        "diagnostic_only": args.diagnostic_only,
        "device": args.device,
    }
    resolved_config_path.write_text(
        yaml.safe_dump(resolved_config, sort_keys=False), encoding="utf-8"
    )

    seed = int(training["seed"])
    set_seed(seed)
    device = torch.device(args.device)
    weight_path, weight_provenance = verify_pretrained_weight(config)
    crop_size = tuple(int(value) for value in config["data"]["crop_size"])
    scale_range = tuple(float(value) for value in config["data"]["scale_range"])
    train_dataset = VOCSegmentationDataset(
        args.voc_root,
        config["data"]["train_split"],
        SegmentationTrainTransform(
            crop_size=crop_size,
            scale_range=scale_range,
            horizontal_flip_probability=float(
                config["data"]["horizontal_flip_probability"]
            ),
        ),
    )
    val_dataset = VOCSegmentationDataset(
        args.voc_root,
        config["data"]["val_split"],
        SegmentationEvalTransform(),
    )
    train_split_path = (
        args.voc_root
        / "ImageSets/Segmentation"
        / f"{config['data']['train_split']}.txt"
    )
    val_split_path = (
        args.voc_root / "ImageSets/Segmentation" / f"{config['data']['val_split']}.txt"
    )
    workers = int(training["num_workers"])
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=physical_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )

    model = build_deeplabv3plus(
        num_classes=int(config["model"]["num_classes"]),
        encoder_pretrained=False,
        encoder_weights_path=weight_path,
    ).to(device)
    encoder_parameters = [
        parameter
        for module in (model.stem, model.layer1, model.layer2, model.layer3, model.layer4)
        for parameter in module.parameters()
    ]
    decoder_parameters = [
        parameter
        for module in (model.aspp, model.low_level_projection, model.decoder)
        for parameter in module.parameters()
    ]
    model_parameter_ids = {id(parameter) for parameter in model.parameters()}
    grouped_parameter_ids = [
        id(parameter) for parameter in (*encoder_parameters, *decoder_parameters)
    ]
    if len(grouped_parameter_ids) != len(set(grouped_parameter_ids)):
        raise RuntimeError("optimizer parameter groups overlap")
    if set(grouped_parameter_ids) != model_parameter_ids:
        raise RuntimeError("optimizer parameter groups do not cover the model exactly")
    base_lr = float(training["base_lr"])
    optimizer = torch.optim.SGD(
        [
            {
                "params": encoder_parameters,
                "lr": base_lr * float(training["encoder_lr_multiplier"]),
                "name": "encoder",
            },
            {
                "params": decoder_parameters,
                "lr": base_lr * float(training["decoder_lr_multiplier"]),
                "name": "decoder",
            },
        ],
        lr=base_lr,
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
    )
    amp_enabled = bool(training["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled, init_scale=1024.0)
    criterion = nn.CrossEntropyLoss(ignore_index=int(config["data"]["ignore_index"]))

    start_step = 0
    best_miou = float("-inf")
    best_step = 0
    resume_info: dict[str, Any] | None = None
    if args.resume is not None:
        resume_path = resolve_app_path(args.resume)
        payload = load_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
        )
        if bool(payload["diagnostic_only"]) != bool(args.diagnostic_only):
            raise ValueError("resume checkpoint diagnostic status does not match this run")
        start_step = int(payload["step"])
        selection = payload.get("metrics", {}).get("selection", {})
        best_miou = float(selection.get("best_miou", payload["metrics"].get("miou", -math.inf)))
        best_step = int(selection.get("best_step", start_step))
        resume_info = {
            "path": str(resume_path),
            "sha256": file_sha256(resume_path),
            "step": start_step,
        }
    if start_step >= max_iterations:
        raise ValueError("resume checkpoint is already at or beyond max iterations")

    warmup_steps = min(int(training["warmup_iterations"]), max_iterations - 1)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    history_mode = "a" if args.resume is not None and history_path.exists() else "w"
    train_iterator = iter(train_loader)
    latest_validation: dict[str, Any] | None = None
    best_checkpoint: dict[str, Any] | None = None
    latest_checkpoint: dict[str, Any] | None = None
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    with history_path.open(history_mode, encoding="utf-8", newline="\n") as stream:
        for step in range(start_step + 1, max_iterations + 1):
            factor = poly_lr_factor(
                step,
                max_iterations,
                warmup_steps=warmup_steps,
                power=float(training["poly_power"]),
            )
            learning_rates = set_learning_rate(optimizer, factor)
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for _ in range(accumulation_steps):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    logits = model(images)
                    micro_loss = criterion(logits, masks)
                    backward_loss = micro_loss / accumulation_steps
                if not torch.isfinite(micro_loss):
                    raise RuntimeError(f"non-finite train loss at iteration {step}")
                accumulated_loss += float(micro_loss.detach())
                scaler.scale(backward_loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            gradient_norm_value = float(gradient_norm)
            if not math.isfinite(gradient_norm_value):
                raise RuntimeError(f"non-finite gradient norm at iteration {step}")
            scaler.step(optimizer)
            scaler.update()

            train_loss = accumulated_loss / accumulation_steps
            train_entry = {
                "type": "train",
                "step": step,
                "loss": train_loss,
                "gradient_norm": gradient_norm_value,
                "amp_scale": float(scaler.get_scale()),
                "learning_rates": learning_rates,
            }
            stream.write(json.dumps(train_entry) + "\n")
            stream.flush()
            writer.add_scalar("Loss/train_ce", train_loss, step)
            writer.add_scalar("System/gradient_norm", gradient_norm_value, step)
            writer.add_scalar("LearningRate/encoder", learning_rates[0], step)
            writer.add_scalar("LearningRate/decoder", learning_rates[1], step)
            print(
                f"iteration={step}/{max_iterations} loss={train_loss:.6f} "
                f"grad={gradient_norm_value:.6f} "
                f"encoder_lr={learning_rates[0]:.8f} decoder_lr={learning_rates[1]:.8f}",
                flush=True,
            )

            should_validate = step % validate_every == 0 or step == max_iterations
            if not should_validate:
                continue
            latest_validation = evaluate_segmentation(
                model,
                val_loader,
                device,
                num_classes=int(config["model"]["num_classes"]),
                ignore_index=int(config["data"]["ignore_index"]),
                amp=amp_enabled,
                max_batches=args.validation_limit,
            )
            if args.validation_limit is None and any(
                value is None for value in latest_validation["per_class_iou"]
            ):
                raise RuntimeError(
                    "full official validation did not contain all 21 class unions"
                )
            validation_path = validation_dir / f"step_{step:06d}.json"
            write_json(validation_path, latest_validation)
            validation_entry = {"type": "validation", "step": step, **latest_validation}
            stream.write(json.dumps(validation_entry) + "\n")
            stream.flush()
            writer.add_scalar("Loss/val_ce", latest_validation["loss"], step)
            writer.add_scalar("Metrics/val_miou", latest_validation["miou"], step)
            writer.add_scalar(
                "Metrics/val_pixel_accuracy",
                latest_validation["pixel_accuracy"],
                step,
            )

            candidate_miou = float(latest_validation["miou"])
            improved = math.isfinite(candidate_miou) and candidate_miou > best_miou
            if improved:
                best_miou = candidate_miou
                best_step = step
            checkpoint_metrics = {
                **latest_validation,
                "selection": {"best_miou": best_miou, "best_step": best_step},
            }
            checkpoint_name = "best.pt" if improved else "last.pt"
            latest_checkpoint = save_checkpoint(
                output_dir / checkpoint_name,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                config=resolved_config,
                metrics=checkpoint_metrics,
                diagnostic_only=args.diagnostic_only,
            )
            if improved:
                best_checkpoint = latest_checkpoint
            print(
                f"validation step={step} miou={candidate_miou:.6f} "
                f"best={best_miou:.6f}@{best_step}",
                flush=True,
            )

    writer.flush()
    writer.close()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - started
    if latest_validation is None or best_step == 0:
        raise RuntimeError("training completed without a valid checkpoint selection")
    if best_checkpoint is None:
        best_path = output_dir / "best.pt"
        if not best_path.is_file():
            raise RuntimeError("resume run has no best checkpoint")
        best_checkpoint = {
            "path": str(best_path),
            "bytes": best_path.stat().st_size,
            "sha256": file_sha256(best_path),
            "step": best_step,
            "diagnostic_only": args.diagnostic_only,
        }

    event_files = sorted(tensorboard_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError("TensorBoard did not create an event file")
    accumulator = EventAccumulator(str(tensorboard_dir))
    accumulator.Reload()
    event_info = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in event_files
    ]
    result: dict[str, Any] = {
        "ok": True,
        "diagnostic_only": args.diagnostic_only,
        "timestamp": run_timestamp.isoformat(timespec="seconds"),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "resolved_config": str(resolved_config_path),
        "resolved_config_sha256": file_sha256(resolved_config_path),
        "seed": seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "git": git_state,
        "train_split": config["data"]["train_split"],
        "train_samples": len(train_dataset),
        "train_split_file": str(train_split_path.resolve()),
        "train_split_sha256": file_sha256(train_split_path),
        "val_split": config["data"]["val_split"],
        "val_samples": len(val_dataset),
        "val_split_file": str(val_split_path.resolve()),
        "val_split_sha256": file_sha256(val_split_path),
        "validation_limit": args.validation_limit,
        "validation_is_full_official_split": args.validation_limit is None,
        "start_step": start_step,
        "max_iterations": max_iterations,
        "validate_every": validate_every,
        "physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": physical_batch_size * accumulation_steps,
        "amp": amp_enabled,
        "pretrained_weight": {
            "path": str(weight_path),
            "sha256": file_sha256(weight_path),
            "weight_id": weight_provenance["weight_id"],
            "source_url": weight_provenance["source_url"],
        },
        "resume": resume_info,
        "best_miou": best_miou,
        "best_step": best_step,
        "best_checkpoint": best_checkpoint,
        "latest_checkpoint": latest_checkpoint,
        "latest_validation": latest_validation,
        "history": str(history_path),
        "history_sha256": file_sha256(history_path),
        "tensorboard": {
            "directory": str(tensorboard_dir),
            "events": event_info,
            "scalar_tags": sorted(accumulator.Tags().get("scalars", [])),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
