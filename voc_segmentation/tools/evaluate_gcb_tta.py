#!/usr/bin/env python3
"""Evaluate selected GCB-Seg checkpoint with validation-only logit TTA."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision
import yaml
from torch.utils.data import DataLoader


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from datasets import SegmentationEvalTransform, VOCSegmentationDataset  # noqa: E402
from engine import evaluate_segmentation_tta, file_sha256, load_checkpoint  # noqa: E402
from tools.train_segformer import build_model, set_seed, verify_pretrained_weight  # noqa: E402


CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gcb_selected_a2_tta_val.yaml"),
    )
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path("/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/gcb_selected_a2_tta_val"),
    )
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--diagnostic-only", action="store_true")
    parser.add_argument("--git-commit")
    parser.add_argument("--git-worktree-clean", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (APP_ROOT / path).resolve()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.validation_limit is not None and not args.diagnostic_only:
        raise ValueError("--validation-limit is allowed only with --diagnostic-only")
    if args.validation_limit is not None and args.validation_limit <= 0:
        raise ValueError("--validation-limit must be positive")
    if args.git_commit is not None and not re.fullmatch(r"[0-9a-fA-F]{40}", args.git_commit):
        raise ValueError("--git-commit must be a full hexadecimal SHA")
    if not args.diagnostic_only and (args.git_commit is None or not args.git_worktree_clean):
        raise RuntimeError(
            "reportable evaluation requires --git-commit and --git-worktree-clean"
        )

    config_path = resolve(args.config)
    evaluation_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not (
        evaluation_config["model"]["num_classes"] == 21
        and evaluation_config["data"]["ignore_index"] == 255
        and evaluation_config["data"]["reduce_zero_label"] is False
        and evaluation_config["data"]["val_split"] == "val"
        and evaluation_config["data"]["train_split"] != "val"
        and evaluation_config["training"]["save_best_metric"] == "miou"
    ):
        raise RuntimeError("TTA config violates the VOC/model-selection contract")
    output_dir = resolve(args.output_dir)
    result_path = output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite completed TTA result: {result_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_config_path = resolve(Path(evaluation_config["source_config"]["path"]))
    checkpoint_path = resolve(Path(evaluation_config["checkpoint"]["path"]))
    baseline_path = resolve(Path(evaluation_config["baseline"]["validation"]))
    locked_paths = {
        source_config_path: evaluation_config["source_config"]["sha256"],
        checkpoint_path: evaluation_config["checkpoint"]["sha256"],
        baseline_path: evaluation_config["baseline"]["validation_sha256"],
    }
    for path, expected in locked_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {path}: {actual} != {expected}")

    source_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not math.isclose(
        float(baseline["miou"]),
        float(evaluation_config["baseline"]["miou"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise RuntimeError("locked single-scale baseline mIoU does not match")
    if not (
        baseline["full_dataset"] is True
        and baseline["samples"] == 1449
        and baseline["valid_pixel_count"] == 244886857
    ):
        raise RuntimeError("single-scale baseline is not a full official validation")

    split_path = args.voc_root / "ImageSets/Segmentation/val.txt"
    if file_sha256(split_path) != evaluation_config["data"]["split_sha256"]:
        raise RuntimeError("official validation split SHA-256 mismatch")

    set_seed(int(source_config["training"]["seed"]))
    device = torch.device(args.device)
    weight_path, weight_provenance = verify_pretrained_weight(source_config)
    model = build_model(source_config, weight_path).to(device)
    payload = load_checkpoint(checkpoint_path, model=model, map_location=device)
    if payload["diagnostic_only"] is not False:
        raise RuntimeError("TTA checkpoint must be non-diagnostic")
    if int(payload["step"]) != int(evaluation_config["checkpoint"]["step"]):
        raise RuntimeError("TTA checkpoint step mismatch")
    if payload["config"].get("runtime", {}).get("diagnostic_only") is not False:
        raise RuntimeError("checkpoint runtime contract is diagnostic")
    if {key: payload["config"][key] for key in source_config} != source_config:
        raise RuntimeError("checkpoint configuration does not match selected A2 source")

    dataset = VOCSegmentationDataset(
        args.voc_root,
        evaluation_config["data"]["split"],
        SegmentationEvalTransform(),
    )
    if len(dataset) != int(evaluation_config["data"]["samples"]):
        raise RuntimeError("official validation sample count mismatch")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(evaluation_config["runtime"]["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    scales = tuple(float(value) for value in evaluation_config["inference"]["scales"])
    horizontal_flip = bool(evaluation_config["inference"]["horizontal_flip"])
    progress_every = int(evaluation_config["runtime"]["progress_every"])
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def progress(batch_count: int, sample_count: int) -> None:
        if batch_count == 1 or batch_count % progress_every == 0 or batch_count == len(dataset):
            peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            print(
                f"tta_validation batch={batch_count}/{len(dataset)} "
                f"samples={sample_count} elapsed_seconds={time.perf_counter() - started:.3f} "
                f"peak_cuda_memory_bytes={peak}",
                flush=True,
            )

    metrics = evaluate_segmentation_tta(
        model,
        loader,
        device,
        scales=scales,
        horizontal_flip=horizontal_flip,
        num_classes=int(source_config["model"]["num_classes"]),
        ignore_index=int(evaluation_config["data"]["ignore_index"]),
        amp=bool(evaluation_config["runtime"]["amp"]),
        max_batches=args.validation_limit,
        progress_callback=progress,
    )
    elapsed = time.perf_counter() - started
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    baseline_miou = float(baseline["miou"])
    delta = float(metrics["miou"]) - baseline_miou
    full_official = args.validation_limit is None
    accepted = bool(
        full_official
        and delta > float(evaluation_config["selection"]["minimum_improvement"])
    )
    resolved = dict(evaluation_config)
    resolved["execution"] = {
        "device": args.device,
        "diagnostic_only": bool(args.diagnostic_only),
        "validation_limit": args.validation_limit,
    }
    resolved_path = output_dir / "resolved_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8", newline="\n"
    )
    result = {
        "ok": True,
        "diagnostic_only": bool(args.diagnostic_only),
        "timestamp": datetime.now(CHINA_TIME).isoformat(),
        "git": {
            "commit": args.git_commit.lower() if args.git_commit else None,
            "worktree_clean_asserted": bool(args.git_worktree_clean),
            "capture_method": "explicit host injection" if args.git_commit else None,
        },
        "environment": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
        "resolved_config": {
            "path": str(resolved_path),
            "sha256": file_sha256(resolved_path),
        },
        "source_config": {
            "path": str(source_config_path),
            "sha256": file_sha256(source_config_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "step": int(payload["step"]),
            "diagnostic_only": bool(payload["diagnostic_only"]),
        },
        "pretrained_weight": {
            "path": str(weight_path),
            "sha256": file_sha256(weight_path),
            "source_url": weight_provenance["source_url"],
        },
        "validation": {
            "split": "val",
            "split_file": str(split_path),
            "split_sha256": file_sha256(split_path),
            **metrics,
        },
        "baseline": {
            "path": str(baseline_path),
            "sha256": file_sha256(baseline_path),
            "miou": baseline_miou,
        },
        "selection": {
            "metric": "miou",
            "tta_miou": float(metrics["miou"]),
            "baseline_miou": baseline_miou,
            "delta": delta,
            "accepted": accepted,
            "rule": "accept only when full-official-val TTA mIoU is strictly greater",
        },
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": int(peak_memory),
    }
    write_json(result_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
