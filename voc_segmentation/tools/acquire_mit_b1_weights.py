#!/usr/bin/env python3
"""Acquire, convert, and validate NVIDIA MiT-B1 ImageNet-1K weights."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from engine import file_sha256  # noqa: E402
from models.encoders import build_mit_b1  # noqa: E402


CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")
MODEL_ID = "nvidia/mit-b1"
REVISION = "13ddceec4e8bdf401e7cd7acf5aebc526222518c"
SOURCE_FILENAME = "pytorch_model.bin"
SOURCE_URL = (
    f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{SOURCE_FILENAME}"
)
EXPECTED_SOURCE_BYTES = 54_732_285
EXPECTED_SOURCE_SHA256 = (
    "980b86b60db37b1b1528086f6c53d253d879e9e09ebe07397b40fd275738a3bf"
)
MODEL_CARD_URL = "https://huggingface.co/nvidia/mit-b1"
OFFICIAL_REPOSITORY_URL = "https://github.com/NVlabs/SegFormer"
OFFICIAL_LICENSE_URL = "https://github.com/NVlabs/SegFormer/blob/master/LICENSE"
IMAGENET_TERMS_URL = "https://www.image-net.org/download-images.php"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/weights/mit_b1_imagenet1k"),
    )
    return parser.parse_args()


def _take(
    source: Mapping[str, Tensor],
    consumed: set[str],
    name: str,
) -> Tensor:
    value = source.get(name)
    if not isinstance(value, Tensor):
        raise KeyError(f"missing tensor in NVIDIA checkpoint: {name}")
    consumed.add(name)
    return value


def convert_huggingface_state_dict(source: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Convert the NVIDIA/Hugging Face MiT-B1 encoder to project naming."""

    converted: dict[str, Tensor] = {}
    consumed: set[str] = set()
    for stage_index in range(4):
        source_prefix = f"segformer.encoder.patch_embeddings.{stage_index}"
        target_prefix = f"patch_embed{stage_index + 1}"
        for suffix in ("weight", "bias"):
            converted[f"{target_prefix}.proj.{suffix}"] = _take(
                source, consumed, f"{source_prefix}.proj.{suffix}"
            )
            converted[f"{target_prefix}.norm.{suffix}"] = _take(
                source, consumed, f"{source_prefix}.layer_norm.{suffix}"
            )

    for stage_index in range(4):
        for block_index in range(2):
            source_prefix = f"segformer.encoder.block.{stage_index}.{block_index}"
            target_prefix = f"block{stage_index + 1}.{block_index}"
            direct_mappings = {
                "layer_norm_1": "norm1",
                "attention.self.query": "attn.q",
                "attention.output.dense": "attn.proj",
                "layer_norm_2": "norm2",
                "mlp.dense1": "mlp.fc1",
                "mlp.dwconv.dwconv": "mlp.dwconv.dwconv",
                "mlp.dense2": "mlp.fc2",
            }
            if stage_index < 3:
                direct_mappings.update(
                    {
                        "attention.self.sr": "attn.sr",
                        "attention.self.layer_norm": "attn.norm",
                    }
                )
            for source_name, target_name in direct_mappings.items():
                for suffix in ("weight", "bias"):
                    converted[f"{target_prefix}.{target_name}.{suffix}"] = _take(
                        source,
                        consumed,
                        f"{source_prefix}.{source_name}.{suffix}",
                    )

            for suffix in ("weight", "bias"):
                key = _take(
                    source,
                    consumed,
                    f"{source_prefix}.attention.self.key.{suffix}",
                )
                value = _take(
                    source,
                    consumed,
                    f"{source_prefix}.attention.self.value.{suffix}",
                )
                converted[f"{target_prefix}.attn.kv.{suffix}"] = torch.cat(
                    [key, value], dim=0
                )

    for stage_index in range(4):
        for suffix in ("weight", "bias"):
            converted[f"norm{stage_index + 1}.{suffix}"] = _take(
                source,
                consumed,
                f"segformer.encoder.layer_norm.{stage_index}.{suffix}",
            )

    ignored = {"classifier.weight", "classifier.bias"}
    remaining = set(source) - consumed
    if remaining != ignored:
        raise RuntimeError(
            "unexpected NVIDIA checkpoint keys after conversion: "
            f"{sorted(remaining)}"
        )
    return converted


def _validate_source(path: Path) -> tuple[str, Mapping[str, Tensor]]:
    if path.stat().st_size != EXPECTED_SOURCE_BYTES:
        raise RuntimeError(f"unexpected source byte count: {path.stat().st_size}")
    sha256 = file_sha256(path)
    if sha256 != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source SHA-256 mismatch: {sha256}")
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("NVIDIA checkpoint is not a state dictionary")
    return sha256, payload


def _validate_converted(state_dict: Mapping[str, Tensor]) -> dict[str, Any]:
    encoder = build_mit_b1()
    expected = encoder.state_dict()
    if set(state_dict) != set(expected):
        raise RuntimeError("converted state dictionary does not exactly cover MiT-B1")
    shape_mismatches = {
        name: [list(state_dict[name].shape), list(expected[name].shape)]
        for name in expected
        if state_dict[name].shape != expected[name].shape
    }
    if shape_mismatches:
        raise RuntimeError(f"converted tensor shape mismatches: {shape_mismatches}")
    finite = all(
        bool(torch.isfinite(value).all())
        for value in state_dict.values()
        if value.is_floating_point()
    )
    if not finite:
        raise RuntimeError("converted MiT-B1 state dictionary contains non-finite values")
    encoder.load_state_dict(dict(state_dict), strict=True)
    encoder.eval()
    with torch.no_grad():
        features = encoder(torch.zeros(1, 3, 64, 96))
    forward_finite = all(bool(torch.isfinite(feature).all()) for feature in features)
    if not forward_finite:
        raise RuntimeError("strictly loaded MiT-B1 produced non-finite features")
    return {
        "strict_native_load_ok": True,
        "finite_floating_tensors": finite,
        "finite_forward_ok": forward_finite,
        "feature_shapes": [list(feature.shape) for feature in features],
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "hf_pytorch_model.bin"
    downloaded = not source_path.is_file()
    if downloaded:
        temporary = source_path.with_suffix(source_path.suffix + ".part")
        if temporary.exists():
            raise FileExistsError(f"refusing to overwrite partial file: {temporary}")
        torch.hub.download_url_to_file(SOURCE_URL, temporary, progress=True)
        source_sha256 = file_sha256(temporary)
        if temporary.stat().st_size != EXPECTED_SOURCE_BYTES:
            raise RuntimeError(f"unexpected source byte count: {temporary.stat().st_size}")
        if source_sha256 != EXPECTED_SOURCE_SHA256:
            raise RuntimeError(f"source SHA-256 mismatch: {source_sha256}")
        temporary.replace(source_path)

    source_sha256, source_state = _validate_source(source_path)
    converted_state = convert_huggingface_state_dict(source_state)
    validation = _validate_converted(converted_state)
    converted_path = output_dir / "mit_b1.pth"
    converted_this_run = not converted_path.is_file()
    if converted_this_run:
        temporary = converted_path.with_suffix(converted_path.suffix + ".part")
        if temporary.exists():
            raise FileExistsError(f"refusing to overwrite partial file: {temporary}")
        torch.save(converted_state, temporary)
        temporary.replace(converted_path)
    saved_state = torch.load(converted_path, map_location="cpu")
    if not isinstance(saved_state, Mapping):
        raise TypeError("converted checkpoint is not a state dictionary")
    _validate_converted(saved_state)
    converted_sha256 = file_sha256(converted_path)

    provenance: dict[str, Any] = {
        "ok": True,
        "timestamp": datetime.now(CHINA_TIME).isoformat(timespec="seconds"),
        "weight_id": MODEL_ID,
        "revision": REVISION,
        "scope": (
            "ImageNet-1K classification initialization for the MiT-B1 encoder "
            "used by the SegFormer baseline and later GCB-Seg experiments"
        ),
        "source_url": SOURCE_URL,
        "model_card_url": MODEL_CARD_URL,
        "official_repository_url": OFFICIAL_REPOSITORY_URL,
        "source_archive": str(source_path),
        "source_archive_bytes": source_path.stat().st_size,
        "source_sha256": source_sha256,
        "expected_source_bytes": EXPECTED_SOURCE_BYTES,
        "expected_source_sha256": EXPECTED_SOURCE_SHA256,
        "downloaded_this_run": downloaded,
        "source_tensor_count": sum(isinstance(value, Tensor) for value in source_state.values()),
        "ignored_source_keys": ["classifier.weight", "classifier.bias"],
        "conversion": (
            "Hugging Face SegFormer encoder names mapped to official-style MiT "
            "stage names; per-block key and value projections concatenated into kv"
        ),
        "converted_checkpoint": str(converted_path),
        "converted_checkpoint_bytes": converted_path.stat().st_size,
        "converted_sha256": converted_sha256,
        "converted_this_run": converted_this_run,
        "converted_tensor_count": len(saved_state),
        **validation,
        "torch_version": torch.__version__,
        "license": {
            "name": "NVIDIA Source Code License for SegFormer",
            "url": OFFICIAL_LICENSE_URL,
            "use_limitation": "non-commercial research or evaluation only",
            "imagenet_terms_url": IMAGENET_TERMS_URL,
            "note": (
                "The official SegFormer repository identifies this license and "
                "limits use to non-commercial research or evaluation. This course "
                "project uses the weights only for educational evaluation and "
                "asserts no broader rights. ImageNet terms remain applicable."
            ),
        },
    }
    provenance_path = output_dir / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
