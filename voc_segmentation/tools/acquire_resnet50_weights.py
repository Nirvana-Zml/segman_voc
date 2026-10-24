#!/usr/bin/env python3
"""Acquire and validate torchvision ResNet-50 ImageNet-1K V2 weights."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import torchvision
from torchvision.models import ResNet50_Weights, resnet50


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from engine import file_sha256  # noqa: E402


CHINA_TIME = timezone(timedelta(hours=8), name="Asia/Shanghai")
WEIGHT_ID = "ResNet50_Weights.IMAGENET1K_V2"
EXPECTED_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
EXPECTED_SHA256_PREFIX = "11ad3fa6"
DOCS_URL = "https://docs.pytorch.org/vision/0.16/models/generated/torchvision.models.resnet50.html"
RECIPE_URL = "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621"
TORCHVISION_LICENSE_URL = "https://github.com/pytorch/vision/blob/v0.16.2/LICENSE"
TORCHVISION_MODEL_LICENSE_NOTE_URL = "https://github.com/pytorch/vision#pre-trained-model-license"
IMAGENET_TERMS_URL = "https://www.image-net.org/download-images.php"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/weights/resnet50_imagenet1k_v2"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = ResNet50_Weights.IMAGENET1K_V2
    if weights.url != EXPECTED_URL:
        raise RuntimeError(f"unexpected torchvision weight URL: {weights.url}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = output_dir / Path(weights.url).name
    downloaded = not weight_path.is_file()
    if downloaded:
        temporary = weight_path.with_suffix(weight_path.suffix + ".part")
        torch.hub.download_url_to_file(
            weights.url,
            temporary,
            hash_prefix=EXPECTED_SHA256_PREFIX,
            progress=True,
        )
        temporary.replace(weight_path)

    sha256 = file_sha256(weight_path)
    if not sha256.startswith(EXPECTED_SHA256_PREFIX):
        raise RuntimeError(
            f"weight SHA-256 prefix mismatch: {sha256} does not start with "
            f"{EXPECTED_SHA256_PREFIX}"
        )
    state_dict = torch.load(weight_path, map_location="cpu")
    validation_model = resnet50(weights=None)
    validation_model.load_state_dict(state_dict, strict=True)
    tensor_count = sum(isinstance(value, torch.Tensor) for value in state_dict.values())
    finite = all(
        bool(torch.isfinite(value).all())
        for value in state_dict.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    )
    if not finite:
        raise RuntimeError("pretrained state dict contains non-finite tensors")

    provenance: dict[str, Any] = {
        "ok": True,
        "timestamp": datetime.now(CHINA_TIME).isoformat(timespec="seconds"),
        "weight_id": WEIGHT_ID,
        "scope": (
            "ImageNet-1K classification initialization for the ResNet-50 encoder "
            "of the DeepLabV3+ CNN baseline"
        ),
        "source_url": weights.url,
        "documentation_url": DOCS_URL,
        "recipe_url": weights.meta["recipe"],
        "torchvision_version": torchvision.__version__,
        "torch_version": torch.__version__,
        "archive": str(weight_path),
        "archive_bytes": weight_path.stat().st_size,
        "sha256": sha256,
        "expected_sha256_prefix": EXPECTED_SHA256_PREFIX,
        "downloaded_this_run": downloaded,
        "state_dict_tensor_count": tensor_count,
        "strict_resnet50_load_ok": True,
        "finite_floating_tensors": finite,
        "num_params": weights.meta["num_params"],
        "imagenet1k_acc1": weights.meta["_metrics"]["ImageNet-1K"]["acc@1"],
        "imagenet1k_acc5": weights.meta["_metrics"]["ImageNet-1K"]["acc@5"],
        "license": {
            "torchvision_code": "BSD-3-Clause",
            "torchvision_license_url": TORCHVISION_LICENSE_URL,
            "model_license_note_url": TORCHVISION_MODEL_LICENSE_NOTE_URL,
            "imagenet_terms_url": IMAGENET_TERMS_URL,
            "note": (
                "Torchvision code is BSD-3-Clause. Torchvision states that "
                "pretrained models may carry licenses or terms derived from their "
                "training data and places permission responsibility on the user. "
                "ImageNet describes access for non-commercial research or education "
                "under its conditions. This course project uses the weights only for "
                "educational model initialization; no broader rights are asserted."
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
