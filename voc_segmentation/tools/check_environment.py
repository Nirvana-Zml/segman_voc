#!/usr/bin/env python3
"""Report the pinned runtime and optionally require a usable CUDA device."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Exit nonzero unless PyTorch can allocate a CUDA tensor.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON evidence path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    import PIL
    import numpy
    import tensorboard
    import timm
    import torch
    import torchvision
    import yaml

    cuda_available = torch.cuda.is_available()
    cuda_allocation_ok = False
    cuda_error = None
    if cuda_available:
        try:
            probe = torch.ones(1, device="cuda")
            cuda_allocation_ok = bool(probe.item() == 1)
        except RuntimeError as exc:
            cuda_error = str(exc)

    ok = (not args.require_cuda) or cuda_allocation_ok
    result = {
        "ok": ok,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_allocation_ok": cuda_allocation_ok,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_error": cuda_error,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "pyyaml": yaml.__version__,
        "tensorboard": tensorboard.__version__,
        "timm": timm.__version__,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
