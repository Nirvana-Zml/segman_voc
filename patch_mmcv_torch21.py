"""Patch mmcv parallel/_functions.py for torch>=2.1.0 stream API."""
from __future__ import annotations

import ast
import os
import re
import sys

import mmcv


def main() -> int:
    path = os.path.join(os.path.dirname(mmcv.__file__), "parallel", "_functions.py")
    src = open(path, encoding="utf-8").read()

    already = (
        'torch.device("cuda", device)' in src
        or "torch.device('cuda', device)" in src
    )
    if already:
        ast.parse(src)
        print(f"mmcv already patched: {path}")
        return 0

    pattern = re.compile(
        r"(?P<indent>[ \t]*)# Perform CPU to GPU copies in a background stream\n"
        r"(?P=indent)streams = \[_get_stream\(device\) for device in target_gpus\]"
    )
    match = pattern.search(src)
    if not match:
        print(f"ERROR: patch target not found in {path}", file=sys.stderr)
        return 1

    indent = match.group("indent")
    replacement = (
        f"{indent}# Perform CPU to GPU copies in a background stream\n"
        f"{indent}from packaging import version\n"
        f"{indent}if version.parse(torch.__version__) >= version.parse('2.1.0'):\n"
        f"{indent}    streams = [\n"
        f"{indent}        _get_stream(torch.device('cuda', device))\n"
        f"{indent}        for device in target_gpus\n"
        f"{indent}    ]\n"
        f"{indent}else:\n"
        f"{indent}    streams = [_get_stream(device) for device in target_gpus]"
    )
    new_src = pattern.sub(replacement, src, count=1)
    ast.parse(new_src)
    open(path, "w", encoding="utf-8").write(new_src)
    print(f"Patched: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
