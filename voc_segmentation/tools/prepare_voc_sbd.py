#!/usr/bin/env python3
"""Convert official SBD category masks and build leak-free VOC trainaug."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from datasets.sbd import prepare_trainaug  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path("/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"),
    )
    parser.add_argument(
        "--sbd-root",
        type=Path,
        default=Path("/data/sbd/benchmark_RELEASE/dataset"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("/data/sbd/benchmark.tgz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/sbd"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_trainaug(
        args.voc_root, args.sbd_root, args.archive, args.output_dir
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
