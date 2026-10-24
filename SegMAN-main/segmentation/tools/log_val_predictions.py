#!/usr/bin/env python3
"""Write fixed VOC validation predictions from checkpoints to TensorBoard."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from mmseg.apis import inference_segmentor, init_segmentor
from mmseg.datasets import PascalVOCDataset


def _checkpoint_step(path: Path, fallback: int) -> int:
    match = re.search(r'iter_(\d+)', path.stem)
    return int(match.group(1)) if match else fallback


def _colorize(mask: np.ndarray) -> np.ndarray:
    palette = np.asarray(PascalVOCDataset.PALETTE, dtype=np.uint8)
    rgb = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    valid = (mask >= 0) & (mask < len(palette))
    rgb[valid] = palette[mask[valid]]
    return rgb


def _expand_checkpoints(entries: list[Path]) -> list[Path]:
    checkpoints: list[Path] = []
    for entry in entries:
        if entry.is_dir():
            checkpoints.extend(entry.glob('*.pth'))
        elif entry.is_file():
            checkpoints.append(entry)
    return sorted(set(checkpoints), key=lambda path: (_checkpoint_step(path, 0), path.name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoints', type=Path, nargs='+')
    parser.add_argument('--voc-root', type=Path, required=True)
    parser.add_argument('--sample-ids', nargs='+', required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoints = _expand_checkpoints(args.checkpoints)
    if not checkpoints:
        raise SystemExit('No checkpoint files found.')

    samples = []
    for sample_id in args.sample_ids:
        image_path = args.voc_root / 'JPEGImages' / f'{sample_id}.jpg'
        mask_path = args.voc_root / 'SegmentationClass' / f'{sample_id}.png'
        if not image_path.is_file() or not mask_path.is_file():
            raise SystemExit(f'Missing validation sample: {sample_id}')
        samples.append((sample_id, image_path, mask_path))

    writer = SummaryWriter(str(args.log_dir))
    try:
        for checkpoint_index, checkpoint in enumerate(checkpoints, 1):
            step = _checkpoint_step(checkpoint, checkpoint_index)
            model = init_segmentor(args.config, str(checkpoint), device=args.device)
            for sample_id, image_path, mask_path in samples:
                with Image.open(image_path) as image:
                    image_rgb = np.asarray(image.convert('RGB'))
                with Image.open(mask_path) as mask_image:
                    ground_truth = np.asarray(mask_image)
                prediction = np.asarray(inference_segmentor(model, str(image_path))[0])
                writer.add_image(
                    f'val/{sample_id}/image', image_rgb, step, dataformats='HWC')
                writer.add_image(
                    f'val/{sample_id}/ground_truth', _colorize(ground_truth),
                    step, dataformats='HWC')
                writer.add_image(
                    f'val/{sample_id}/prediction', _colorize(prediction),
                    step, dataformats='HWC')
            writer.flush()
            print(f'logged {checkpoint.name} at step {step}')
    finally:
        writer.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
