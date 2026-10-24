#!/usr/bin/env python3
"""Run SegMAN inference on an image directory and save class-index masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from mmseg.apis import inference_segmentor, init_segmentor


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('image_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_paths = sorted(
        path for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(f'No supported images found in {args.image_dir}')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.output_dir / f'{path.stem}.png' for path in image_paths]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            f'{len(existing)} prediction files already exist; use --overwrite '
            'to replace matching files.')

    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    for index, (image_path, target_path) in enumerate(zip(image_paths, targets), 1):
        result = inference_segmentor(model, str(image_path))[0]
        mask = np.asarray(result)
        if mask.ndim != 2:
            raise RuntimeError(f'{image_path.name}: expected 2-D mask, got {mask.shape}')
        if mask.size and (int(mask.min()) < 0 or int(mask.max()) > 20):
            raise RuntimeError(
                f'{image_path.name}: predicted label range '
                f'{int(mask.min())}..{int(mask.max())}, expected 0..20')
        mask = mask.astype(np.uint8, copy=False)
        with Image.open(image_path) as image:
            if mask.shape != (image.height, image.width):
                raise RuntimeError(
                    f'{image_path.name}: mask shape {mask.shape} does not match '
                    f'image {(image.height, image.width)}')
        Image.fromarray(mask).save(target_path)
        if index % 25 == 0 or index == len(image_paths):
            print(f'predicted {index}/{len(image_paths)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
