#!/usr/bin/env python3
"""Audit Pascal VOC segmentation data before starting a training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


VALID_LABELS = frozenset(range(21)) | {255}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def _read_split(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def audit_voc(voc_root: Path, test_image_dir: Path | None = None) -> dict[str, Any]:
    """Return a machine-readable audit report for VOC and optional test data."""
    voc_root = Path(voc_root)
    split_root = voc_root / 'ImageSets' / 'Segmentation'
    splits = {
        name: _read_split(split_root / f'{name}.txt')
        for name in ('train', 'val', 'trainval')
    }
    errors: list[str] = []

    for name, entries in splits.items():
        if not entries:
            errors.append(f'missing or empty split: {name}.txt')
        if len(entries) != len(set(entries)):
            errors.append(f'duplicate identifiers in {name}.txt')

    overlap = sorted(set(splits['train']) & set(splits['val']))
    if overlap:
        errors.append(f'train/val overlap: {len(overlap)} samples')
    if set(splits['trainval']) != set(splits['train']) | set(splits['val']):
        errors.append('trainval.txt is not the union of train.txt and val.txt')

    missing: list[str] = []
    size_mismatches: list[str] = []
    unreadable: list[str] = []
    labels: set[int] = set()
    for split in ('train', 'val'):
        for sample_id in splits[split]:
            image_path = voc_root / 'JPEGImages' / f'{sample_id}.jpg'
            mask_path = voc_root / 'SegmentationClass' / f'{sample_id}.png'
            if not image_path.is_file() or not mask_path.is_file():
                missing.append(sample_id)
                continue
            try:
                with Image.open(image_path) as image, Image.open(mask_path) as mask:
                    if image.size != mask.size:
                        size_mismatches.append(sample_id)
                    labels.update(int(value) for value in np.unique(np.asarray(mask)))
            except (OSError, ValueError) as exc:
                unreadable.append(f'{sample_id}: {exc}')

    invalid_labels = sorted(labels - VALID_LABELS)
    if missing:
        errors.append(f'missing image or mask: {len(missing)} samples')
    if size_mismatches:
        errors.append(f'image/mask size mismatch: {len(size_mismatches)} samples')
    if unreadable:
        errors.append(f'unreadable image or mask: {len(unreadable)} samples')
    if invalid_labels:
        errors.append(f'invalid labels: {invalid_labels}')

    test_summary: dict[str, Any] | None = None
    if test_image_dir is not None:
        test_image_dir = Path(test_image_dir)
        files = sorted(
            path for path in test_image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if test_image_dir.is_dir() else []
        test_errors: list[str] = []
        stems = [path.stem for path in files]
        if len(stems) != len(set(stems)):
            test_errors.append('test image stems are not unique')
        for path in files:
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, ValueError) as exc:
                test_errors.append(f'{path.name}: {exc}')
        if len(files) != 400:
            test_errors.append(f'expected 400 test images, found {len(files)}')
        errors.extend(f'test set: {message}' for message in test_errors)
        test_summary = {
            'count': len(files),
            'unique_stems': len(set(stems)),
            'errors': test_errors,
        }

    return {
        'ok': not errors,
        'voc_root': str(voc_root),
        'split_counts': {name: len(entries) for name, entries in splits.items()},
        'train_val_overlap': len(overlap),
        'missing_count': len(missing),
        'size_mismatch_count': len(size_mismatches),
        'unreadable_count': len(unreadable),
        'labels': sorted(labels),
        'invalid_labels': invalid_labels,
        'test': test_summary,
        'errors': errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('voc_root', type=Path)
    parser.add_argument('--test-image-dir', type=Path)
    parser.add_argument('--output', type=Path, help='Optional JSON report path.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_voc(args.voc_root, args.test_image_dir)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
