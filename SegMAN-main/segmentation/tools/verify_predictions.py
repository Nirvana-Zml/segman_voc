#!/usr/bin/env python3
"""Verify that a prediction directory matches the required submission format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def verify_predictions(
    image_dir: Path,
    prediction_dir: Path,
    expected_count: int = 400,
    class_min: int = 0,
    class_max: int = 20,
) -> dict[str, Any]:
    """Return validation details without modifying prediction files."""
    image_dir = Path(image_dir)
    prediction_dir = Path(prediction_dir)
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if image_dir.is_dir() else []
    predictions = sorted(prediction_dir.glob('*.png')) if prediction_dir.is_dir() else []
    image_by_stem = {path.stem: path for path in images}
    prediction_by_stem = {path.stem: path for path in predictions}
    errors: list[str] = []

    if len(images) != expected_count:
        errors.append(f'expected {expected_count} input images, found {len(images)}')
    if len(predictions) != expected_count:
        errors.append(f'expected {expected_count} PNG predictions, found {len(predictions)}')

    missing = sorted(set(image_by_stem) - set(prediction_by_stem))
    extra = sorted(set(prediction_by_stem) - set(image_by_stem))
    if missing:
        errors.append(f'missing predictions: {len(missing)}')
    if extra:
        errors.append(f'extra predictions: {len(extra)}')

    invalid_files: list[str] = []
    observed_values: set[int] = set()
    for stem in sorted(set(image_by_stem) & set(prediction_by_stem)):
        image_path = image_by_stem[stem]
        prediction_path = prediction_by_stem[stem]
        try:
            with Image.open(image_path) as image, Image.open(prediction_path) as prediction:
                prediction_array = np.asarray(prediction)
                reasons: list[str] = []
                if prediction.mode != 'L':
                    reasons.append(f'mode={prediction.mode}, expected L')
                if prediction_array.dtype != np.uint8:
                    reasons.append(f'dtype={prediction_array.dtype}, expected uint8')
                if image.size != prediction.size:
                    reasons.append(f'size={prediction.size}, expected {image.size}')
                values = {int(value) for value in np.unique(prediction_array)}
                observed_values.update(values)
                outside = sorted(value for value in values
                                 if value < class_min or value > class_max)
                if outside:
                    reasons.append(f'out-of-range values={outside}')
                if reasons:
                    invalid_files.append(f'{prediction_path.name}: {"; ".join(reasons)}')
        except (OSError, ValueError) as exc:
            invalid_files.append(f'{prediction_path.name}: unreadable: {exc}')

    if invalid_files:
        errors.append(f'invalid prediction files: {len(invalid_files)}')

    return {
        'ok': not errors,
        'input_count': len(images),
        'prediction_count': len(predictions),
        'missing': missing,
        'extra': extra,
        'observed_values': sorted(observed_values),
        'invalid_files': invalid_files,
        'errors': errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_dir', type=Path)
    parser.add_argument('prediction_dir', type=Path)
    parser.add_argument('--expected-count', type=int, default=400)
    parser.add_argument('--class-min', type=int, default=0)
    parser.add_argument('--class-max', type=int, default=20)
    parser.add_argument('--output', type=Path, help='Optional JSON report path.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = verify_predictions(
        args.image_dir,
        args.prediction_dir,
        expected_count=args.expected_count,
        class_min=args.class_min,
        class_max=args.class_max,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + '\n', encoding='utf-8')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
