from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


TOOLS_DIR = Path(__file__).resolve().parents[1] / 'segmentation' / 'tools'
sys.path.insert(0, str(TOOLS_DIR))

from verify_predictions import verify_predictions  # noqa: E402


class PredictionFormatTest(unittest.TestCase):
    def test_valid_grayscale_prediction(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            image_dir = Path(tmp) / 'images'
            prediction_dir = Path(tmp) / 'predictions'
            image_dir.mkdir()
            prediction_dir.mkdir()
            Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(
                image_dir / 'sample.jpg')
            Image.fromarray(np.full((3, 4), 20, dtype=np.uint8)).save(
                prediction_dir / 'sample.png')

            report = verify_predictions(image_dir, prediction_dir, expected_count=1)
            self.assertTrue(report['ok'], report['errors'])

    def test_rejects_rgb_and_invalid_class(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            image_dir = Path(tmp) / 'images'
            prediction_dir = Path(tmp) / 'predictions'
            image_dir.mkdir()
            prediction_dir.mkdir()
            Image.fromarray(np.zeros((3, 4, 3), dtype=np.uint8)).save(
                image_dir / 'sample.jpg')
            Image.fromarray(np.full((3, 4, 3), 30, dtype=np.uint8)).save(
                prediction_dir / 'sample.png')

            report = verify_predictions(image_dir, prediction_dir, expected_count=1)
            self.assertFalse(report['ok'])
            self.assertEqual(len(report['invalid_files']), 1)
            self.assertIn('mode=RGB', report['invalid_files'][0])
            self.assertIn('out-of-range', report['invalid_files'][0])


if __name__ == '__main__':
    unittest.main()
