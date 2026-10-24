from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


TOOLS_DIR = Path(__file__).resolve().parents[1] / 'segmentation' / 'tools'
sys.path.insert(0, str(TOOLS_DIR))

from audit_voc import audit_voc  # noqa: E402


class VocAuditTest(unittest.TestCase):
    def test_valid_minimal_dataset(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp) / 'VOC2012'
            (root / 'JPEGImages').mkdir(parents=True)
            (root / 'SegmentationClass').mkdir()
            split_dir = root / 'ImageSets' / 'Segmentation'
            split_dir.mkdir(parents=True)
            split_dir.joinpath('train.txt').write_text('train_sample\n', encoding='utf-8')
            split_dir.joinpath('val.txt').write_text('val_sample\n', encoding='utf-8')
            split_dir.joinpath('trainval.txt').write_text(
                'train_sample\nval_sample\n', encoding='utf-8')

            image = np.zeros((3, 4, 3), dtype=np.uint8)
            mask = np.array(
                [[0, 1, 20, 255], [0, 1, 20, 255], [0, 1, 20, 255]],
                dtype=np.uint8)
            for sample_id in ('train_sample', 'val_sample'):
                Image.fromarray(image).save(root / 'JPEGImages' / f'{sample_id}.jpg')
                Image.fromarray(mask).save(
                    root / 'SegmentationClass' / f'{sample_id}.png')

            report = audit_voc(root)
            self.assertTrue(report['ok'], report['errors'])
            self.assertEqual(report['labels'], [0, 1, 20, 255])

    def test_detects_overlap_and_invalid_label(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp) / 'VOC2012'
            (root / 'JPEGImages').mkdir(parents=True)
            (root / 'SegmentationClass').mkdir()
            split_dir = root / 'ImageSets' / 'Segmentation'
            split_dir.mkdir(parents=True)
            for name in ('train', 'val', 'trainval'):
                split_dir.joinpath(f'{name}.txt').write_text('same\n', encoding='utf-8')
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(
                root / 'JPEGImages' / 'same.jpg')
            Image.fromarray(np.full((2, 2), 42, dtype=np.uint8)).save(
                root / 'SegmentationClass' / 'same.png')

            report = audit_voc(root)
            self.assertFalse(report['ok'])
            self.assertEqual(report['train_val_overlap'], 1)
            self.assertEqual(report['invalid_labels'], [42])


if __name__ == '__main__':
    unittest.main()
