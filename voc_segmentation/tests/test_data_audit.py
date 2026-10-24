from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
TEST_TMP_ROOT = APP_ROOT / ".test_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)

from datasets.audit import AuditExpectations, audit_dataset  # noqa: E402


def write_sample(voc_root: Path, sample_id: str, label: int) -> None:
    image = np.full((4, 5, 3), 32 + label, dtype=np.uint8)
    mask = np.full((4, 5), label, dtype=np.uint8)
    mask[0, 0] = 255
    Image.fromarray(image, mode="RGB").save(voc_root / "JPEGImages" / f"{sample_id}.jpg")
    Image.fromarray(mask, mode="L").save(voc_root / "SegmentationClass" / f"{sample_id}.png")


class DataAuditTest(unittest.TestCase):
    def make_tree(self, root: Path) -> tuple[Path, Path, Path]:
        voc_root = root / "VOC2012"
        test_root = root / "test"
        output = root / "artifacts"
        (voc_root / "JPEGImages").mkdir(parents=True)
        (voc_root / "SegmentationClass").mkdir()
        (voc_root / "ImageSets/Segmentation").mkdir(parents=True)
        test_root.mkdir()
        return voc_root, test_root, output

    def test_valid_small_dataset_passes_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as temp:
            voc_root, test_root, output = self.make_tree(Path(temp))
            write_sample(voc_root, "train_a", 1)
            write_sample(voc_root, "train_b", 2)
            write_sample(voc_root, "val_a", 3)
            (voc_root / "ImageSets/Segmentation/train.txt").write_text(
                "train_a\ntrain_b\n", encoding="utf-8"
            )
            (voc_root / "ImageSets/Segmentation/val.txt").write_text(
                "val_a\n", encoding="utf-8"
            )
            for index in range(2):
                Image.new("RGB", (5, 4)).save(test_root / f"test_{index}.jpg")

            summary = audit_dataset(
                voc_root,
                test_root,
                output,
                AuditExpectations(train_count=2, val_count=1, test_count=2, overlay_count=3),
            )

            self.assertTrue(summary["ok"])
            self.assertEqual(summary["train_val_overlap"], 0)
            self.assertEqual(summary["overlay_count"], 3)
            self.assertTrue((output / "split_summary.json").is_file())
            self.assertTrue((output / "class_pixel_statistics.csv").is_file())
            self.assertEqual(len(list((output / "overlays").glob("*.jpg"))), 3)

    def test_overlap_and_invalid_label_fail(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as temp:
            voc_root, test_root, output = self.make_tree(Path(temp))
            write_sample(voc_root, "shared", 1)
            mask_path = voc_root / "SegmentationClass/shared.png"
            invalid = np.asarray(Image.open(mask_path)).copy()
            invalid[1, 1] = 42
            Image.fromarray(invalid, mode="L").save(mask_path)
            (voc_root / "ImageSets/Segmentation/train.txt").write_text(
                "shared\n", encoding="utf-8"
            )
            (voc_root / "ImageSets/Segmentation/val.txt").write_text(
                "shared\n", encoding="utf-8"
            )
            Image.new("RGB", (5, 4)).save(test_root / "test.jpg")

            summary = audit_dataset(
                voc_root,
                test_root,
                output,
                AuditExpectations(train_count=1, val_count=1, test_count=1, overlay_count=1),
            )

            self.assertFalse(summary["ok"])
            self.assertEqual(summary["train_val_overlap"], 1)
            self.assertEqual(summary["details"]["invalid_label_samples"], 2)


if __name__ == "__main__":
    unittest.main()
