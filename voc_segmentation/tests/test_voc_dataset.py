from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
TEST_TMP_ROOT = APP_ROOT / ".test_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)

from datasets.voc import SegmentationTrainTransform, VOCSegmentationDataset  # noqa: E402


class VocDatasetTest(unittest.TestCase):
    def test_transform_preserves_labels_and_uses_ignore_padding(self) -> None:
        image = Image.fromarray(np.full((2, 3, 3), 127, dtype=np.uint8), mode="RGB")
        mask = Image.fromarray(
            np.array([[0, 1, 20], [255, 3, 4]], dtype=np.uint8), mode="L"
        )
        transform = SegmentationTrainTransform(
            crop_size=(4, 5),
            scale_range=(1.0, 1.0),
            horizontal_flip_probability=0.0,
            color_jitter=ColorJitter(),
        )

        image_tensor, mask_tensor = transform(image, mask)

        self.assertEqual(tuple(image_tensor.shape), (3, 4, 5))
        self.assertEqual(tuple(mask_tensor.shape), (4, 5))
        self.assertEqual(set(mask_tensor.unique().tolist()), {0, 1, 3, 4, 20, 255})
        self.assertEqual(mask_tensor.dtype.name if hasattr(mask_tensor.dtype, "name") else str(mask_tensor.dtype), "torch.int64")

    def test_reads_official_split_without_label_conversion(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as temp:
            root = Path(temp)
            (root / "ImageSets/Segmentation").mkdir(parents=True)
            (root / "JPEGImages").mkdir()
            (root / "SegmentationClass").mkdir()
            (root / "ImageSets/Segmentation/train.txt").write_text(
                "sample\n", encoding="utf-8"
            )
            Image.fromarray(np.zeros((8, 9, 3), dtype=np.uint8), mode="RGB").save(
                root / "JPEGImages/sample.jpg"
            )
            palette_mask = Image.new("P", (9, 8), 20)
            palette = [channel for value in range(256) for channel in (value,) * 3]
            palette_mask.putpalette(palette)
            palette_mask.save(root / "SegmentationClass/sample.png")
            transform = SegmentationTrainTransform(
                crop_size=(8, 9),
                scale_range=(1.0, 1.0),
                horizontal_flip_probability=0.0,
                color_jitter=ColorJitter(),
            )
            dataset = VOCSegmentationDataset(root, "train", transform)

            sample = dataset[0]

            self.assertEqual(len(dataset), 1)
            self.assertEqual(sample["id"], "sample")
            self.assertEqual(sample["source"], "voc")
            self.assertEqual(sample["original_size"], (8, 9))
            self.assertEqual(set(sample["mask"].unique().tolist()), {20})


if __name__ == "__main__":
    unittest.main()
