from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
TEST_TMP_ROOT = APP_ROOT / ".test_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)

from datasets.sbd import load_sbd_segmentation  # noqa: E402


class SbdConversionTest(unittest.TestCase):
    def test_loads_gtcls_segmentation_without_label_shift(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as temp:
            path = Path(temp) / "sample.mat"
            expected = np.array([[0, 1, 20], [255, 3, 4]], dtype=np.uint8)
            gtcls = np.empty(
                (1, 1),
                dtype=[("Boundaries", "O"), ("Segmentation", "O"), ("CategoriesPresent", "O")],
            )
            boundaries = np.empty((20, 1), dtype=object)
            for index in range(20):
                boundaries[index, 0] = np.zeros(expected.shape, dtype=np.uint8)
            gtcls[0, 0]["Boundaries"] = boundaries
            gtcls[0, 0]["Segmentation"] = expected
            gtcls[0, 0]["CategoriesPresent"] = np.array([[1, 3, 4, 20]], dtype=np.uint8)
            savemat(path, {"GTcls": gtcls})

            actual = load_sbd_segmentation(path)

            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
