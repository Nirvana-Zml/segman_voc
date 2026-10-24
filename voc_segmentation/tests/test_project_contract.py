from __future__ import annotations

import sys
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from project_contract import (  # noqa: E402
    CLASS_IDS,
    IGNORE_INDEX,
    NUM_CLASSES,
    PREDICTION_COUNT,
    PREDICTION_MODE,
    REDUCE_ZERO_LABEL,
)


class ProjectContractTest(unittest.TestCase):
    def test_voc_label_contract(self) -> None:
        self.assertEqual(NUM_CLASSES, 21)
        self.assertEqual(CLASS_IDS, tuple(range(21)))
        self.assertEqual(IGNORE_INDEX, 255)
        self.assertFalse(REDUCE_ZERO_LABEL)

    def test_prediction_contract(self) -> None:
        self.assertEqual(PREDICTION_COUNT, 400)
        self.assertEqual(PREDICTION_MODE, "L")


if __name__ == "__main__":
    unittest.main()
