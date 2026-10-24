"""Stable dataset and output contracts shared by the standalone project."""

from pathlib import Path


NUM_CLASSES = 21
IGNORE_INDEX = 255
REDUCE_ZERO_LABEL = False
CLASS_IDS = tuple(range(NUM_CLASSES))

CONTAINER_DATA_ROOT = Path("/data")
VOC2012_RELATIVE_ROOT = Path("VOCtrainval_11-May-2012/VOCdevkit/VOC2012")
TEST400_RELATIVE_ROOT = Path("test400/test400/images")

DEFAULT_SEED = 42
PREDICTION_COUNT = 400
PREDICTION_MODE = "L"
