#!/usr/bin/env python3
"""Run the shared formal trainer with the GCB A1 ablation defaults."""

from __future__ import annotations

import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools.train_segformer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        main(
            default_config=Path("configs/gcb_ablation_a1_gcm.yaml"),
            default_output_dir=Path("runs/gcb_ablation_a1_gcm_voc_train"),
        )
    )
