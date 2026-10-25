from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
TEST_TMP_ROOT = APP_ROOT / ".test_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)

from engine import file_sha256  # noqa: E402
from tools import train_segformer  # noqa: E402


class _TwoPartModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.decode_head = nn.Linear(4, 2)


class SegFormerTrainerTest(unittest.TestCase):
    def test_adamw_groups_are_disjoint_exhaustive_and_named(self) -> None:
        model = _TwoPartModel()
        optimizer = train_segformer.build_optimizer(
            model,
            {"encoder_lr": 6e-5, "decoder_lr": 6e-4, "weight_decay": 0.01},
        )

        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(
            [(group["name"], group["lr"]) for group in optimizer.param_groups],
            [("encoder", 6e-5), ("decoder", 6e-4)],
        )
        grouped = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(grouped), len({id(parameter) for parameter in grouped}))
        self.assertEqual(
            {id(parameter) for parameter in grouped},
            {id(parameter) for parameter in model.parameters()},
        )

    def test_weight_verification_requires_converted_checkpoint_hash(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as temporary:
            root = Path(temporary)
            weight_path = root / "mit_b1.pth"
            weight_path.write_bytes(b"native-mit-checkpoint")
            digest = file_sha256(weight_path)
            provenance_path = root / "provenance.json"
            provenance_path.write_text(
                json.dumps({"ok": True, "converted_sha256": digest}),
                encoding="utf-8",
            )
            config = {"model": {"encoder_weights_path": str(weight_path)}}

            with patch.object(train_segformer, "EXPECTED_WEIGHT_SHA256", digest):
                actual_path, provenance = train_segformer.verify_pretrained_weight(
                    config
                )
            self.assertEqual(actual_path, weight_path.resolve())
            self.assertEqual(provenance["converted_sha256"], digest)

            provenance_path.write_text(
                json.dumps({"ok": True, "sha256": digest}), encoding="utf-8"
            )
            with patch.object(train_segformer, "EXPECTED_WEIGHT_SHA256", digest):
                with self.assertRaises(RuntimeError):
                    train_segformer.verify_pretrained_weight(config)


if __name__ == "__main__":
    unittest.main()
