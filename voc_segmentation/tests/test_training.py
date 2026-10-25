from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from engine import (
    is_recoverable_amp_overflow,
    poly_lr_factor,
    reconcile_history_for_resume,
    set_learning_rate,
)


class TrainingHelpersTest(unittest.TestCase):
    def test_warmup_and_poly_endpoints(self) -> None:
        self.assertAlmostEqual(poly_lr_factor(1, 1), 1.0)
        self.assertAlmostEqual(poly_lr_factor(1, 10, warmup_steps=2), 0.5)
        self.assertAlmostEqual(poly_lr_factor(2, 10, warmup_steps=2), 1.0)
        self.assertAlmostEqual(poly_lr_factor(10, 10, warmup_steps=2), 0.0)

    def test_learning_rate_preserves_group_base(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.2)
        self.assertEqual(set_learning_rate(optimizer, 0.5), [0.1])
        self.assertEqual(set_learning_rate(optimizer, 0.25), [0.05])

    def test_rejects_invalid_schedule(self) -> None:
        with self.assertRaises(ValueError):
            poly_lr_factor(0, 10)
        with self.assertRaises(ValueError):
            poly_lr_factor(1, 10, warmup_steps=10)

    def test_amp_overflow_requires_verified_scale_backoff(self) -> None:
        self.assertFalse(
            is_recoverable_amp_overflow(
                2.0, amp_enabled=True, scale_before=1024.0, scale_after=2048.0
            )
        )
        self.assertTrue(
            is_recoverable_amp_overflow(
                float("inf"),
                amp_enabled=True,
                scale_before=16384.0,
                scale_after=8192.0,
            )
        )
        with self.assertRaises(RuntimeError):
            is_recoverable_amp_overflow(
                float("nan"),
                amp_enabled=False,
                scale_before=1.0,
                scale_after=1.0,
            )
        with self.assertRaises(RuntimeError):
            is_recoverable_amp_overflow(
                float("inf"),
                amp_enabled=True,
                scale_before=16384.0,
                scale_after=16384.0,
            )

    def test_resume_history_is_archived_and_trimmed_to_checkpoint(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = root / "history.jsonl"
            archive = root / "archive" / "before_resume.jsonl"
            records = [
                {"type": "train", "step": 1},
                {"type": "validation", "step": 1},
                {"type": "train", "step": 2},
                {"type": "amp_overflow", "step": 3},
            ]
            original = "".join(json.dumps(item) + "\n" for item in records)
            history.write_text(original, encoding="utf-8")

            result = reconcile_history_for_resume(
                history, start_step=1, archive_path=archive
            )

            self.assertEqual(result["original_records"], 4)
            self.assertEqual(result["kept_records"], 2)
            self.assertEqual(result["dropped_records"], 2)
            self.assertEqual(archive.read_text(encoding="utf-8"), original)
            kept = [
                json.loads(line) for line in history.read_text().splitlines()
            ]
            self.assertEqual([item["type"] for item in kept], ["train", "validation"])

    def test_resume_history_rejects_duplicate_completed_steps(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            history = root / "history.jsonl"
            history.write_text(
                '{"type":"train","step":1}\n'
                '{"type":"train","step":1}\n',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                reconcile_history_for_resume(
                    history,
                    start_step=1,
                    archive_path=root / "archive.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
