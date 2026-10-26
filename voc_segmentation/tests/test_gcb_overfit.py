from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools import overfit_gcb_seg  # noqa: E402


class _TwoPartModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.decoder = nn.Linear(4, 2)


class GCBOverfitToolTest(unittest.TestCase):
    def test_optimizer_groups_are_disjoint_exhaustive_and_named(self) -> None:
        model = _TwoPartModel()
        optimizer = overfit_gcb_seg.build_optimizer(
            model,
            encoder_learning_rate=1e-4,
            decoder_learning_rate=1e-3,
        )

        self.assertEqual(
            [(group["name"], group["lr"]) for group in optimizer.param_groups],
            [("encoder", 1e-4), ("decoder", 1e-3)],
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

    def test_detached_losses_requires_complete_finite_components(self) -> None:
        losses = {
            name: torch.tensor(float(index + 1))
            for index, name in enumerate(overfit_gcb_seg.LOSS_NAMES)
        }
        values = overfit_gcb_seg.detached_losses(losses)
        self.assertEqual(tuple(values), overfit_gcb_seg.LOSS_NAMES)
        self.assertTrue(all(math.isfinite(value) for value in values.values()))

        incomplete = dict(losses)
        incomplete.pop("edge_dice")
        with self.assertRaises(ValueError):
            overfit_gcb_seg.detached_losses(incomplete)

        nonfinite = dict(losses)
        nonfinite["total"] = torch.tensor(float("nan"))
        with self.assertRaises(RuntimeError):
            overfit_gcb_seg.detached_losses(nonfinite)


if __name__ == "__main__":
    unittest.main()
