from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from tools import train_segformer  # noqa: E402


class _MappingCriterion(nn.Module):
    def forward(
        self, outputs: dict[str, torch.Tensor], masks: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        total = outputs["logits"].mean() + masks.to(torch.float32).mean()
        return {"total": total, "ce": total * 0.75, "edge": total * 0.25}


class GCBTrainerTest(unittest.TestCase):
    def test_mapping_loss_components_remain_available_for_history(self) -> None:
        outputs = {"logits": torch.ones(1, 2, 2, 2, requires_grad=True)}
        masks = torch.zeros(1, 2, 2, dtype=torch.long)
        losses = train_segformer.compute_training_losses(
            outputs, masks, _MappingCriterion()
        )

        self.assertEqual(set(losses), {"total", "ce", "edge"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["total"].backward()
        self.assertIsNotNone(outputs["logits"].grad)

    def test_nonfinite_mapping_loss_is_rejected(self) -> None:
        class _NonFinite(nn.Module):
            def forward(self, outputs, masks):  # type: ignore[no-untyped-def]
                return {"total": outputs["logits"].sum() * float("nan")}

        with self.assertRaises(RuntimeError):
            train_segformer.compute_training_losses(
                {"logits": torch.ones(1, 1, 1, 1)},
                torch.zeros(1, 1, 1, dtype=torch.long),
                _NonFinite(),
            )


if __name__ == "__main__":
    unittest.main()
