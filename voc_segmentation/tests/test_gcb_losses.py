from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from losses import GCBSegLoss, lovasz_softmax_loss, semantic_boundary_targets  # noqa: E402


class GCBSegLossTest(unittest.TestCase):
    def test_boundary_targets_ignore_255_and_do_not_create_outer_edges(self) -> None:
        constant = torch.zeros(1, 5, 7, dtype=torch.long)
        target, valid = semantic_boundary_targets(constant)
        self.assertEqual(int(target.sum()), 0)
        self.assertTrue(valid.all())

        labels = constant.clone()
        labels[:, :, 4:] = 1
        labels[:, 2, 3] = 255
        target, valid = semantic_boundary_targets(labels)
        self.assertGreater(int(target.sum()), 0)
        self.assertFalse(bool(valid[0, 0, 2, 3]))
        self.assertEqual(float(target[0, 0, 2, 3]), 0.0)

        void_separated = torch.tensor(
            [[[0, 0, 255, 1, 1], [0, 0, 255, 1, 1], [0, 0, 255, 1, 1]]],
            dtype=torch.long,
        )
        target, valid = semantic_boundary_targets(void_separated)
        self.assertGreater(int(target.sum()), 0)
        self.assertFalse(bool(valid[0, 0, 1, 2]))
        self.assertEqual(float(target[0, 0, 1, 2]), 0.0)

        padding_only = torch.tensor(
            [[[0, 0, 255, 255], [0, 0, 255, 255]]], dtype=torch.long
        )
        target, _ = semantic_boundary_targets(padding_only)
        self.assertEqual(int(target.sum()), 0)

    def test_lovasz_uses_present_classes_and_ignores_255(self) -> None:
        labels = torch.tensor([[[0, 1], [255, 1]]], dtype=torch.long)
        good = torch.full((1, 3, 2, 2), -4.0)
        good[0, 0, 0, 0] = 4.0
        good[0, 1, 0, 1] = 4.0
        good[0, 1, 1, 1] = 4.0
        bad = -good
        good_loss = lovasz_softmax_loss(good, labels)
        bad_loss = lovasz_softmax_loss(bad, labels)
        self.assertLess(float(good_loss), float(bad_loss))

        changed_ignore = good.clone()
        changed_ignore[:, :, 1, 0] = torch.tensor([100.0, -100.0, 50.0])
        self.assertAlmostEqual(
            float(good_loss),
            float(lovasz_softmax_loss(changed_ignore, labels)),
            places=6,
        )

    def test_composite_loss_is_finite_and_backward_safe(self) -> None:
        generator = torch.Generator().manual_seed(42)
        labels = torch.randint(0, 4, (2, 12, 16), generator=generator)
        labels[0, :2, :3] = 255
        outputs = {
            "logits": torch.randn(2, 21, 12, 16, generator=generator, requires_grad=True),
            "aux_logits": torch.randn(
                2, 21, 12, 16, generator=generator, requires_grad=True
            ),
            "edge_logits": torch.randn(
                2, 1, 12, 16, generator=generator, requires_grad=True
            ),
        }
        losses = GCBSegLoss()(outputs, labels)
        self.assertEqual(
            set(losses),
            {"total", "ce", "lovasz", "edge", "edge_bce", "edge_dice", "aux"},
        )
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        expected = (
            losses["ce"]
            + 0.5 * losses["lovasz"]
            + 0.2 * losses["edge"]
            + 0.4 * losses["aux"]
        )
        torch.testing.assert_close(losses["total"], expected)
        losses["total"].backward()
        self.assertTrue(
            all(
                value.grad is not None and torch.isfinite(value.grad).all()
                for value in outputs.values()
            )
        )

    def test_missing_required_heads_are_rejected(self) -> None:
        labels = torch.zeros(1, 4, 4, dtype=torch.long)
        logits = torch.randn(1, 21, 4, 4)
        with self.assertRaises(ValueError):
            GCBSegLoss()({"logits": logits}, labels)


if __name__ == "__main__":
    unittest.main()
