from __future__ import annotations

import unittest

import torch

from engine import poly_lr_factor, set_learning_rate


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


if __name__ == "__main__":
    unittest.main()
