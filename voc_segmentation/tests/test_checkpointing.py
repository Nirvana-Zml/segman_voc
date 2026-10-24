from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
TEST_TMP_ROOT = APP_ROOT / ".test_tmp"
TEST_TMP_ROOT.mkdir(exist_ok=True)

from engine import load_checkpoint, save_checkpoint  # noqa: E402


class CheckpointingTest(unittest.TestCase):
    def test_round_trip_preserves_model_optimizer_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TMP_ROOT) as temp:
            path = Path(temp) / "diagnostic.pt"
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            loss = model(torch.ones(1, 3)).sum()
            loss.backward()
            optimizer.step()
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            info = save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scaler=None,
                step=7,
                config={"model": {"num_classes": 21}},
                metrics={"loss": 0.25},
                diagnostic_only=True,
            )
            for parameter in model.parameters():
                parameter.data.zero_()

            payload = load_checkpoint(path, model=model, optimizer=optimizer)

            self.assertEqual(payload["step"], 7)
            self.assertTrue(payload["diagnostic_only"])
            self.assertEqual(payload["metrics"]["loss"], 0.25)
            self.assertEqual(info["bytes"], path.stat().st_size)
            self.assertEqual(len(info["sha256"]), 64)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, expected[name])


if __name__ == "__main__":
    unittest.main()
