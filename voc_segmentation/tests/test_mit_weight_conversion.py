from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from models.encoders import build_mit_b1  # noqa: E402
from tools.acquire_mit_b1_weights import convert_huggingface_state_dict  # noqa: E402


def native_to_huggingface_fixture(
    native: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    fixture: dict[str, torch.Tensor] = {}
    for stage_index in range(4):
        source_prefix = f"patch_embed{stage_index + 1}"
        target_prefix = f"segformer.encoder.patch_embeddings.{stage_index}"
        for suffix in ("weight", "bias"):
            fixture[f"{target_prefix}.proj.{suffix}"] = native[
                f"{source_prefix}.proj.{suffix}"
            ]
            fixture[f"{target_prefix}.layer_norm.{suffix}"] = native[
                f"{source_prefix}.norm.{suffix}"
            ]

    for stage_index in range(4):
        for block_index in range(2):
            source_prefix = f"block{stage_index + 1}.{block_index}"
            target_prefix = f"segformer.encoder.block.{stage_index}.{block_index}"
            direct_mappings = {
                "norm1": "layer_norm_1",
                "attn.q": "attention.self.query",
                "attn.proj": "attention.output.dense",
                "norm2": "layer_norm_2",
                "mlp.fc1": "mlp.dense1",
                "mlp.dwconv.dwconv": "mlp.dwconv.dwconv",
                "mlp.fc2": "mlp.dense2",
            }
            if stage_index < 3:
                direct_mappings.update(
                    {
                        "attn.sr": "attention.self.sr",
                        "attn.norm": "attention.self.layer_norm",
                    }
                )
            for source_name, target_name in direct_mappings.items():
                for suffix in ("weight", "bias"):
                    fixture[f"{target_prefix}.{target_name}.{suffix}"] = native[
                        f"{source_prefix}.{source_name}.{suffix}"
                    ]
            for suffix in ("weight", "bias"):
                key, value = native[
                    f"{source_prefix}.attn.kv.{suffix}"
                ].chunk(2, dim=0)
                fixture[f"{target_prefix}.attention.self.key.{suffix}"] = key
                fixture[f"{target_prefix}.attention.self.value.{suffix}"] = value

    for stage_index in range(4):
        for suffix in ("weight", "bias"):
            fixture[f"segformer.encoder.layer_norm.{stage_index}.{suffix}"] = native[
                f"norm{stage_index + 1}.{suffix}"
            ]
    fixture["classifier.weight"] = torch.zeros(1000, 512)
    fixture["classifier.bias"] = torch.zeros(1000)
    return fixture


class MitWeightConversionTest(unittest.TestCase):
    def test_huggingface_conversion_exactly_recovers_native_state(self) -> None:
        native = build_mit_b1().state_dict()
        fixture = native_to_huggingface_fixture(native)

        converted = convert_huggingface_state_dict(fixture)

        self.assertEqual(set(converted), set(native))
        self.assertEqual(len(converted), 176)
        for name, expected in native.items():
            with self.subTest(name=name):
                self.assertTrue(torch.equal(converted[name], expected))


if __name__ == "__main__":
    unittest.main()
