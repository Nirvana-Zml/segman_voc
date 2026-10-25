# Step 0081: Implement and test MiT acquisition and SegFormer smoke tooling

- Time: 2026-10-26T00:12:18+08:00
- Milestone: `04-transformer-baseline`
- Status: **Passed**

## Objective and outcome

Added deterministic HF-to-native MiT conversion coverage, expanded the resource-safe AdamW schedule contract, documented reproduction commands, and implemented a provenance-gated pretrained SegFormer optimizer-step smoke.

## Commands

- `Run focused conversion SegFormer config tests; rerun acquisition tool; run full unittest discovery`

## Changed files

- `tools/acquire_mit_b1_weights.py; tools/smoke_segformer.py; tests/test_mit_weight_conversion.py; tests/test_config_contract.py; configs/segformer_b1.yaml; README and project docs`

## Evidence

- `Focused tests 9 of 9 passed in 1.096s; idempotent acquisition rerun preserved hashes; full container suite 29 of 29 passed in 5.894s`

## Decisions and assumptions

- Require smoke to match the provenance converted hash and use disjoint exhaustive encoder and decoder AdamW groups.

## Next action

Run the pretrained 384 by 384 batch-2 CUDA optimizer-step smoke.
