# Milestone 07: Test400 inference and validation

- State: **Not started**
- Last updated: 2026-10-24 16:15 +08:00

## Objective

Generate final class-index masks from the selected non-diagnostic checkpoint and
prove they satisfy the submission format.

## Required evidence

- The chosen checkpoint is traced to milestone 06 validation evidence
- Exactly 400 PNG files with input-matching stems and dimensions
- Single-channel `L` mode, `uint8`, and values only in `0..20`
- Passing JSON report from `verify_predictions.py`
- No color visualization or extra file in the submission mask directory

## Current evidence

No final prediction directory or passing validation report exists. The one-file
`work_dirs/smoke_predictions/` artifact is a tooling diagnostic, not a result.

## Next action

Wait for final model selection, then run inference and save
`work_dirs/prediction_validation.json`.
