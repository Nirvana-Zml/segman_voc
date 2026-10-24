# Milestone 03: SegMAN-S one-iteration smoke run

- State: **Passed**
- Last updated: 2026-10-24 16:15 +08:00

## Objective

Verify a real VOC batch can complete SegMAN-S loading, forward pass, loss,
backpropagation, optimizer step, checkpoint writing, and TensorBoard logging.

## Observed result

- Iterations completed: 1
- Training loss: 2.99777
- Segmentation accuracy reported: 10.71496
- Peak memory reported by the training log: 2636 MB
- Checkpoint and TensorBoard event were written.

## Evidence

- `../../work_dirs/smoke_s/20261024_062943.log.json`
- `../../work_dirs/smoke_s/iter_1.pth`
- `../../work_dirs/smoke_s/tf_logs/`
- `../../work_dirs/smoke_s/segman_s_voc12.py`

## Acceptance result

Passed as diagnostic evidence only. `iter_1.pth` is not eligible for final
evaluation or test400 submission.

## Next action

Use the formal configuration for the reportable baseline.
