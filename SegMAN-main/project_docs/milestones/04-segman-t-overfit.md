# Milestone 04: SegMAN-T two-sample overfit diagnostic

- State: **Passed (diagnostic only)**
- Last updated: 2026-10-24 16:15 +08:00

## Objective

Verify that image/label alignment, the segmentation loss, backpropagation, and
optimization can fit two deterministic real VOC2012 samples.

## Observed result

- Iterations completed: 100
- Initial loss: 2.97378
- Final loss: 0.15274
- Final reported segmentation accuracy: 99.0037
- Peak memory reported: 1202 MB
- Final checkpoint was written at iteration 100.

## Evidence

- `../../work_dirs/overfit_t_complete/20261024_065645.log.json`
- `../../work_dirs/overfit_t_complete/iter_100.pth`
- `../../work_dirs/overfit_t_complete/tf_logs/`
- `../../work_dirs/overfit_t_complete/segman_t_voc12_overfit.py`

## Acceptance result

Passed. This proves the learning path on a tiny diagnostic sample; it is not a
validation score and its checkpoint must never be used for final predictions.

## Next action

Start a non-diagnostic formal baseline and evaluate on the independent
validation split.
