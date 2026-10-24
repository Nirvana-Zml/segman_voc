# Step 0050: Reject zero-learning-rate pretrained trainer smoke

- Time: 2026-10-25T00:01:12+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The first one-iteration pretrained diagnostic exercised forward, backward, checkpointing, TensorBoard, and limited native-size validation, but polynomial decay set its only optimizer-step learning rate to zero. The artifact is rejected as parameter-update evidence.

## Commands

- `docker exec voc2012-gcb python tools/train_deeplabv3plus.py --diagnostic-only --output-dir artifacts/cnn_baseline/pretrained_formal_smoke --max-iterations 1 --validate-every 1 --validation-limit 2`

## Changed files

- `voc_segmentation/engine/training.py`
- `voc_segmentation/tests/test_training.py`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `artifacts/cnn_baseline/pretrained_formal_smoke/result.json: diagnostic_only=true, learning rate logged as 0.0`

## Decisions and assumptions

- Keep the rejected artifact immutable and rerun into pretrained_formal_smoke_v2 after testing a one-step schedule special case.

## Next action

Run the full regression suite, then rerun the pretrained diagnostic and independently verify its logs and checkpoint.
