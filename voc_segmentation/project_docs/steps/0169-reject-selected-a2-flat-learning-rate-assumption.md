# Step 0169: Reject selected A2 flat learning-rate assumption

- Time: 2026-10-28T12:52:12+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

The third read-only audit stopped because the temporary checker assumed flat encoder/decoder LR fields instead of the saved learning_rates list; no trainer started.

## Commands

- `docker exec voc2012-gcb sh -lc '<resource checks>; cd /workspace/voc_segmentation && python runs/gcb_selected_a2_trainaug/resume_preflight.py'`

## Changed files

- `Ignored runtime checker only`

## Evidence

- `KeyError encoder_lr; history records contain learning_rates and nested losses`

## Decisions and assumptions

- Validate the actual history schema, all finite scalar values, and exact CE-only loss composition.

## Next action

Correct the checker and rerun the complete read-only audit.
