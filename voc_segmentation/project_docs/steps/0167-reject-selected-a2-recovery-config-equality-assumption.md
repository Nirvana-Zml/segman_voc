# Step 0167: Reject selected A2 recovery config-equality assumption

- Time: 2026-10-28T12:52:12+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

Read-only recovery audit stopped because the temporary checker incorrectly required source YAML to equal resolved YAML including its expected runtime block; no trainer started.

## Commands

- `docker exec voc2012-gcb sh -lc '<resource checks>; cd /workspace/voc_segmentation && python runs/gcb_selected_a2_trainaug/resume_preflight.py'`

## Changed files

- `Ignored runtime checker only`

## Evidence

- `AssertionError at source_config == resolved_config; GPU 0/8188 MiB; no trainer`

## Decisions and assumptions

- Compare all source sections exactly and validate the resolved runtime block separately.

## Next action

Correct the checker and rerun the complete read-only audit.
