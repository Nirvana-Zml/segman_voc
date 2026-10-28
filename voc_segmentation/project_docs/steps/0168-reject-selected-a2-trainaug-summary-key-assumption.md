# Step 0168: Reject selected A2 trainaug summary key assumption

- Time: 2026-10-28T12:52:12+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

The corrected read-only audit stopped because the temporary checker used a nonexistent trainaug overlap key; all preceding hashes passed and no trainer started.

## Commands

- `docker exec voc2012-gcb sh -lc '<resource checks>; cd /workspace/voc_segmentation && python runs/gcb_selected_a2_trainaug/resume_preflight.py'`

## Changed files

- `Ignored runtime checker only`

## Evidence

- `KeyError official_val_overlap_count; trainaug_summary.json stores train_val_overlap`

## Decisions and assumptions

- Use the saved summary schema field train_val_overlap.

## Next action

Correct the checker and rerun the complete read-only audit.
