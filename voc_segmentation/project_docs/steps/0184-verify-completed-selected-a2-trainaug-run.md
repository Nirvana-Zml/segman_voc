# Step 0184: Verify completed selected A2 trainaug run

- Time: 2026-10-29T08:01:14+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Verified 40000 unique train steps, 40 recomputed full validations, 14 bounded AMP recoveries, finite checkpoints, and best mIoU 0.776017 at step 39000.

## Commands

- `docker exec voc2012-gcb sh -lc 'cd /workspace/voc_segmentation && python runs/gcb_selected_a2_trainaug/audit_terminal_result.py'`

## Changed files

- `voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md; ignored terminal audit artifact`

## Evidence

- `audit 14545e48...; result 79065492...; history 2be3ea7a...; best.pt d2e6fb31... step 39000; last.pt 06ddc3f8... step 40000`

## Decisions and assumptions

- Select step-39000 best.pt for single-scale baseline; keep milestone 06 open until val-proven TTA/ensemble decision.

## Next action

Evaluate 0.75/1.0/1.25 plus horizontal-flip logit TTA on official val and retain it only if mIoU improves.
