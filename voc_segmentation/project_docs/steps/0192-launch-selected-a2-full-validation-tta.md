# Step 0192: Launch selected A2 full validation TTA

- Time: 2026-10-29T08:23:09+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **In progress**

## Objective and outcome

Launched one reportable 1449-image official-val six-pass TTA evaluation from clean commit f4f01e8; the first sample completed and the CUDA/process state is healthy.

## Commands

- `docker exec -d voc2012-gcb sh /workspace/voc_segmentation/runs/gcb_selected_a2_tta_val.launch.sh`

## Changed files

- `voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md; ignored runs/gcb_selected_a2_tta_val artifacts`

## Evidence

- `wrapper/evaluator PIDs 345/353; batch 1/1449; GPU 813/8188 MiB at 17%; peak allocation 577077248 bytes; wrapper SHA b959918a...`

## Decisions and assumptions

- Let the validation-only evaluator finish; retain TTA only if its full-val mIoU exceeds 0.7760166572511324.

## Next action

Audit result.json and exit code, independently recompute metrics, then record the selected inference method.
