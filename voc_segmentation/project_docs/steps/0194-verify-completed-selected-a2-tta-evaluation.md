# Step 0194: Verify completed selected A2 TTA evaluation

- Time: 2026-10-29T08:38:02+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Verified exit-0 full official-val TTA and independently reproduced all confusion-matrix metrics: mIoU 0.786179, a 0.010162 gain over the locked single-scale baseline.

## Commands

- `independently recompute metrics from runs/gcb_selected_a2_tta_val/result.json and verify terminal Docker state`

## Changed files

- `voc_segmentation/artifacts/selected_a2_tta_terminal_audit.json; voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md`

## Evidence

- `1449 samples; 244886857 valid pixels; result SHA 0f371a4f...; audit SHA 09dc0100...; elapsed 592.572 s; peak CUDA 646150144 bytes`

## Decisions and assumptions

- Accept the validation-proven six-pass TTA because 0.7861791234 is strictly greater than 0.7760166573.

## Next action

Persist the selected checkpoint/inference contract and close formal selection.
