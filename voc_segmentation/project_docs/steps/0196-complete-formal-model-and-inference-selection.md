# Step 0196: Complete formal model and inference selection

- Time: 2026-10-29T08:38:07+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Fixed selected-A2 step-39000 best.pt plus validation-proven six-pass TTA, explicitly omitted unsupported ensemble members, and advanced project status to test400 inference.

## Commands

- `PowerShell ConvertFrom-Json/Get-FileHash; python project_status.py --repo .`

## Changed files

- `voc_segmentation/artifacts/model_selection.json; voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md`

## Evidence

- `model selection SHA 67e40e95...; selected val mIoU 0.786179; project_status formal_training_and_selection=complete and next_stage=test400_predictions`

## Decisions and assumptions

- Use only best.pt SHA d2e6fb31... with 0.75/1.0/1.25 horizontal-flip mean-logit TTA; do not tune on test400.

## Next action

Implement or verify test400 TTA prediction tooling, generate exactly 400 masks, and validate dimensions, names, dtype, channels, and class IDs.
