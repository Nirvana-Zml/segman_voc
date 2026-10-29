# Step 0188: Pass selected A2 TTA CUDA diagnostic

- Time: 2026-10-29T08:10:28+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Ran six-pass TTA on two official-val images; output/provenance contracts passed with 577077248 peak CUDA bytes and no reportable metric selection.

## Commands

- `python tools/evaluate_gcb_tta.py --config configs/gcb_selected_a2_tta_val.yaml --output-dir runs/gcb_selected_a2_tta_val_diagnostic --diagnostic-only --validation-limit 2`

## Changed files

- `voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md; ignored diagnostic artifacts`

## Evidence

- `diagnostic result da8fafd0...; 2 samples; six logits/image; elapsed 1.811 s; accepted=false`

## Decisions and assumptions

- The subset mIoU is non-reportable; proceed only with full official-val evaluation.

## Next action

Commit source/docs, pass a reportable preflight, and launch the full 1449-image TTA evaluation.
