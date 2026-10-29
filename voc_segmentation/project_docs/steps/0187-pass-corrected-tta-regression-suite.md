# Step 0187: Pass corrected TTA regression suite

- Time: 2026-10-29T08:10:28+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Added explicit VOC/model-selection fields to the TTA config and reran the complete suite successfully.

## Commands

- `python -m unittest discover -s tests -v`

## Changed files

- `voc_segmentation/configs/gcb_selected_a2_tta_val.yaml; voc_segmentation/tools/evaluate_gcb_tta.py`

## Evidence

- `53/53 tests passed in 5.315 seconds`

## Decisions and assumptions

- Keep the global all-config contract enforced for evaluation configs.

## Next action

Run a bounded CUDA TTA diagnostic on official validation samples.
