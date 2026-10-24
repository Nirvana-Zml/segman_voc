# Step 0039: Fix VOC palette-mask loading

- Time: 2026-10-24T23:17:40+08:00
- Milestone: `03-cnn-baseline`
- Status: **Passed**

## Objective and outcome

Updated the dataset to preserve class indices from official Pillow P-mode masks by reading their uint8 index arrays directly, and corrected the synthetic regression fixture with an explicit palette. The full suite passed.

## Commands

- `docker exec voc2012-gcb python -m unittest discover -s tests -v`

## Changed files

- `voc_segmentation/datasets/voc.py`
- `voc_segmentation/tests/test_voc_dataset.py`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`
- `voc_segmentation/project_docs/STATUS.md`

## Evidence

- `11 tests passed in 2.708 seconds inside voc2012-gcb.`
- `Real VOC P-mode indices are preserved; the regression fixture retains class index 20 after PNG save/reload.`

## Decisions and assumptions

- Normalize both L and P masks through their raw 2D uint8 index arrays; never use palette-to-luminance conversion for semantic labels.

## Next action

Rerun the hardened CUDA smoke and accept it only with a finite loss and gradient norm.
