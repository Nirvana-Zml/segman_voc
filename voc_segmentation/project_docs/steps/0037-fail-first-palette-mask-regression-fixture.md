# Step 0037: Fail first palette-mask regression fixture

- Time: 2026-10-24T23:13:54+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The new palette-mask test expected index 20 but the synthetic P-mode PNG reopened as index 1. Diagnostics showed this came from saving a palette image without an explicit palette; the real VOC sample remained P mode with valid indices.

## Commands

- `docker exec voc2012-gcb python -m unittest discover -s tests -v`
- `docker exec voc2012-gcb python -c <palette-index diagnostic>`

## Changed files

- `voc_segmentation/datasets/voc.py`
- `voc_segmentation/tests/test_voc_dataset.py`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`
- `voc_segmentation/project_docs/STATUS.md`

## Evidence

- `11 tests ran in 3.133 seconds: 10 passed and test_reads_official_split_without_label_conversion failed because observed index was 1 instead of 20.`
- `Synthetic image: before save P [20], after save P [1]; real VOC 2008_005679.png: P [0,12,255].`

## Decisions and assumptions

- Construct the synthetic P-mode fixture with Image.new, putdata, and an explicit 256-entry palette before saving.

## Next action

Correct the fixture, rerun all tests, then repeat the CUDA smoke.
