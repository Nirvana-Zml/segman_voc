# Step 0036: Fail first DeepLabV3+ CUDA smoke on palette mask

- Time: 2026-10-24T23:12:21+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The first official-VOC CUDA smoke stopped in dataset loading before any forward pass because the loader rejected Pillow palette-mode masks. Official VOC PNGs store class IDs as palette indices, so P mode is valid single-channel label data.

## Commands

- `docker exec voc2012-gcb python tools/smoke_deeplabv3plus.py`

## Changed files

- `voc_segmentation/configs/deeplabv3plus_r50.yaml`
- `voc_segmentation/tools/smoke_deeplabv3plus.py`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`
- `voc_segmentation/project_docs/STATUS.md`

## Evidence

- `ValueError: mask must be single-channel L mode: .../SegmentationClass/2008_005679.png`
- `Failure occurred in VOCSegmentationDataset.__getitem__; no forward, loss, backward, optimizer step, result JSON, or checkpoint completed.`

## Decisions and assumptions

- Read P-mode mask indices through a uint8 array and reconstruct mode L; do not use PIL convert('L'), which would apply palette colors.

## Next action

Add a P-mode regression test, fix the loader, rerun all tests, then repeat the CUDA smoke.
