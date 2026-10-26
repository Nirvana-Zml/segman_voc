# Step 0108: Correct VOC void-band boundary targets

- Time: 2026-10-26T08:52:20+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Boundary targets now recover semantic contours whose different valid endpoint classes are separated by one to three consecutive VOC 255 pixels, while keeping ignore pixels invalid and padding-only void regions edge-free.

## Commands

- `docker exec voc2012-gcb python -m unittest tests.test_gcb_seg tests.test_gcb_losses -v; docker exec voc2012-gcb python -m unittest discover -s tests -v`

## Changed files

- `voc_segmentation/losses/gcb.py; voc_segmentation/tests/test_gcb_losses.py`

## Evidence

- `Focused GCB tests passed 8/8 in 3.045 seconds; complete container suite passed 40/40 in 5.253 seconds; tests cover recovered void-separated boundaries and zero padding-only boundaries.`

## Decisions and assumptions

- Use max_void_width=3 for planned VOC runs; bridging requires valid unequal classes on both ends and a fully ignored intervening run.

## Next action

Rerun the deterministic pretrained GCB-Seg CUDA optimizer-step smoke and inspect its saved evidence.
