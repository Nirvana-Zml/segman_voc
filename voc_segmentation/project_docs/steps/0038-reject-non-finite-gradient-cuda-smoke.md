# Step 0038: Reject non-finite-gradient CUDA smoke

- Time: 2026-10-24T23:15:29+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The corrected loader let the CUDA smoke reach forward and backward, but the unscaled gradient norm was NaN. Although the first script version wrote ok=true, this result is explicitly rejected because no finite optimizer step was proven.

## Commands

- `docker exec voc2012-gcb python tools/smoke_deeplabv3plus.py`

## Changed files

- `voc_segmentation/artifacts/cnn_baseline/smoke.json (rejected gitignored artifact)`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`
- `voc_segmentation/project_docs/STATUS.md`

## Evidence

- `loss=3.0377931594848633; gradient_norm=NaN; input/logits=(2,3,384,384)/(2,21,384,384); peak CUDA bytes=1065757696; elapsed=2.046 seconds.`

## Decisions and assumptions

- Treat any non-finite gradient norm as a hard failure even when loss is finite; lower the diagnostic AMP initial scale from 65536 to 1024 and assert finiteness before optimizer.step.

## Next action

Patch the smoke integrity gate, rerun regression tests, and repeat the CUDA smoke until a finite optimizer step completes.
