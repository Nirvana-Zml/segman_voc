# Step 0107: Reject degenerate first GCB CUDA boundary smoke

- Time: 2026-10-26T08:52:16+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The first pretrained GCB-Seg CUDA optimizer step completed and emitted ok=true, but audit found only 31 positive boundary pixels among 234318 valid pixels because VOC void bands separated object classes; the artifact was rejected and overwritten.

## Commands

- `docker exec voc2012-gcb python tools/smoke_gcb_seg.py; direct fixed-batch transition and class-count audit`

## Changed files

- `Initial ignored artifacts/gcb_diagnostics/smoke.json only; no accepted source state`

## Evidence

- `Optimizer update and finite values succeeded, but immediate-neighbor target contained 31/234318 positive boundaries; shift-distance audit found 4, 37, 176, and 727 differences at distances 1 through 4.`

## Decisions and assumptions

- Do not accept ok=true alone; boundary supervision must contain a non-degenerate positive signal. Correct target generation before overfit.

## Next action

Bridge only class-different valid endpoints across short all-255 VOC void bands, test padding safety, and rerun all tests.
