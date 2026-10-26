# Step 0112: Pass pretrained GCB two-sample overfit gate

- Time: 2026-10-26T09:04:42+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

The deterministic pretrained GCB-Seg diagnostic completed 100 fixed-sample AMP updates and passed the complete-composite-objective decrease threshold with finite loss components and gradients.

## Commands

- `docker exec voc2012-gcb python tools/overfit_gcb_seg.py`

## Changed files

- `Gitignored artifacts/gcb_diagnostics/overfit_two_samples result, history, checkpoint, and TensorBoard evidence; milestone and status documentation`

## Evidence

- `Composite objective 4.708998 to 0.603445, ratio 0.128147 <= 0.5; 100 finite records; AMP 1024 to 1024; 2051 positive boundaries; same-sample pixel accuracy 0.994826; 24.154 seconds; peak CUDA 467688960 bytes.`

## Decisions and assumptions

- Accept only as diagnostic evidence. The checkpoint and same-sample metrics are not eligible for validation selection, comparison, or submission.

## Next action

Independently audit the history formulas, checkpoint, TensorBoard series, provenance, optimizer coverage, and file hashes.
