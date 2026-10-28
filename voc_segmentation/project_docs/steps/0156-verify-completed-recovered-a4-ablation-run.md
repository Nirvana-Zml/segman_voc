# Step 0156: Verify completed recovered A4 ablation run

- Time: 2026-10-28T08:05:32+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Verified exit 0, 20000 unique train steps, 20 full official validations, six bounded AMP recoveries, recomputed all confusion metrics, and validated finite checkpoints and full-retention TensorBoard evidence. Best A4 is step 20000 at mIoU 0.739100.

## Commands

- `docker exec voc2012-gcb python .test_tmp/a4_terminal_audit.py; independent process/GPU/exit snapshot`

## Changed files

- `Ignored A4 terminal result, history, checkpoints and events; STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `result SHA fda20e6a...; history SHA 5a4c54e4...; best.pt SHA feec1bc0...; best/terminal mIoU 0.739100; no trainer; GPU 0 percent`

## Decisions and assumptions

- Accept A4 as complete; it improves A3 but trails A2 by 0.016543 mIoU, so A2 remains validation-selected.

## Next action

Consolidate verified A0-A4 evidence and close milestone 05.
