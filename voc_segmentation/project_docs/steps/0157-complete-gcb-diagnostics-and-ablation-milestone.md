# Step 0157: Complete GCB diagnostics and ablation milestone

- Time: 2026-10-28T08:05:32+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Created and hash-verified the A0-A4 evidence summary, selected A2 gated GFF solely by official-validation mIoU, and advanced project status to formal training and selection.

## Commands

- `validate artifacts/gcb_diagnostics.json hashes and selection; run project_status.py`

## Changed files

- `Ignored artifacts/gcb_diagnostics.json; STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `gcb_diagnostics.json SHA 0011ddfd...; A0/A1/A2/A3/A4 best mIoU 0.733889/0.736626/0.755643/0.726746/0.739100; project status milestone 05 complete`

## Decisions and assumptions

- Use A2 gated GFF without BAB, edge loss, or Lovasz as the milestone-06 trainaug architecture.

## Next action

Create, test, and preflight the reportable A2-derived SBD trainaug configuration.
