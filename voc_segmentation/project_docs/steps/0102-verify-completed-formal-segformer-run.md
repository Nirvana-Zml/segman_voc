# Step 0102: Verify completed formal SegFormer run

- Time: 2026-10-26T08:31:23+08:00
- Milestone: `04-transformer-baseline`
- Status: **Passed**

## Objective and outcome

Independently verified the terminal 20000-step reportable SegFormer run, all 20 complete official validations, six bounded AMP recoveries, validation-selected best checkpoint, final checkpoint, and every recorded artifact hash.

## Commands

- `Read exit code and terminal artifacts; independently parse history and validation JSON; load best and last checkpoints; recompute hashes and best-mIoU selection`

## Changed files

- `STATUS.md; milestones/04-transformer-baseline.md; project step ledger; existing gitignored formal-run artifacts remained unchanged`

## Evidence

- `Exit 0; 20000 unique train steps; 20 full 1449-image validations; 6 valid overflows; best step 19000 mIoU 0.7338885223855234; result SHA-256 6b9fac34b0b6fb36c5e6a63f226d3c61592eb476717b05ae3c4edf72a2305ecd; all independent checks passed.`

## Decisions and assumptions

- Mark the Transformer baseline passed and use only non-diagnostic best.pt selected by full-validation mIoU for future comparisons.

## Next action

Align the project-status helper with the verified Transformer result path, then begin GCB-Seg diagnostics.
