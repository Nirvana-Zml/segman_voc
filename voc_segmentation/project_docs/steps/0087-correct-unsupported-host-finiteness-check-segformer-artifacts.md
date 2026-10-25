# Step 0087: Correct unsupported host finiteness check for SegFormer artifacts

- Time: 2026-10-26T00:19:47+08:00
- Milestone: `04-transformer-baseline`
- Status: **Failed**

## Objective and outcome

The first host artifact audit used System.Double.IsFinite, which is unavailable in the installed PowerShell/.NET runtime and emitted method errors; hash comparisons still ran, but the two finite-value booleans were rejected as evidence.

## Commands

- `Parse result and history with PowerShell and call System.Double.IsFinite`

## Changed files

- `project_docs/INDEX.md; project_docs/steps/0087-correct-unsupported-host-finiteness-check-segformer-artifacts.md`

## Evidence

- `PowerShell reported that System.Double has no IsFinite method for each history value; no experiment artifact changed.`

## Decisions and assumptions

- Do not cite the affected PowerShell booleans; rerun finite-value validation with Python math.isfinite.

## Next action

Run a corrected independent audit in the target container.
