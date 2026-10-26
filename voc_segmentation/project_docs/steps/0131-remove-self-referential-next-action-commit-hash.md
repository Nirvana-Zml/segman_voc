# Step 0131: Remove self-referential next-action commit hash

- Time: 2026-10-26T18:47:54+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Rephrased the A2 next action to use the current clean ledger revision instead of a literal commit that becomes stale when the status correction itself is committed.

## Commands

- `Patch STATUS.md and milestone 05 next action; git diff --check`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Next action is stable across its own documentation commit and still requires a clean revision before A2 launch.`

## Decisions and assumptions

- Do not embed the current documentation HEAD in forward-looking status text when committing that text necessarily creates a newer HEAD.

## Next action

Run the A2 gated-GFF formal preflight from the current clean ledger revision.
