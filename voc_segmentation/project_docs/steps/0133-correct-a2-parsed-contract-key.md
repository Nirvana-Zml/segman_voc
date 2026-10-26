# Step 0133: Correct A2 parsed-contract key

- Time: 2026-10-26T18:59:10+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The first read-only A1/A2 comparison requested nonexistent training key val_interval and stopped with KeyError. The configuration uses validate_every; no file or artifact changed.

## Commands

- `docker exec voc2012-gcb python -c parsed A1/A2 contract audit`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `KeyError: val_interval; direct config inspection showed validate_every: 1000.`

## Decisions and assumptions

- Correct the audit key and rerun the complete parsed-contract comparison.

## Next action

Run the corrected comparison and accept preflight only if all common sections and intended deltas match.
