# Step 0150: Correct A4 recovery audit invocation

- Time: 2026-10-27T23:42:46+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The first container call was denied by the Docker named pipe; the approved retry reached the read-only checker, whose incorrect event key caused a local assertion after checkpoint and hash checks. No trainer started and no run evidence changed.

## Commands

- `docker exec voc2012-gcb python .test_tmp/a4_recovery_audit.py; corrected checker from event to type`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger only`

## Evidence

- `Docker permission-denied output; AssertionError at history classification; process table showed no trainer and GPU 0 percent`

## Decisions and assumptions

- Reject the incomplete audit and rerun after correcting only the temporary read-only checker.

## Next action

Run the corrected full A4 recovery preflight.
