# Step 0152: Retry A4 resume Docker inspections

- Time: 2026-10-28T00:03:38+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

Ordinary Docker access was denied by the Windows named pipe during wrapper validation and again during post-launch inspection. Approved retries passed; the launch itself was issued once and the active job was not duplicated.

## Commands

- `docker exec voc2012-gcb sh -n and hash checks; docker exec voc2012-gcb live audit/process/GPU checks; retry both with approved access`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger only`

## Evidence

- `Two permission-denied batches followed by successful wrapper/hash and live-process audits`

## Decisions and assumptions

- Treat both as access-layer inspection failures, not training failures; retain the single successful background launch.

## Next action

Record and verify the active A4 recovery launch.
