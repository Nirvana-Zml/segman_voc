# Step 0120: Reject incomplete initial A1 formal preflight attempts

- Time: 2026-10-26T09:36:32+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

Initial read-only A1 preflight attempts were incomplete because several Docker queries hit Windows named-pipe permission denial and the first approved process regex matched its checking shell; no training process started and no formal output was created.

## Commands

- `Parallel docker exec GPU, disk, process, output, hash, and config checks; approved serial retry; first direct process-table query`

## Changed files

- `None; all attempts were read-only`

## Evidence

- `Three parallel queries and the first ps query returned permission denied; serial GPU/disk/hash/config checks passed, while pgrep reported only its own sh -lc command as PID 2400.`

## Decisions and assumptions

- Do not treat partial checks or a self-matching regex as a passed preflight; require approved direct process-table evidence.

## Next action

Rerun the process table under approved Docker access and accept the preflight only if every other saved check remains valid.
