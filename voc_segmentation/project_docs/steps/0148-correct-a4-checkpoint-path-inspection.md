# Step 0148: Correct A4 checkpoint path inspection

- Time: 2026-10-27T23:01:01+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The first read-only checkpoint loop embedded shell variable f in a PowerShell double-quoted command, so PowerShell expanded it before Docker and both iterations listed the run directory. Explicit-path checks then confirmed best.pt exists and last.pt is absent; no state changed.

## Commands

- `docker exec sh checkpoint loop with PowerShell-expanded variable; corrected explicit ls best.pt and test last.pt absent`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `The failed loop printed the directory listing twice; corrected checks found 183284062-byte best.pt and no last.pt.`

## Decisions and assumptions

- Use explicit container paths rather than nested shell variables in PowerShell command strings.

## Next action

Pause exact trainer PID and preserve a checkpoint-aligned recovery state.
