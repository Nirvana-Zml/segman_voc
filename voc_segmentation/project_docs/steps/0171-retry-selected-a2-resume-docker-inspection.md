# Step 0171: Retry selected A2 resume Docker inspection

- Time: 2026-10-28T12:58:08+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

An ordinary final wrapper/resource inspection was denied by the Docker Desktop named pipe; no resume command had been issued and the approved retry passed.

## Commands

- `docker exec voc2012-gcb sh -lc '<wrapper syntax, archive hash, process, and GPU checks>'`

## Changed files

- `None`

## Evidence

- `permission denied at npipe dockerDesktopLinuxEngine; subsequent approved check found GPU 0/8188 MiB and no trainer`

## Decisions and assumptions

- Use approved Docker access for the required final checks without changing the recovery contract.

## Next action

Validate the wrapper and preserved interruption artifacts, then launch once.
