# Step 0031: Interrupt slow bind-mount SBD conversion

- Time: 2026-10-24T19:53:11+08:00
- Milestone: `02-data-audit-and-sbd`
- Status: **In progress**

## Objective and outcome

Interrupted the first full conversion after 694 of 9,118 masks because Windows bind-mount small-file I/O made progress impractically slow. No successful result was claimed from the partial output.

## Commands

- `docker exec voc2012-gcb python tools/prepare_voc_sbd.py`

## Changed files

None.

## Evidence

- `Observed progress reached 694/9118 before Ctrl+C; process exited after KeyboardInterrupt.`

## Decisions and assumptions

- Run archive extraction and conversion in container-local /tmp storage, then copy the fully validated artifacts to the host.

## Next action

Repeat conversion in container-local storage and verify host-side artifact counts after copying.
