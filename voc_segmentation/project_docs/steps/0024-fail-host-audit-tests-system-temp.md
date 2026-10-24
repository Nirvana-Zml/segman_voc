# Step 0024: Fail host audit tests in system temp

- Time: 2026-10-24T19:28:06+08:00
- Milestone: `02-data-audit-and-sbd`
- Status: **Failed**

## Objective and outcome

The expanded six-test suite ran four existing tests successfully but both new data-audit tests failed because the managed Windows sandbox denied writes below the system temporary directory.

## Commands

- `python -m unittest discover -s voc_segmentation/tests -v`

## Changed files

None.

## Evidence

- `4 tests passed; 2 data-audit tests errored with WinError 5 under C:/Users/.../AppData/Local/Temp`

## Decisions and assumptions

- The failure is in the host test write location; rerun after constraining temporary files to the workspace.

## Next action

Move test temporary files into a gitignored project-local directory and retry.
