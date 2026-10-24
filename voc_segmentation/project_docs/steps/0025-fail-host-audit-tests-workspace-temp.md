# Step 0025: Fail host audit tests in workspace temp

- Time: 2026-10-24T19:28:18+08:00
- Milestone: `02-data-audit-and-sbd`
- Status: **Failed**

## Objective and outcome

The retry still failed because the managed Windows sandbox denied nested directories created by Python TemporaryDirectory even under voc_segmentation/.test_tmp; four unrelated tests continued to pass.

## Commands

- `python -m unittest discover -s voc_segmentation/tests -v`

## Changed files

- `.gitignore`
- `voc_segmentation/tests/test_data_audit.py`

## Evidence

- `4 tests passed; 2 data-audit tests errored with WinError 5 under voc_segmentation/.test_tmp`

## Decisions and assumptions

- Stop adapting production tests to the host sandbox and run the unchanged behavior checks in the actual voc2012-gcb target container.

## Next action

Run the complete test suite inside voc2012-gcb.
