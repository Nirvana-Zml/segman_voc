# Step 0029: Fail first SBD conversion fixture test

- Time: 2026-10-24T19:53:03+08:00
- Milestone: `02-data-audit-and-sbd`
- Status: **Failed**

## Objective and outcome

The first synthetic MAT fixture failed because uninitialized object cells in GTcls.Boundaries were serialized as None; this was a test-fixture defect, not a converter failure.

## Commands

- `docker exec voc2012-gcb python -m unittest discover -s tests -v`

## Changed files

None.

## Evidence

- `tests.test_sbd_conversion.SbdConversionTest errored while scipy.io.savemat serialized an uninitialized Boundaries cell containing None.`

## Decisions and assumptions

- Initialize every one of the 20 boundary cells with a valid zero array so the fixture matches the SBD struct shape.

## Next action

Correct the fixture and rerun the entire focused suite inside voc2012-gcb.
