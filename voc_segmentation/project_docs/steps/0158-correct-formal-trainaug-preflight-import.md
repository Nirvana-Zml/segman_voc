# Step 0158: Correct formal trainaug preflight import

- Time: 2026-10-28T08:15:29+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

The first read-only actual-data preflight could not import the application packages because the temporary script omitted the app root from sys.path. No trainer or output was created; adding the root corrected the checker.

## Commands

- `docker exec voc2012-gcb python .test_tmp/formal_trainaug_preflight.py; add /workspace/voc_segmentation to sys.path`

## Changed files

- `STATUS.md; milestone 06; immutable step ledger only`

## Evidence

- `ModuleNotFoundError: No module named datasets; corrected retry passed`

## Decisions and assumptions

- Reject the incomplete preflight and rerun after correcting only temporary checker imports.

## Next action

Run the corrected actual-data trainaug preflight.
