# Step 0193: Retry TTA terminal Docker inspection

- Time: 2026-10-29T08:37:59+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

The first terminal Docker inspection was denied at the Windows named pipe after the evaluator had already written exit 0 and result.json; no artifact or process state changed.

## Commands

- `docker exec voc2012-gcb sh -lc <terminal read-only inspection>`

## Changed files

- `voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md`

## Evidence

- `Host-visible result.json and exit_code.txt existed; Docker API named-pipe access denied`

## Decisions and assumptions

- Retry the identical read-only terminal inspection with approved access.

## Next action

Verify process exit, released GPU, exit code, and container-side hashes.
