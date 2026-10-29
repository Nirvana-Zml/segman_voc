# Step 0189: Retry TTA Docker preflight access

- Time: 2026-10-29T08:20:52+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

The ordinary reportable TTA preflight was denied at the Windows Docker named pipe before container access; no evaluator or trainer started and no output artifact was created.

## Commands

- `docker exec voc2012-gcb sh -lc <read-only TTA preflight>`

## Changed files

- `voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md`

## Evidence

- `Docker named-pipe access denied; output directory remained absent`

## Decisions and assumptions

- Retry the identical read-only preflight with approved Docker access.

## Next action

Run the approved reportable TTA preflight without launching evaluation.
