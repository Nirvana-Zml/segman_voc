# Step 0191: Retry selected A2 TTA detached launch

- Time: 2026-10-29T08:23:06+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

The first detached launch was denied by the Windows Docker named pipe before container access; it created no process and no output directory.

## Commands

- `docker exec -d voc2012-gcb sh /workspace/voc_segmentation/runs/gcb_selected_a2_tta_val.launch.sh`

## Changed files

- `voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md`

## Evidence

- `Docker named-pipe access denied; subsequent audit found only the approved-retry process pair`

## Decisions and assumptions

- Retry the same preflighted wrapper once with approved Docker access.

## Next action

Launch once with approved access and audit the first completed TTA sample.
