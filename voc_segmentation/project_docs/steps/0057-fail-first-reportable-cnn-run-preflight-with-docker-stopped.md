# Step 0057: Fail first reportable CNN run preflight with Docker stopped

- Time: 2026-10-25T07:48:08+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The first reportable DeepLabV3+ run preflight could not connect to Docker Desktop's Linux Engine pipe. GPU, disk, and existing-process checks did not execute; the output directory did not exist and no training process started.

## Commands

- `docker exec voc2012-gcb sh -lc nvidia-smi, df, and pgrep preflight`

## Changed files

- `voc_segmentation/project_docs/STATUS.md`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `Docker error: dockerDesktopLinuxEngine pipe not found`
- `Host run directory absent before attempted launch`

## Decisions and assumptions

- Restart Docker Desktop and rerun all preflight checks; do not create or accept a training run from this attempt.

## Next action

Start Docker Desktop, confirm voc2012-gcb and GPU availability, then launch the clean-commit formal baseline.
