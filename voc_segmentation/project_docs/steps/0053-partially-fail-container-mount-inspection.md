# Step 0053: Partially fail container mount inspection

- Time: 2026-10-25T00:10:12+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The inspection assumed a root docker-compose.yml that is absent, while docker inspect succeeded and proved the container mounts only data/ and voc_segmentation/. The .git directory is not visible in-container.

## Commands

- `Get-Content docker-compose.yml; docker inspect voc2012-gcb --format {{json .Mounts}}`

## Changed files

- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `docker-compose.yml not found; docker inspect lists /data and /workspace/voc_segmentation mounts only`

## Decisions and assumptions

- Require explicit host commit injection for reportable container runs rather than treating missing in-container Git state as clean.

## Next action

Add and test a validated --git-commit input, then document the host command.
