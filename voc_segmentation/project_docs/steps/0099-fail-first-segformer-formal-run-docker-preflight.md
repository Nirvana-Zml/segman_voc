# Step 0099: Fail first SegFormer formal-run Docker preflight

- Time: 2026-10-26T00:35:23+08:00
- Milestone: `04-transformer-baseline`
- Status: **Failed**

## Objective and outcome

The first read-only formal-run preflight could not connect to the Docker Desktop named pipe from the managed shell, so container, GPU, disk, process, output, and weight checks did not execute and no training process started.

## Commands

- `docker inspect voc2012-gcb; docker exec voc2012-gcb sh -lc [GPU, disk, process, output and weight checks]`

## Changed files

- `project_docs/INDEX.md; project_docs/steps/0099-fail-first-segformer-formal-run-docker-preflight.md`

## Evidence

- `Both Docker calls returned permission denied for npipe dockerDesktopLinuxEngine.`

## Decisions and assumptions

- Request approved Docker access and rerun the complete preflight unchanged before launching.

## Next action

Repeat the preflight with approved Docker Desktop access.
