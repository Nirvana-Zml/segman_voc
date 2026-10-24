# Step 0049: Fail first formal-trainer container regression invocation

- Time: 2026-10-24T23:59:19+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The managed host shell was denied access to the Docker Desktop named pipe before the container test process started. This is an invocation-boundary failure, not a unit-test result.

## Commands

- `docker exec voc2012-gcb python -m unittest discover -s tests -v`

## Changed files

- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `Host error: permission denied on npipe:////./pipe/dockerDesktopLinuxEngine`

## Decisions and assumptions

- Retain the implementation unchanged and rerun through the approved Docker execution boundary.

## Next action

Run the complete in-container regression suite with approved Docker access, then correct any actual test failures.
