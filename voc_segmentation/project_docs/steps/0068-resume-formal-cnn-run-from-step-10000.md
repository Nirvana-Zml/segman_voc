# Step 0068: Resume formal CNN run from step 10000

- Time: 2026-10-25T16:03:57+08:00
- Milestone: `03-cnn-baseline`
- Status: **In progress**

## Objective and outcome

Passed recovery preflight, preserved exit code 143, and resumed the formal run from verified step 10000 under clean commit 270f733. PID 66 is active and history continues uniquely from step 10001.

## Commands

- `docker exec -d voc2012-gcb sh /workspace/voc_segmentation/runs/deeplabv3plus_r50_voc_train/resume_from_010000.sh; docker exec voc2012-gcb <process, GPU, log, and JSONL checks>`

## Changed files

- `voc_segmentation/runs/deeplabv3plus_r50_voc_train/resume_from_010000.sh (gitignored); exit_code.user_stop_step_010000.txt (gitignored); voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `Started 2026-10-25 16:01:32 +08:00; PID 66; commit 270f7331552b93250e5c979b6c2d345e57384c5b; history kept=10011 dropped=0; at step 10044: 10044 unique train records, 10 validations, 1 AMP overflow, 0 duplicates; exit_code.txt absent; initial GPU 1929 MiB at 67%`

## Decisions and assumptions

- Continue the same 20000-step formal run; treat the launch as in-progress evidence until terminal artifacts pass independent verification. Two monitoring-query quoting/key errors were corrected without affecting training.

## Next action

Monitor PID 66 through terminal exit, then verify result.json, history, and best/final checkpoints.
