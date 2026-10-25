# Step 0071: Resume formal CNN run from step 18000

- Time: 2026-10-25T21:20:57+08:00
- Milestone: `03-cnn-baseline`
- Status: **In progress**

## Objective and outcome

Passed recovery preflight, preserved the pause exit code, and resumed the formal run from verified step 18000 under clean commit 05048a5. PID 68 is active and history continues uniquely from step 18001.

## Commands

- `docker exec -d voc2012-gcb sh /workspace/voc_segmentation/runs/deeplabv3plus_r50_voc_train/resume_from_018000.sh; docker exec voc2012-gcb <process, GPU, log, and JSONL checks>`

## Changed files

- `voc_segmentation/runs/deeplabv3plus_r50_voc_train/resume_from_018000.sh (gitignored); exit_code.user_pause_step_018000.txt (gitignored); voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `Started 2026-10-25 21:18:13 +08:00; PID 68; commit 05048a5f57636cd0f17553779d5a7fb4ee33e1c3; history kept=18023 dropped=0; at step 18033: 18033 unique train records, 18 validations, 5 AMP overflows, 0 duplicates; exit_code.txt absent; GPU 1929 MiB at 86%`

## Decisions and assumptions

- Continue the same 20000-step formal run and treat this launch as in-progress evidence until terminal artifacts pass independent verification.

## Next action

Monitor PID 68 through terminal exit, then verify result.json, history, and best/final checkpoints.
