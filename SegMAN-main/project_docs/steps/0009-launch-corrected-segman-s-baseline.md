# Step 0009: Launch corrected SegMAN-S baseline

- Time: 2026-10-24T16:36:47+08:00
- Milestone: `05-formal-training`
- Status: **In progress**

## Objective and outcome

Started the corrected random-initialized seed-42 SegMAN-S baseline in a dedicated retry directory; the run reached iteration 550 with finite decreasing loss and remains active.

## Commands

- `./SegMAN-main/scripts/train_voc_s.ps1 -WorkDir /workspace/segman/work_dirs/segman_s_voc12_random_seed42_r1 -Seed 42`

## Changed files

- `SegMAN-main/project_docs/STATUS.md`
- `SegMAN-main/project_docs/milestones/05-formal-training.md`

## Evidence

- `SegMAN-main/work_dirs/segman_s_voc12_random_seed42_r1/segman_s_voc12.py`
- `SegMAN-main/work_dirs/segman_s_voc12_random_seed42_r1/20261024_083317.log.json`
- `SegMAN-main/work_dirs/segman_s_voc12_random_seed42_r1/tf_logs`

## Decisions and assumptions

- No pretrained weight was supplied; this is the random-initialized formal baseline.
- Host launcher PID 46392; iteration 550 loss 1.75146, decode accuracy 60.4916, memory 3013 MiB.

## Next action

Monitor the first validation at iteration 732 and record terminal completion separately.
