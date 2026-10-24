# Step 0011: Record first formal validation

- Time: 2026-10-24T16:44:35+08:00
- Milestone: `05-formal-training`
- Status: **In progress**

## Objective and outcome

Recorded the completed iteration-732 validation and confirmed the random-initialized SegMAN-S run continued through iteration 1000.

## Commands

- `Parse the latest *.log.json and inspect launcher PID 46392 and checkpoints`

## Changed files

- `SegMAN-main/project_docs/STATUS.md`
- `SegMAN-main/project_docs/milestones/05-formal-training.md`

## Evidence

- `SegMAN-main/work_dirs/segman_s_voc12_random_seed42_r1/20261024_083317.log.json`
- `SegMAN-main/work_dirs/segman_s_voc12_random_seed42_r1/best_mIoU_iter_732.pth`

## Decisions and assumptions

- Iteration 732: mIoU 0.0349, mAcc 0.0476, aAcc 0.7332; this early result is not final model-selection evidence.

## Next action

Keep monitoring the active run to iteration 40260 and record its terminal state separately.
