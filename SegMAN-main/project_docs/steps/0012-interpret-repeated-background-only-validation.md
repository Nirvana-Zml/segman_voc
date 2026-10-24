# Step 0012: Interpret repeated background-only validation

- Time: 2026-10-24T16:51:26+08:00
- Milestone: `05-formal-training`
- Status: **In progress**

## Objective and outcome

Confirmed the first two validations predict background for every pixel; the active random-initialized run continued to iteration 1800, so epoch 5 is the next diagnostic gate.

## Commands

- `Parse all train and val records from work_dirs/segman_s_voc12_random_seed42_r1/20261024_083317.log.json`

## Changed files

- `SegMAN-main/project_docs/STATUS.md`
- `SegMAN-main/project_docs/milestones/05-formal-training.md`

## Evidence

- `Epochs 1 and 2: mIoU 0.0349, mAcc 0.0476, aAcc 0.7332; foreground IoU and accuracy all zero`
- `Latest observed train iteration 1800; recent loss range 1.54246 to 1.81056`

## Decisions and assumptions

- Do not stop solely on two early random-initialized validations; inspect epoch 5 / iteration 3660 as the next gate.

## Next action

At epoch 5, require non-zero foreground IoU or perform focused diagnosis before continuing the full run.
