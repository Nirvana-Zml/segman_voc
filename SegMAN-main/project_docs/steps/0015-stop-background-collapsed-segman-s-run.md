# Step 0015: Stop background-collapsed SegMAN-S run

- Time: 2026-10-24T17:13:02+08:00
- Milestone: `05-formal-training`
- Status: **Passed**

## Objective and outcome

Stopped the active SegMAN-S trainer at the user's request after five validations remained background-only; preserved all diagnostic artifacts.

## Commands

- `docker exec segman_v2 kill -TERM 509`

## Changed files

- `SegMAN-main/project_docs/STATUS.md`
- `SegMAN-main/project_docs/milestones/05-formal-training.md`

## Evidence

- `SegMAN-main/work_dirs/segman_s_voc12_random_seed42_r1/20261024_083317.log.json`
- `SegMAN-main/work_dirs/segman_s_voc12_random_seed42_r1/iter_3660.pth`
- `Final periodic train record: iter 4350, loss 1.46453; 5 validations at mIoU 0.0349`

## Decisions and assumptions

- The host launcher and all container trainer/DataLoader processes exited; the container remains running.
- This interrupted background-only run is not eligible for model selection or test400 inference.

## Next action

Implement and validate a simpler self-built semantic-segmentation baseline.
