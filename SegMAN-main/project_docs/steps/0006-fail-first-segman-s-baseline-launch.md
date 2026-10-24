# Step 0006: Fail first SegMAN-S baseline launch

- Time: 2026-10-24T16:29:40+08:00
- Milestone: `05-formal-training`
- Status: **Failed**

## Objective and outcome

The first formal launch exited before iteration 1 because a DataLoader worker could not allocate a shared tensor in the container's 64 MiB /dev/shm.

## Commands

- `./SegMAN-main/scripts/train_voc_s.ps1 -WorkDir /workspace/segman/work_dirs/segman_s_voc12_random_seed42 -Seed 42`

## Changed files

- `SegMAN-main/project_docs/STATUS.md`
- `SegMAN-main/project_docs/milestones/05-formal-training.md`

## Evidence

- `SegMAN-main/work_dirs/segman_s_voc12_random_seed42/launcher.stderr.log`
- `SegMAN-main/work_dirs/segman_s_voc12_random_seed42/20261024_082752.log.json`

## Decisions and assumptions

- The failed attempt has zero train records and is not reportable training evidence.

## Next action

Set data.workers_per_gpu to 1, test the configuration, and retry in a fresh work directory.
