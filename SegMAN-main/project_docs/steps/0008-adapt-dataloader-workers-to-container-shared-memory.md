# Step 0008: Adapt DataLoader workers to container shared memory

- Time: 2026-10-24T16:31:38+08:00
- Milestone: `05-formal-training`
- Status: **Passed**

## Objective and outcome

Reduced formal T/S train and validation DataLoader workers from 4/2 to 1/1 for the container's 64 MiB shared memory and passed the focused configuration contract test.

## Commands

- `docker exec -w /workspace/segman segman_v2 /root/anaconda3/bin/conda run --no-capture-output -n segman python -m unittest discover -s tests -p test_voc_configs.py -v`

## Changed files

- `SegMAN-main/segmentation/local_configs/segman/voc/segman_s_voc12.py`
- `SegMAN-main/segmentation/local_configs/segman/voc/segman_t_voc12.py`
- `SegMAN-main/tests/test_voc_configs.py`
- `SegMAN-main/README_PROJECT.md`
- `SegMAN-main/project_docs/milestones/05-formal-training.md`

## Evidence

- `Focused configuration test: 1/1 passed`

## Decisions and assumptions

- Keep physical batch size 2 and change only DataLoader concurrency.

## Next action

Retry the formal seed-42 SegMAN-S baseline in work_dirs/segman_s_voc12_random_seed42_r1.
