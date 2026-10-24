# Step 0005: Pass formal training preflight

- Time: 2026-10-24T16:26:59+08:00
- Milestone: `05-formal-training`
- Status: **Passed**

## Objective and outcome

Confirmed the container, GPU, disk, clean target directory, absence of duplicate trainers, and focused tests are ready for the formal SegMAN-S baseline.

## Commands

- `docker ps -a --filter name=segman_v2`
- `docker exec segman_v2 pgrep -af tools/train.py`
- `docker exec segman_v2 nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader`
- `docker exec segman_v2 df -h /workspace/segman/work_dirs`
- `docker exec -w /workspace/segman segman_v2 /root/anaconda3/bin/conda run --no-capture-output -n segman python -m unittest discover -s tests -v`

## Changed files

- `SegMAN-main/project_docs/milestones/05-formal-training.md`

## Evidence

- `5/5 focused unit tests passed`
- `RTX 4060 Laptop GPU: 7869 MiB free before launch`
- `Workspace: 102 GB free before launch`

## Decisions and assumptions

- The target work directory is work_dirs/segman_s_voc12_random_seed42 and was absent at preflight.

## Next action

Launch the formal random-initialized SegMAN-S job and record it separately.
