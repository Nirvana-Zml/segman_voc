# Step 0058: Launch reportable DeepLabV3+ VOC-train run

- Time: 2026-10-25T07:52:22+08:00
- Milestone: `03-cnn-baseline`
- Status: **In progress**

## Objective and outcome

After Docker Desktop restarted, the container/GPU/disk/process preflight passed and the 20,000-iteration reportable DeepLabV3+ run launched from clean commit b7ed0f5. PID 28 remained active with 35 finite training records; no validation or terminal result is accepted yet.

## Commands

- `docker exec voc2012-gcb sh -lc nvidia-smi, df, and pgrep preflight`
- `docker exec -d voc2012-gcb sh runs/deeplabv3plus_r50_voc_train/launch.sh`
- `docker exec voc2012-gcb sh -lc pgrep and nvidia-smi observation`

## Changed files

- `voc_segmentation/runs/deeplabv3plus_r50_voc_train/launch.sh (gitignored runtime launcher)`
- `voc_segmentation/project_docs/STATUS.md`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `Run start: 2026-10-25 07:50:42 +08:00`
- `Clean source commit: b7ed0f50860ee4c28911b4f841e8a1f7dc90c882`
- `Active container PID 28; 35 JSONL training records; exit_code.txt absent`
- `Step 35: loss 2.829548, gradient norm 8.267307, encoder/decoder LR 0.00007000/0.00070000`
- `GPU observation: about 1,671 MiB used; workspace free space 96 GB`

## Decisions and assumptions

- Treat this as launch evidence only; do not pass the CNN gate before complete validation and terminal artifact verification.

## Next action

Monitor iteration 1,000 full official validation, then monitor the run to terminal exit.
