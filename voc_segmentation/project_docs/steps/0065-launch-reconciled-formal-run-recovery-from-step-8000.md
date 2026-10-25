# Step 0065: Launch reconciled formal run recovery from step 8000

- Time: 2026-10-25T13:11:23+08:00
- Milestone: `03-cnn-baseline`
- Status: **In progress**

## Objective and outcome

Launched the second formal recovery from the unchanged step-8000 checkpoint under clean commit d789bc0. The trainer archived all 8,087 prior history records, retained 8,008 records through the checkpoint, dropped 79 uncheckpointed/duplicate records, and resumed with unique active step numbers. PID 422 remained active; no terminal result is accepted yet.

## Commands

- `docker exec -d voc2012-gcb sh runs/deeplabv3plus_r50_voc_train/resume_from_008000_reconciled.sh`
- `PowerShell JSONL uniqueness/archive verification plus docker pgrep and nvidia-smi`

## Changed files

- `voc_segmentation/runs/deeplabv3plus_r50_voc_train/resume_from_008000_reconciled.sh (gitignored runtime launcher)`
- `voc_segmentation/project_docs/STATUS.md`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `Recovery start: 2026-10-25 13:09:39 +08:00; clean commit d789bc06c6db6883e4010ceba587d3212484ad5c`
- `History reconciliation: original 8087, kept 8008, dropped 79`
- `Archive SHA-256 fcb176067aafa16520f8d85f6178b92e1ed811adb4dcec5830a38c157f1b2618`
- `At observation: max train step 8018, 8018 train records, 8 validations, 0 duplicate train steps`
- `PID 422 active; exit_code.txt absent; GPU about 2030 MiB and 99% utilization`

## Decisions and assumptions

- Continue from the validated step-8000 model/optimizer/scaler state while preserving both failed segments in the immutable runtime archive and console log.

## Next action

Monitor the complete step-9000 official validation, then continue to terminal exit and verify the selected best checkpoint.
