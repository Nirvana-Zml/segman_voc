# Step 0010: Confirm first validation is active

- Time: 2026-10-24T16:41:18+08:00
- Milestone: `05-formal-training`
- Status: **In progress**

## Objective and outcome

Confirmed the corrected formal run remains active after iteration 700 and has entered the first full-validation window; no validation summary is available yet.

## Commands

- `Inspect JSON log, launcher process, container trainer processes, and nvidia-smi`

## Changed files

- `SegMAN-main/project_docs/STATUS.md`
- `SegMAN-main/project_docs/milestones/05-formal-training.md`

## Evidence

- `SegMAN-main/work_dirs/segman_s_voc12_random_seed42_r1/20261024_083317.log.json`
- `Latest train record: iter 700, loss 1.58806; host launcher PID 46392 alive`

## Decisions and assumptions

- Do not claim validation metrics until the evaluator writes the complete summary.

## Next action

Wait for the iteration-732 validation summary, then continue monitoring the background run.
