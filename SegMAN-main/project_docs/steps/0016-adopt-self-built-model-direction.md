# Step 0016: Adopt self-built model direction

- Time: 2026-10-24T17:18:24+08:00
- Milestone: `01-source-and-tests`
- Status: **Information**

## Objective and outcome

Recorded the user's decision to retire SegMAN as the active architecture and reopen source work for a self-built semantic-segmentation baseline.

## Commands

- `python .agents/skills/segman-project-driver/scripts/project_status.py`

## Changed files

- `SegMAN-main/project_docs/STATUS.md`
- `SegMAN-main/project_docs/milestones/01-source-and-tests.md`
- `SegMAN-main/project_docs/milestones/05-formal-training.md`
- `SegMAN-main/project_docs/decisions/0002-replace-segman-with-self-built-model.md`

## Evidence

- `SegMAN-main/project_docs/steps/0015-stop-background-collapsed-segman-s-run.md`
- `SegMAN-main/work_dirs/segman_s_voc12_random_seed42_r1/20261024_083317.log.json`

## Decisions and assumptions

- Retire SegMAN as the active model candidate while preserving its code, artifacts, and compatible project tooling.
- Freeze the replacement model contract before writing or training model code.

## Next action

Draft the minimal replacement-model contract and choose its encoder/decoder blocks, tensor shapes, normalization, initialization, and resource target.
