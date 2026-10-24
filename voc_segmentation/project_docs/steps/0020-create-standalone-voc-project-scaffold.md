# Step 0020: Create standalone VOC project scaffold

- Time: 2026-10-24T17:57:32+08:00
- Milestone: `00-project-reset`
- Status: **Passed**

## Objective and outcome

Replaced the deleted SegMAN application contract with a standalone PyTorch root, pinned Docker environment, hardware-safe experiment configs, focused tests, a new project driver, and an active evidence ledger.

## Commands

- `python C:/Users/Nirvanaml/.codex/skills/.system/skill-creator/scripts/quick_validate.py .agents/skills/voc-segmentation-project-driver`
- `python -m unittest discover -s voc_segmentation/tests -v`
- `python -c <in-memory compile of six Python sources>`
- `git diff --check`

## Changed files

- `Dockerfile`
- `.dockerignore`
- `.gitignore`
- `AGENTS.md`
- `voc_segmentation/`
- `.agents/skills/voc-segmentation-project-driver/`
- `.agents/skills/segman-project-driver/SKILL.md`

## Evidence

- `New skill validation: passed`
- `Focused tests: 4 passed`
- `In-memory syntax compile: 6 files passed`
- `git diff --check: exit code 0`

## Decisions and assumptions

- Use voc_segmentation/ as the active root and never restore SegMAN-main implicitly.
- Use MiT-B1/384 as the first 8 GB-safe profile; gate MiT-B2/512 on a saved memory profile.

## Next action

Build voc2012-gcb:dev and save a successful CUDA environment report.
