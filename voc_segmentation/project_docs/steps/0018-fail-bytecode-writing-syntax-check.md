# Step 0018: Fail bytecode-writing syntax check

- Time: 2026-10-24T17:57:06+08:00
- Milestone: `01-environment-and-source`
- Status: **Failed**

## Objective and outcome

The initial py_compile verification could not create __pycache__ under the read-only repository agent directory; no source syntax failure was reported.

## Commands

- `python -m py_compile .agents/skills/voc-segmentation-project-driver/scripts/project_status.py .agents/skills/voc-segmentation-project-driver/scripts/record_step.py voc_segmentation/project_contract.py voc_segmentation/tools/check_environment.py voc_segmentation/tests/test_project_contract.py voc_segmentation/tests/test_config_contract.py`

## Changed files

None.

## Evidence

- `Exit code 1: WinError 5 access denied for .agents/skills/voc-segmentation-project-driver/scripts/__pycache__`

## Decisions and assumptions

- Treat this as a verification-command write-path failure, not as evidence that the source is syntactically invalid.

## Next action

Compile the same sources in memory without writing bytecode.
