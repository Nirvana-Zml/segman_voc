# Step 0007: Fail package-style focused test invocation

- Time: 2026-10-24T16:30:53+08:00
- Milestone: `05-formal-training`
- Status: **Failed**

## Objective and outcome

The worker-count configuration test did not run because tests is not a Python package and the package-style unittest module name could not be imported.

## Commands

- `docker exec -w /workspace/segman segman_v2 /root/anaconda3/bin/conda run --no-capture-output -n segman python -m unittest tests.test_voc_configs -v`

## Changed files

- `SegMAN-main/project_docs/milestones/05-formal-training.md`

## Evidence

- `ModuleNotFoundError: No module named tests.test_voc_configs`

## Decisions and assumptions

- This failure is an invocation error and provides no evidence about the modified configuration.

## Next action

Rerun with unittest discover and the test_voc_configs.py filename pattern.
