# Step 0162: Pass selected A2 trainaug formal-run preflight

- Time: 2026-10-28T08:22:03+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Verified clean source, idle GPU, no trainer, 757 GB free, absent formal output, exact A2 selection and all config/data/weight/val hashes for the 40000-step trainaug run.

## Commands

- `docker exec voc2012-gcb python .test_tmp/a2_trainaug_formal_preflight.py; ps; nvidia-smi; df; host git clean check`

## Changed files

- `STATUS.md; milestone 06; immutable step ledger`

## Evidence

- `source 2420f789...; GPU 113/8188 MiB at 0 percent; output absent; config 5fcbfa7c...; trainaug 10582 overlap 0; manifest 7fa9f8f0...; val cdc1326d...`

## Decisions and assumptions

- The reportable job may launch only after this preflight ledger is committed, using that new clean full commit.

## Next action

Commit preflight evidence, then launch and early-audit the 40000-step selected A2 trainaug run.
