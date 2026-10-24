# Milestone 00: Standalone project reset

- State: **Passed**
- Last updated: 2026-10-24 17:56 +08:00

## Objective

Replace the deleted SegMAN application with a standalone PyTorch project,
current driver, and durable evidence ledger without restoring legacy source.

## Current evidence

- Root `Dockerfile` no longer references SegMAN/MMCV/MMSeg.
- `voc_segmentation/` contains the source contract, configs, tests, and README.
- `.agents/skills/voc-segmentation-project-driver/` contains the new workflow.
- This ledger continues numbering after legacy step 0017.

## Pass gate

Passed. The new skill validator succeeded, the status helper ran, four focused
contract tests passed, and the Python sources passed a no-bytecode syntax
compile. The first record-helper call exposed a missing physical `steps/`
directory; the directory is now versioned and the failure is retained. Active
step records continue at 0018.
