# Step 0186: Reject implicit VOC fields in TTA config

- Time: 2026-10-29T08:10:28+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

The first complete 53-test suite rejected the reference-style TTA YAML because it lacked the explicit top-level VOC/model-selection fields required of every config; no TTA evaluation ran.

## Commands

- `python -m unittest discover -s tests -v`

## Changed files

- `None during the failed run`

## Evidence

- `53 tests ran with one KeyError model in test_all_configs_preserve_voc_contract`

## Decisions and assumptions

- Add explicit 21-class, ignore-255, no-reduction, independent-val, and mIoU-selection fields without weakening the common contract test.

## Next action

Correct the TTA YAML/tool contract and rerun all tests.
