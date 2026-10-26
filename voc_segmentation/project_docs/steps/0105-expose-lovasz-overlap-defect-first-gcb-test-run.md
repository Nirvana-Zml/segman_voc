# Step 0105: Expose Lovasz overlap defect in first GCB test run

- Time: 2026-10-26T08:39:48+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

Implemented the planned GCB-Seg modules, output contract, boundary targets, and composite loss, but the first focused run failed three loss tests because Lovasz gradient used an in-place subtraction on overlapping tensor slices.

## Commands

- `docker exec voc2012-gcb python -m py_compile models/modules/gcb.py models/gcb_seg.py losses/gcb.py; docker exec voc2012-gcb python -m unittest tests.test_gcb_seg tests.test_gcb_losses tests.test_config_contract -v`

## Changed files

- `GCB-Seg model/modules/losses/config/tests and documentation; project step ledger`

## Evidence

- `13 focused tests: 10 passed and 3 errored at _lovasz_gradient with PyTorch RuntimeError for overlapping input/output memory.`

## Decisions and assumptions

- Keep the canonical Lovasz gradient values but compute adjacent differences without an in-place overlapping write.

## Next action

Patch the Lovasz gradient and rerun focused plus full regression suites.
