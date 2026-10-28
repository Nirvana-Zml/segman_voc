# Step 0161: Pass selected A2 trainaug CUDA diagnostic

- Time: 2026-10-28T08:22:03+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Completed and independently audited one real-trainaug optimizer step with finite CE-only loss and gradients, verified provenance, TensorBoard, and a diagnostic-only finite checkpoint.

## Commands

- `docker exec voc2012-gcb python tools/train_gcb_seg.py --config configs/gcb_selected_a2_trainaug.yaml --output-dir runs/diagnostics/gcb_selected_a2_trainaug_one_step --max-iterations 1 --validate-every 1 --validation-limit 2 --diagnostic-only; audit artifacts`

## Changed files

- `Ignored diagnostic result/history/checkpoint/TensorBoard; STATUS.md; milestone 06; immutable step ledger`

## Evidence

- `loss 3.193626; grad 8.631807; peak CUDA 1100638720 bytes; checkpoint SHA c6ede983... with 1036 finite tensors; result SHA 0f75f5f2...`

## Decisions and assumptions

- Accept memory, data-path, optimizer, loss, validation, and provenance paths for formal preflight; never use this diagnostic checkpoint for submission.

## Next action

Pass clean-source formal-run resource, output-absence, hash, and contract checks.
