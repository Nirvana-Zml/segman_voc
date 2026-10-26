# Step 0127: Verify A1 step-1000 full validation

- Time: 2026-10-26T10:14:13+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Independently accepted A1's first reportable validation after reproducing every metric from the global confusion matrix and verifying sequential finite history, the full 1449-image split, non-diagnostic selected checkpoint, finite model tensors, TensorBoard tags, hashes, and continued training.

## Commands

- `Read validation/history artifacts; docker exec Base64-decoded metric/checkpoint/TensorBoard audit; docker exec ps and nvidia-smi`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Step 1000 mIoU 0.604609, foreground mIoU 0.589999, pixel accuracy 0.902396, mean pixel accuracy 0.724140, CE 0.329029; 1449 samples; checkpoint SHA b2165bb8ba17bd20af5b1df2a90b7e4ad7aa72ba2c404f4ee64999b8ec38a0db; validation SHA 1e78e80303725915bab05d34f364d2f73584ba521f67e68a3e9689c542b5f49f; zero overflows; PIDs remained active.`

## Decisions and assumptions

- Accept 0.604609 only as A1 step-1000 official-validation evidence, not the final A1 result; best selection remains provisional until all 20000 steps complete.

## Next action

Monitor and independently audit step 2000, compare against step 1000, and verify best-checkpoint selection.
