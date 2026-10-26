# Step 0138: Verify completed A2 gated-GFF ablation run

- Time: 2026-10-27T07:56:21+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Independently verified A2 terminal exit, all 20000 train steps, 20 complete validations, six recovered AMP overflows, every global-confusion metric, validation-selected checkpoints, all recursively discovered checkpoint tensors, TensorBoard series, provenance, and artifact hashes.

## Commands

- `docker exec process and GPU checks; corrected full-retention A2 terminal audit; compare A0, A1, and A2 result JSON`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Exit 0; best step 18000 mIoU 0.755643 versus A1 0.736626 and A0 0.733889; terminal mIoU 0.752653; 20026 history records; six AMP recoveries; best/last SHA 82ef8b83... / fb72e73a...; result SHA 7abb59bb...; elapsed 34986.868 seconds; peak CUDA 1218130944 bytes.`

## Decisions and assumptions

- Accept A2 as the completed gated-GFF result for seed 42 and the common 20k budget; select best.pt at step 18000, not last.pt. Do not infer statistical significance from one seed.

## Next action

Commit the terminal ledger, then pass A3 BAB and edge-loss formal preflight from the new clean revision.
