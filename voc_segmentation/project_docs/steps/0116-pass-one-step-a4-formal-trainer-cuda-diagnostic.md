# Step 0116: Pass one-step A4 formal-trainer CUDA diagnostic

- Time: 2026-10-26T09:19:01+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

The shared formal trainer completed one effective-batch-8 A4 AdamW+AMP update and a deliberately two-image, non-reportable validation, saving diagnostic history, TensorBoard, resolved config, and checkpoint evidence.

## Commands

- `docker exec voc2012-gcb python tools/train_gcb_seg.py --config configs/gcb_ablation_a4_lovasz.yaml --output-dir runs/diagnostics/gcb_a4_one_step --max-iterations 1 --validate-every 1 --validation-limit 2 --diagnostic-only`

## Changed files

- `Gitignored runs/diagnostics/gcb_a4_one_step evidence; STATUS and milestone 05`

## Evidence

- `Total/CE/Lovasz/edge 3.558069/2.924114/0.932956/0.837382; gradient norm 9.638385; zero AMP overflow; peak CUDA 1165862400 bytes; runtime 13.461 seconds; step-1 diagnostic checkpoint SHA-256 618f90ff79730461cda5c84927bdda3896606e9f6106c4d287c3ba45413ba80e.`

## Decisions and assumptions

- The two-image mIoU 0.208359 proves evaluator integration only and is excluded from the ablation table.

## Next action

Independently audit the formula, history, resolved config, checkpoint tensors, TensorBoard series, provenance, and hashes.
