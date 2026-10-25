# Step 0093: Pass SegFormer trainer checkpoint resume

- Time: 2026-10-26T00:27:59+08:00
- Milestone: `04-transformer-baseline`
- Status: **Passed**

## Objective and outcome

Resumed the forced-overflow checkpoint from step 1, reconciled existing history without duplicates, restored AdamW and AMP state, and completed through step 3.

## Commands

- `docker exec voc2012-gcb python tools/train_segformer.py --diagnostic-only --output-dir artifacts/transformer_baseline/pretrained_formal_overflow_v1 --resume artifacts/transformer_baseline/pretrained_formal_overflow_v1/best.pt --max-iterations 3 --validate-every 2 --validation-limit 1`

## Changed files

- `updated gitignored diagnostic history, checkpoints, TensorBoard, validation and result artifacts`

## Evidence

- `History reconciliation kept 3 and dropped 0; start step 1; unique train and validation steps 1,2,3; one inherited AMP overflow; final checkpoint SHA-256 b1d503bbb509ced4ff6985cfcb85d6458e2b1dc88b73e7585143e0b52430033a.`

## Decisions and assumptions

- Accept resume mechanics as diagnostic evidence; no limited-validation metric is reportable.

## Next action

Independently validate histories, checkpoint metadata, and artifact hashes.
