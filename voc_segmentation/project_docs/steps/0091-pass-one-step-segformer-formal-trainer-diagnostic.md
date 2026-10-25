# Step 0091: Pass one-step SegFormer formal-trainer diagnostic

- Time: 2026-10-26T00:27:51+08:00
- Milestone: `04-transformer-baseline`
- Status: **Passed**

## Objective and outcome

Completed one effective-batch-8 pretrained SegFormer update, limited native-resolution validation, checkpoint selection, JSONL and TensorBoard logging, and result emission on CUDA.

## Commands

- `docker exec voc2012-gcb python tools/train_segformer.py --diagnostic-only --output-dir artifacts/transformer_baseline/pretrained_formal_smoke_v1 --max-iterations 1 --validate-every 1 --validation-limit 2`

## Changed files

- `gitignored artifacts/transformer_baseline/pretrained_formal_smoke_v1; milestone and status documentation`

## Evidence

- `Train CE 3.000786; gradient norm 10.172207; peak CUDA 702257152 bytes; checkpoint SHA-256 9f30fb5be0af6ab2184baee36c26bbcda387e3868d71c03b547d53d907308183; result SHA-256 bffc944d1a59f25f278756050695654646bbb72cef8b7e47bdde31def61d5ffb.`

## Decisions and assumptions

- Accept pipeline operation only; two-image validation and the diagnostic checkpoint are non-reportable.

## Next action

Exercise bounded AMP overflow recovery.
