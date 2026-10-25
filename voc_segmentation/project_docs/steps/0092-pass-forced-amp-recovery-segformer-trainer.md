# Step 0092: Pass forced AMP recovery in SegFormer trainer

- Time: 2026-10-26T00:27:55+08:00
- Milestone: `04-transformer-baseline`
- Status: **Passed**

## Objective and outcome

A forced non-finite gradient triggered verified GradScaler backoff and retried the same logical iteration before completing validation and a diagnostic checkpoint.

## Commands

- `docker exec voc2012-gcb python tools/train_segformer.py --diagnostic-only --diagnostic-force-amp-overflow-once --output-dir artifacts/transformer_baseline/pretrained_formal_overflow_v1 --max-iterations 1 --validate-every 1 --validation-limit 1`

## Changed files

- `gitignored artifacts/transformer_baseline/pretrained_formal_overflow_v1; milestone and status documentation`

## Evidence

- `Attempt 1 AMP scale 1024 to 512 with action retry; attempt 2 completed step 1; result recorded exactly one overflow and diagnostic_only true.`

## Decisions and assumptions

- Treat only verified scale backoff as recoverable; retain configured consecutive and total overflow bounds.

## Next action

Resume the diagnostic checkpoint and verify unambiguous history continuation.
