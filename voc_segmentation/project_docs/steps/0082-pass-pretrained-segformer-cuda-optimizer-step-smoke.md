# Step 0082: Pass pretrained SegFormer CUDA optimizer-step smoke

- Time: 2026-10-26T00:12:23+08:00
- Milestone: `04-transformer-baseline`
- Status: **Passed**

## Objective and outcome

Completed one pretrained MiT-B1 SegFormer AdamW update at 384 by 384 and batch 2 on official VOC train with finite loss and gradients, stable AMP scale, and low peak memory.

## Commands

- `docker exec voc2012-gcb python tools/smoke_segformer.py`

## Changed files

- `gitignored artifacts/transformer_baseline/smoke.json; milestone and status docs`

## Evidence

- `ok=true diagnostic_only=true; loss 2.93890380859375; gradient norm 17.925527572631836; encoder and decoder LR 0.00006 and 0.0006; AMP 1024 to 1024; peak CUDA 623762432 bytes; elapsed 1.747114148 seconds; result SHA-256 742b76f757d38b69385db2337d2a6ebf886d854a9abbe604c17cd38240df30b0`

## Decisions and assumptions

- Accept the 8 GB-safe pretrained optimizer pipeline as diagnostic evidence only. Do not use smoke weights or metrics for validation or submission.

## Next action

Implement and pass the deterministic two-sample SegFormer overfit gate.
