# Step 0076: Fail SegFormer probe on unavailable timm MiT-B1

- Time: 2026-10-26T00:01:53+08:00
- Milestone: `04-transformer-baseline`
- Status: **Failed**

## Objective and outcome

The first SegFormer dependency probe failed before model creation because installed timm 0.9.16 exposes no mit_b1 model. No source or runtime artifact changed.

## Commands

- `docker exec voc2012-gcb Python timm MiT-B1 availability probe`

## Changed files

None.

## Evidence

- `timm version 0.9.16; matching model list empty; RuntimeError Unknown model mit_b1`

## Decisions and assumptions

- Do not add an unpinned replacement dependency. Implement the resource-safe MiT-B1 encoder natively in PyTorch with checkpoint-compatible naming.

## Next action

Implement and test native MiT-B1 plus the SegFormer MLP decoder.
