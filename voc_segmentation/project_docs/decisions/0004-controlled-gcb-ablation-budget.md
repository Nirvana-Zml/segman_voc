# Decision 0004: Controlled GCB ablation budget

- Date: 2026-10-26
- Status: Accepted
- Authority: `VOC2012_Semantic_Segmentation_Project_Plan.md` sections 7.4 and 9.2

## Decision

Use the completed SegFormer-B1 VOC-train run as A0 and run A1-A4 with the same
20,000 optimizer updates, official train/val split, seed 42, 384x384 crop,
scale range 0.5-2.0, horizontal-flip probability 0.5, physical/effective batch
2/8, MiT-B1 checkpoint, encoder/decoder learning rates 0.00006/0.00060,
AdamW weight decay 0.01, warmup 1,500, polynomial power 0.9, validation every
1,000 steps, and the shared native-resolution global-confusion evaluator.

The cumulative variants are:

| ID | Increment over prior variant |
| --- | --- |
| A0 | Original SegFormer-B1 with CE; existing verified formal baseline |
| A1 | Add GCM; retain SegFormer-style linear multi-scale fusion |
| A2 | Replace linear fusion with gated top-down GFF |
| A3 | Add BAB and boundary loss weight 0.2 |
| A4 | Add Lovasz-Softmax weight 0.5 |

Auxiliary loss and label smoothing are both zero for A0-A4 so neither becomes
an unlisted changing factor. The planned full trainaug configuration may use
them only after separate validation evidence; its 40,000-step budget belongs
to formal development, not this VOC-train structural comparison.

## Rationale

The existing SegFormer run already used the selected 20,000-step resource-safe
budget and the same data, seed, augmentation, optimizer, scheduler, and
evaluator contracts. Reusing it avoids an identical long rerun. Changing GCB
ablations to the full trainaug configuration's 40,000 steps would confound
architecture changes with training duration and invalidate A0 comparison.

## Consequences

- Configuration tests must compare every common A1-A4 field against A0.
- Each experiment changes only the cumulative component listed above.
- All model selection uses saved full official-validation mIoU.
- Diagnostic runs and checkpoints remain ineligible for the ablation table.
