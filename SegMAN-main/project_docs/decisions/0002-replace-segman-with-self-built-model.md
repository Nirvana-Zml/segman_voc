# Decision 0002: Replace SegMAN with a self-built model

- Time: 2026-10-24 17:17 +08:00
- Status: Active

## Decision

The SegMAN architecture is no longer the active model candidate. The next
formal candidate will be a self-built semantic-segmentation network.

## Scope

- Preserve the `SegMAN-main/` tree and all stopped-run artifacts; they remain
  useful as environment, tooling, and diagnostic evidence.
- Reuse the audited VOC2012 splits and compatible training, validation,
  TensorBoard, test400 inference, and mask-verification infrastructure.
- Do not reuse a SegMAN checkpoint for model selection or final inference.
- Continue with random initialization while external pretraining permission is
  unknown.

## First implementation gate

Before coding, freeze a minimal model contract containing:

- encoder/decoder blocks and the tensor shape at every stage;
- input `B x 3 x H x W` and output logits `B x 21 x H x W`;
- label IDs `0..20`, ignore label `255`, and nearest-neighbor mask resizing;
- normalization and initialization suitable for the actual small batch;
- parameter/VRAM target, CE loss, and output-resize behavior;
- focused shape/loss checks, one-iteration smoke, and two-sample overfit gates.

No architecture-performance claim is made until validation evidence exists.
