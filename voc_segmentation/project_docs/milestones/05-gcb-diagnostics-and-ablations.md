# Milestone 05: GCB-Seg diagnostics and ablations

- State: **In progress — interfaces and losses verified**
- Last updated: 2026-10-26 08:38 +08:00

Implement GCM, GFF, BAB, auxiliary head, and loss composition. Pass focused
interface/loss tests, smoke, and overfit, then isolate each component in
controlled validation experiments.

## Verified interface evidence

- `models/modules/gcb.py` implements the planned GCM with 1×1, three
  depthwise-separable dilation 1/3/6, and global-pooling branches. Each branch
  emits 64 channels; concatenation is reduced to 256 channels. Convolutional
  blocks use Conv, GroupNorm, GELU, and Dropout2d.
- GFF projects C1/C2/C3 to 256 channels, performs gated top-down fusion from
  F4 through F1, and applies two depthwise-separable refinements at each stage.
  Disabling its gate falls back to direct additive fusion for controlled
  ablation without changing shapes.
- BAB concatenates C1 with upsampled C2, applies two 3×3 blocks, and returns a
  64-channel boundary feature plus one boundary logit. The boundary feature is
  concatenated with F1 before the main segmentation head.
- Training mode returns full-resolution main `[B,21,H,W]`, auxiliary
  `[B,21,H,W]`, and edge `[B,1,H,W]` logits. Eval mode returns only the main
  tensor. Independent switches disable global pooling, gates, BAB, or the
  auxiliary head while preserving main output shape.
- The full MiT-B1 model has 15,841,707 trainable parameters: 13,151,424 in the
  encoder and 2,690,283 in GCB-Seg modules. The accepted MiT-B1 checkpoint
  loaded strictly. New modules contain 28 GroupNorm and zero BatchNorm layers.

## Verified loss evidence

- `losses/gcb.py` implements label-smoothed main CE, present-class-only
  Lovász-Softmax, boundary BCE plus Dice, and auxiliary CE with planned weights
  1.0/0.5/0.2/0.4.
- Boundary targets compare valid horizontal and vertical neighbors, mark both
  sides of a class transition, dilate once with a 3×3 max-pool, mask ignore 255,
  and never create an edge merely from the image boundary.
- Tests prove finite composite components, exact weighting, backward-safe
  gradients for every output, ignore-255 invariance in Lovász, required-head
  enforcement, and absence of outer edges for constant masks.
- The first focused run passed model and configuration checks but three loss
  tests failed at the same Lovász gradient line: an in-place subtraction used
  overlapping slices, which PyTorch 2.1 forbids. The mathematically equivalent
  non-in-place concatenated difference fixed the causal defect.
- Corrected focused tests passed 13/13 in 2.986 seconds. The complete target
  container suite passed 40/40 in 5.282 seconds.

## Next action

Implement and pass a pretrained 384×384, batch-2 CUDA optimizer-step smoke
with all four loss components, finite gradients, stable AMP, exact parameter
groups, and saved diagnostic-only evidence. Then implement the deterministic
two-sample overfit gate before starting ablations.
