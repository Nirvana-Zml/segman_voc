# Milestone 04: SegFormer baseline

- State: **In progress — native MiT-B1/SegFormer interfaces verified**
- Last updated: 2026-10-26 00:01 +08:00

Train and evaluate an 8 GB-safe SegFormer baseline with the same split,
evaluator, seed policy, and artifact contract used by the CNN baseline.

## Verified interface evidence

- The installed `timm==0.9.16` does not provide `mit_b1`; the first dependency
  probe failed with `RuntimeError: Unknown model (mit_b1)` before model
  creation and changed no artifact.
- `models/encoders/mit.py` provides a native MiT-B1 with overlapping patch
  embeddings, spatial-reduction attention ratios 8/4/2/1, Mix-FFN depthwise
  convolution, stochastic depth, and depths 2/2/2/2.
- Encoder features use channels 64/128/320/512 at reductions 4/8/16/32.
  Official-style stage names are preserved to support explicit, provenance-
  checked local checkpoint conversion and strict loading.
- `models/baselines/segformer.py` implements the original-style four-scale MLP
  decoder with 256 decoder channels, fused dropout, 21-class logits, and
  bilinear restoration to the source spatial dimensions.
- The complete model has 13,682,645 parameters: 13,151,424 in the encoder and
  531,221 in the decoder.
- `configs/segformer_b1.yaml` fixes the 8 GB-safe MiT-B1/384 profile, physical
  batch 2, effective batch 8, seed 42, independent official validation, and an
  explicit future ImageNet checkpoint path. The weight file has not yet been
  acquired or accepted.
- Focused SegFormer and configuration tests passed 8/8 in 1.254 seconds. The
  full container suite passed 28/28 in 4.766 seconds. It verifies four-scale
  shapes, finite `[1,21,H,W]` logits, exact spatial restoration, strict local
  encoder checkpoint round-trip, and the resource-safe configuration gate.
- The first step-recording command failed at PowerShell parse time because its
  command-description argument contained nested shell quoting. It changed no
  file; the failure, dependency probe, and implementation result were then
  recorded as separate immutable steps using safe quoting.

## Next action

Acquire and provenance-verify a compatible MiT-B1 ImageNet checkpoint, then
run a 384×384 CUDA optimizer-step smoke before implementing the two-sample
overfit and formal trainer.
