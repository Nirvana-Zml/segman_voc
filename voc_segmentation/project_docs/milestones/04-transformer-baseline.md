# Milestone 04: SegFormer baseline

- State: **In progress — pretrained MiT-B1, CUDA smoke, and overfit verified**
- Last updated: 2026-10-26 00:17 +08:00

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

## Pretrained weight evidence

- Source: NVIDIA Hugging Face model `nvidia/mit-b1`, fixed revision
  `13ddceec4e8bdf401e7cd7acf5aebc526222518c`, ImageNet-1K classification
  encoder. The official NVLabs repository identifies the pretrained weights
  and links the NVIDIA Source Code License for SegFormer.
- Source archive: 54,732,285 bytes, 194 tensors, SHA-256
  `980b86b60db37b1b1528086f6c53d253d879e9e09ebe07397b40fd275738a3bf`,
  exactly matching the repository's LFS metadata.
- Conversion removes only `classifier.weight` and `classifier.bias`, maps all
  encoder fields, and concatenates per-block key/value projections into native
  `kv` tensors. All 176 target keys and shapes match exactly.
- Converted checkpoint: 52,658,798 bytes, SHA-256
  `8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b`.
  Strict native load, finite floating tensors, and finite four-scale forward
  all passed. A second tool run reused the artifacts and reproduced the checks.
- License scope: NVIDIA Source Code License for SegFormer, non-commercial
  research or evaluation only; ImageNet terms also apply. Usage here is
  educational evaluation only, with no broader rights asserted.
- Machine-readable evidence:
  `artifacts/weights/mit_b1_imagenet1k/provenance.json`, SHA-256
  `2bf52c96aeb967605d4fe9975497c861cf3ee824bf62e6c892c0de3d688cf6ca`.
- The immediate first relative-path read after download reported the final file
  absent, but a directory listing showed the atomic rename had completed and
  absolute-path verification matched size and SHA. A separate inspection had
  also queried nonexistent `voc_segmentation/.gitignore` instead of the root
  ignore file. Both were read-only failures with no artifact mutation.

## CUDA smoke evidence

- Command: `docker exec voc2012-gcb python tools/smoke_segformer.py`
- Result: `artifacts/transformer_baseline/smoke.json`, `ok=true`,
  `diagnostic_only=true`, `optimizer_step_completed=true`, SHA-256
  `742b76f757d38b69385db2337d2a6ebf886d854a9abbe604c17cd38240df30b0`.
- Official VOC train count 1,464; deterministic samples `2008_005679` and
  `2008_002175`; input/logits `[2,3,384,384]` / `[2,21,384,384]`.
- CE loss 2.938904; finite pre-clip gradient norm 17.925528; valid pixels
  234,318; encoder/decoder learning rates 0.00006/0.0006.
- AMP scale remained 1,024; peak allocated CUDA memory 623,762,432 bytes;
  elapsed 1.747 seconds on the RTX 4060 Laptop GPU.
- The smoke strictly used the converted checkpoint hash above. It is pipeline
  evidence only and its weights/metrics are ineligible for validation or
  submission.
- Conversion/SegFormer/config focused tests passed 9/9 in 1.096 seconds. The
  complete target-container suite passed 29/29 in 5.894 seconds.

## Two-sample overfit evidence

- Command: `docker exec voc2012-gcb python tools/overfit_segformer.py`
- Result: `artifacts/transformer_baseline/overfit_two_samples/result.json`,
  SHA-256
  `9a6e4eeae1be267ba27bc1c6fc5ada4a31e4df345db2ac1eae6950e7bfea0210`,
  with `ok=true` and `diagnostic_only=true`.
- The deterministic batch contains official VOC-train samples `2008_005679`
  and `2008_002175`, fixed to `[2,3,128,128]`, with seed 42 and the accepted
  pretrained encoder hash.
- One hundred AMP AdamW updates used disjoint, exhaustive encoder and decoder
  groups at learning rates 0.0001 and 0.001. All recorded losses and pre-clip
  gradient norms were finite; the AMP scale remained 1,024.
- Evaluation CE fell from initial training CE 2.910582 to 0.016329, a ratio of
  0.005610 against the required maximum 0.5. Same-sample pixel accuracy was
  0.996751 and mIoU over present classes was 0.952548.
- The 100-line history SHA-256 is
  `f3bff2dbdec740f48a7276a129508803699a559b63be912d235ab8683a29386d`;
  the 164,430,438-byte diagnostic checkpoint SHA-256 is
  `e0269f9027c793286d3a1f7c223be5379a1c7180caffe8f96be334978c760212`;
  the TensorBoard event SHA-256 is
  `255c754fae113eea578e70d5cd32e2384d71d10568296d3fdce920ddc0ae3f1d`.
  Independent post-run checks matched every recorded hash.
- Peak allocated CUDA memory was 409,270,272 bytes and elapsed time was 8.347
  seconds on the RTX 4060 Laptop GPU. These metrics and the saved checkpoint
  are diagnostic only and cannot be used for validation or submission.
- The tool passed an in-container syntax check, and the complete suite passed
  29/29 in 5.618 seconds before the CUDA run.
- Two read-only inspection commands failed without mutating artifacts: the
  checkpoint implementation was first sought at nonexistent
  `engine/checkpoint.py` instead of `engine/checkpointing.py`, and the initial
  host finiteness audit used unavailable `[double]::IsFinite`. The corrected
  container-Python audit verified all 100 loss and gradient values as finite.

## Next action

Implement and verify the formal SegFormer trainer with official VOC train,
independent full validation, safe AMP-overflow handling, resumable unambiguous
history, and best-checkpoint selection by validation mIoU.
