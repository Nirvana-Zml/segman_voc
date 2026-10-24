# Milestone 03: DeepLabV3+ CNN baseline

- State: **In progress — interfaces and CUDA smoke passed**
- Last updated: 2026-10-24 23:17 +08:00

Implement the end-to-end data, loss, metric, checkpoint, TensorBoard,
evaluation, and prediction path with DeepLabV3+ before custom-model work.
Require smoke, two-sample overfit, and independent validation evidence.

## Verified interface evidence

- `datasets/voc.py` loads official train/val and the audited trainaug manifest.
- Training transforms synchronize scale, crop, padding, and horizontal flip;
  masks use nearest-neighbor resize and padding label 255.
- Validation conversion preserves original spatial dimensions.
- `metrics/segmentation.py` accumulates one global confusion matrix, excludes
  label 255, and reports 21-class mIoU, foreground mIoU, pixel accuracy, mean
  pixel accuracy, per-class IoU, and valid-pixel count.
- `models/baselines/deeplabv3plus.py` implements a ResNet-50 DeepLabV3+
  baseline with ASPP and low-level feature fusion. Tests disable weight
  download; reportable runs must use a separately provenance-recorded
  ImageNet checkpoint.
- Final in-container regression suite: 11 tests passed in 2.708 seconds.

## Verification history

The first CUDA smoke stopped during data loading because official VOC masks are
Pillow palette-mode (`P`) images while the new loader accepted only mode `L`.
Preserve the palette indices by reading the uint8 array directly, add a
palette-mask regression test, and rerun the smoke. No model step or result
artifact was produced by the failed attempt.

The first palette regression fixture also failed: a synthetic `P` image made
with `Image.fromarray` and no explicit palette was remapped from index 20 to 1
when saved. Container diagnostics confirmed the real VOC mask remains `P` with
indices `[0, 12, 255]`; the fixture must attach an explicit 256-entry palette.

After fixing palette loading, the second smoke reached forward/backward with a
finite CE loss and about 1.07 GB peak allocated CUDA memory, but its gradient
norm was `NaN`. The result file incorrectly contained `ok=true`; it is rejected
as evidence. Add a finite-gradient hard gate and use a lower diagnostic AMP
initial scale before accepting an optimizer step.

## CUDA smoke evidence

- Command: `docker exec voc2012-gcb python tools/smoke_deeplabv3plus.py`
- Result: `artifacts/cnn_baseline/smoke.json`, `ok=true`,
  `diagnostic_only=true`, `optimizer_step_completed=true`
- Artifact SHA-256:
  `08f9ae8d398caf27c502420cad4d1a7d101f057dc4faa5007cd4e6ab5fbc1aae`
- Device: NVIDIA GeForce RTX 4060 Laptop GPU; PyTorch 2.1.2
- Official train count: 1,464; sampled IDs: `2008_005679`, `2008_002175`
- Input/logits: `[2,3,384,384]` / `[2,21,384,384]`
- Finite CE loss: 3.037793; finite pre-clip gradient norm: 225.873306
- AMP scale: 1,024 before and after; valid pixels: 234,318
- Peak allocated CUDA memory: 1,065,757,696 bytes; elapsed: 1.149 seconds
- Encoder pretraining was disabled, so this is pipeline evidence only and not
  a reportable performance result or submission checkpoint.

## Next action

Implement checkpoint and TensorBoard logging plus a deterministic two-sample
overfit diagnostic. Require a clear saved loss decrease before any formal CNN
baseline training or ImageNet-weight acquisition.
