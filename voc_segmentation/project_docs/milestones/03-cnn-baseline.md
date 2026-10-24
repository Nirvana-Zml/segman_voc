# Milestone 03: DeepLabV3+ CNN baseline

- State: **In progress — pretrained weights verified; formal run pending**
- Last updated: 2026-10-24 23:50 +08:00

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

## Two-sample overfit evidence

- Command: `docker exec voc2012-gcb python tools/overfit_deeplabv3plus.py`
- Result: `artifacts/cnn_baseline/overfit_two_samples/result.json`,
  `ok=true`, `diagnostic_only=true`
- Fixed official-train IDs: `2008_005679`, `2008_002175`; val was not read
- Input: `[2,3,128,128]`; 100 AdamW steps at learning rate 0.001
- Initial/final evaluation CE: 3.002533 / 0.017127
- Loss ratio: 0.005704; required maximum: 0.5
- Diagnostic pixel accuracy: 0.994826 on the same two training samples
- Peak allocated CUDA memory: 891,774,464 bytes; elapsed: 6.455 seconds
- History: 100 JSONL entries, SHA-256
  `12d5acfc7b4d0928c49b1e77ef59f009eb7061a5c18a1d0899f55a1b03d5600b`
- Checkpoint: 484,746,098 bytes, step 100, `diagnostic_only=true`,
  SHA-256 `925a94a271ee13682f5d49477602355b210ce2cb3107838dc663b58ee5215e5b`
- TensorBoard event SHA-256:
  `532faf99fdc41d90b6fd39204b972a94e6292655e8d3ca518d4e0b95dc72e38a`
- TensorBoard tags `Loss/train_ce`, `System/gradient_norm`, and
  `LearningRate/decoder` each contain exactly 100 scalar points.
- These same-sample metrics demonstrate learnability only; they are not
  validation performance and the checkpoint is forbidden for submission.

## Next action

Implement the formal trainer and full-resolution official-val evaluator with
best-21-class-mIoU checkpoint selection, resolved-config capture, TensorBoard,
and restart support. Run a short pretrained-weight smoke before launching a
reportable baseline.

## ResNet-50 ImageNet-1K V2 provenance

- Weight ID: `ResNet50_Weights.IMAGENET1K_V2`
- Fixed environment: torch 2.1.2, torchvision 0.16.2
- Official source:
  `https://download.pytorch.org/models/resnet50-11ad3fa6.pth`
- Torchvision 0.16 documentation:
  `https://docs.pytorch.org/vision/0.16/models/generated/torchvision.models.resnet50.html`
- Torchvision recipe:
  `https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621`
- Local artifact:
  `artifacts/weights/resnet50_imagenet1k_v2/resnet50-11ad3fa6.pth`
- Bytes: 102,540,417
- Full SHA-256:
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`
- Published filename hash prefix `11ad3fa6` matches the computed digest
- Validation: 320 tensors, strict torchvision ResNet-50 state-dict load passed,
  every floating tensor finite
- Project integration: local-file DeepLabV3+ CUDA forward produced finite
  logits with shape `[1,21,64,80]`
- Scope: ImageNet-1K classification initialization for the DeepLabV3+
  ResNet-50 encoder; not segmentation or VOC pretraining
- License boundary: torchvision 0.16.2 source is BSD-3-Clause. Torchvision
  explicitly warns that pretrained-model permissions may derive from the
  training dataset and are the user's responsibility. ImageNet describes
  access for non-commercial research/education under its conditions. This
  course project uses the weights only for educational initialization and
  asserts no broader rights.
- License sources:
  `https://github.com/pytorch/vision/blob/v0.16.2/LICENSE`,
  `https://github.com/pytorch/vision#pre-trained-model-license`, and
  `https://www.image-net.org/download-images.php`
- Machine-readable evidence:
  `artifacts/weights/resnet50_imagenet1k_v2/provenance.json`
