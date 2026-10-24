# Milestone 03: DeepLabV3+ CNN baseline

- State: **In progress — formal trainer verified; reportable run pending**
- Last updated: 2026-10-25 00:18 +08:00

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

The formal trainer, evaluator, regression tests, accepted v4 diagnostic
evidence, and all preceding failure records were committed as `b4201fc`
(`feat: add formal DeepLabV3+ trainer`). Runtime weights and logs remained
gitignored.

The first attempt to commit the verified formal trainer was blocked before
staging because the managed sandbox denied creation of `.git/index.lock`.
No repository state changed. Record the permission-boundary failure and retry
the same scoped Git add/commit operation with elevated repository write access.

A combined mount inspection assumed a root `docker-compose.yml`, which does not
exist, but its `docker inspect` half succeeded and showed only two bind mounts:
host `data/` to read-only `/data` and host `voc_segmentation/` to
`/workspace/voc_segmentation`. The repository `.git` directory is therefore
intentionally unavailable inside the container. Supply the host commit
explicitly for reportable runs instead of weakening provenance checks.

The first v4 diagnostic stopped before data loading because the new Git-state
provenance gate ran `git rev-parse HEAD` inside the container and received exit
status 128. No v4 training artifact was accepted. Determine whether the bind
mount lacks `.git` or Git rejected repository ownership, then make provenance
capture compatible without silently labeling an unknown state as clean.

The first independent artifact-verification one-liner failed before reading
the result because PowerShell stripped the Python path string quotes, producing
`NameError: name 'artifacts' is not defined`. No artifact was modified; rerun
with PowerShell-compatible outer double quotes and inner single quotes.

The first pretrained formal-trainer diagnostic completed forward, backward,
checkpoint, TensorBoard, and two-image native-resolution validation, but its
only optimizer iteration used learning rate 0 because the polynomial schedule
was evaluated at the endpoint of a one-step run. Reject
`artifacts/cnn_baseline/pretrained_formal_smoke/` as optimizer-step evidence;
special-case a one-step diagnostic to use the configured base learning rate
and rerun into a new artifact directory.

The first attempt to run the expanded regression suite after adding the formal
trainer could not connect to Docker Desktop from the managed host shell
(`permission denied` on `dockerDesktopLinuxEngine`). No container test process
started, so this is recorded as an infrastructure invocation failure rather
than a test failure; rerun through the approved Docker execution boundary.

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

Launch the reportable VOC-train run, evaluate all 1,449 official validation
images at each configured interval, and independently verify the selected best
checkpoint and complete metrics before advancing to the transformer baseline.

## Formal trainer and pretrained diagnostic evidence

- `tools/train_deeplabv3plus.py` implements AMP, physical batch 2 with four-way
  accumulation (effective batch 8), finite loss/gradient gates, norm clipping,
  linear warmup plus polynomial decay, and checkpoint restart.
- The ImageNet-initialized encoder uses learning rate 0.001 while the new ASPP
  and decoder use 0.01. A runtime coverage gate rejects overlapping or missing
  optimizer parameters.
- Validation preserves each source image's dimensions, aggregates one global
  21-class confusion matrix, ignores label 255, and selects `best.pt` strictly
  by validation mIoU. A validation limit is rejected outside diagnostic mode.
- Complete in-container regression suite: 20 tests passed in 5.722 seconds.
- Accepted command: `docker exec voc2012-gcb python tools/train_deeplabv3plus.py
  --diagnostic-only --git-commit
  7496a7a0a641e9039f4ada378afc81f3c5b38c2e --output-dir
  artifacts/cnn_baseline/pretrained_formal_smoke_v4 --max-iterations 1
  --validate-every 1 --validation-limit 2`.
- Result: `artifacts/cnn_baseline/pretrained_formal_smoke_v4/result.json`,
  `ok=true`, `diagnostic_only=true`, full official validation `false`.
- One effective-batch-8 step: train CE 2.973795, gradient norm 7.913279,
  encoder/decoder learning rates 0.001/0.01; elapsed 15.158 seconds and peak
  allocated CUDA memory 1,262,054,912 bytes.
- The two-image native-resolution diagnostic mIoU 0.002043 is pipeline evidence
  only and is not a baseline metric.
- Best diagnostic checkpoint: 323,230,158 bytes, step 1, SHA-256
  `ee43b5f640535fef43a89d233b7b4c2a60cb9d4c582df9dee11c85436a12a9b6`.
- TensorBoard event SHA-256:
  `f0f5717bc487b900c575849d76b14044e3b7a2e014d7a16c7d0396c2c411fe44`;
  seven expected scalar tags each contain one point.
- Independent verification confirmed the checkpoint/event hashes,
  `diagnostic_only=true`, checkpoint step 1, both named optimizer groups, 188
  optimizer state entries, and two JSONL history records.
- Result provenance also records base commit
  `7496a7a0a641e9039f4ada378afc81f3c5b38c2e`, Python 3.10.13, CUDA 12.1,
  cuDNN 8902, train split SHA-256
  `aa623bd2c8ce4443a8aaae51c524c0eb165e8e44caf6aa9e3ad33d7b75a3ef20`,
  and val split SHA-256
  `cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d`.

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
