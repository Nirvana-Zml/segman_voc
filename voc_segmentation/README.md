# VOC2012 standalone semantic-segmentation project

This directory is the active application root for the course project. The
current design authority is
`../VOC2012_Semantic_Segmentation_Project_Plan.md`; the deleted SegMAN tree is
legacy history and is not a runtime dependency.

## Current implementation boundary

The repository currently contains the reproducible environment, configuration
contracts, project ledger, and focused contract tests. Dataset loaders, model
implementations, training, evaluation, and prediction tools are the next
milestones and must not be claimed complete until their tests and runtime
evidence exist.

The first GPU-safe candidate for the available 8 GB device is MiT-B1 with a
384×384 crop, physical batch size 2, and gradient accumulation to an effective
batch size of 8. MiT-B2 with a 512×512 crop remains a target configuration and
requires a saved memory profile before formal training.

## Reproducible environment

Build from the repository root:

```powershell
docker build --pull -t voc2012-gcb:dev .
```

If apt downloads from Ubuntu's default mirrors fail (slow or `Connection failed`),
rebuild with another mirror, for example:

```powershell
docker build --pull -t voc2012-gcb:dev --build-arg APT_MIRROR=mirrors.aliyun.com .
```

The Dockerfile defaults `APT_MIRROR` to `mirrors.tuna.tsinghua.edu.cn`.

Verify the GPU runtime:

```powershell
docker run --rm --gpus all --shm-size=4g `
  voc2012-gcb:dev `
  python tools/check_environment.py --require-cuda
```

Run focused tests:

```powershell
docker run --rm voc2012-gcb:dev `
  python -m unittest discover -s tests -v
```

For development, mount the source and datasets instead of copying datasets into
the image:

```powershell
docker run --rm -it --gpus all --shm-size=4g `
  --mount "type=bind,source=$((Resolve-Path ./voc_segmentation).Path),target=/workspace/voc_segmentation" `
  --mount "type=bind,source=$((Resolve-Path ./data).Path),target=/data,readonly" `
  voc2012-gcb:dev
```

## Dataset locations

- VOC2012 host root: `../data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012`
- VOC2012 container root: `/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012`
- test400 host root: `../data/test400/test400/images`
- test400 container root: `/data/test400/test400/images`
- SBD host root: `../data/sbd/benchmark_RELEASE/dataset`
- SBD container root: `/data/sbd/benchmark_RELEASE/dataset`
- SBD archive: `../data/sbd/benchmark.tgz` (kept outside the image)

Audit the mounted official data inside the persistent container:

```powershell
docker exec voc2012-gcb python tools/check_dataset.py
```

The command reads all 2,913 labeled samples and 400 test images, then writes
`artifacts/data_check/split_summary.json`, class statistics, an invalid-sample
list, and 20 deterministic overlay images.

After downloading the official SBD archive to `/data/sbd/benchmark.tgz` and
extracting it, build the leak-free trainaug manifest and converted masks:

```powershell
docker exec voc2012-gcb python tools/prepare_voc_sbd.py
```

The converter verifies the official archive MD5, records SHA-256 provenance,
prefers official VOC labels for the 1,464 train IDs, converts only the 9,118
additional SBD labels, and rejects any official validation overlap.

The verified outputs are under `artifacts/sbd/`: 10,582 trainaug entries,
9,118 converted extra masks, provenance hashes, and the conversion audit.

## CNN baseline diagnostics

Run the one-step DeepLabV3+ CUDA smoke inside the persistent container:

```powershell
docker exec voc2012-gcb python tools/smoke_deeplabv3plus.py
```

This diagnostic uses the official VOC train split, 384×384 crops, batch size 2,
AMP, CE loss, and one optimizer step. It deliberately disables pretrained
weight download and writes a diagnostic-only result to
`artifacts/cnn_baseline/smoke.json`; it is not a reportable checkpoint.

Run the deterministic two-sample overfit gate:

```powershell
docker exec voc2012-gcb python tools/overfit_deeplabv3plus.py
```

It uses two fixed official-train samples only, disables pretrained weights,
and writes a diagnostic checkpoint, loss history, TensorBoard event file, and
validated result under `artifacts/cnn_baseline/overfit_two_samples/`. These
diagnostic weights must never be used for validation or submission.

Acquire the exact torchvision 0.16.2 ResNet-50 ImageNet-1K V2 encoder weights:

```powershell
docker exec voc2012-gcb python tools/acquire_resnet50_weights.py
```

The command downloads from `download.pytorch.org`, validates the published
SHA-256 prefix, strictly loads the state dict into torchvision ResNet-50,
checks all floating tensors for finiteness, and records source, scope, license
notes, byte size, and full SHA-256 under `artifacts/weights/`. Training must use
the explicit local path in `configs/deeplabv3plus_r50.yaml`.

Run the short pretrained trainer diagnostic:

```powershell
$vocCommit = git rev-parse HEAD
docker exec voc2012-gcb python tools/train_deeplabv3plus.py --diagnostic-only --git-commit $vocCommit --output-dir artifacts/cnn_baseline/pretrained_formal_smoke_v4 --max-iterations 1 --validate-every 1 --validation-limit 2
```

The validation limit is accepted only in diagnostic mode. Its metrics are not
reportable. The formal VOC-train baseline uses the versioned defaults and the
entire official validation split. From PowerShell, require a clean host
worktree and inject its full commit because the container intentionally does
not mount `.git`:

```powershell
if (git status --porcelain) { throw "Commit the worktree before formal training." }
$vocCommit = git rev-parse HEAD
docker exec voc2012-gcb python tools/train_deeplabv3plus.py --git-commit $vocCommit --git-worktree-clean
```

The trainer writes the resolved configuration, JSONL history, full-resolution
validation records, TensorBoard events, and validation-mIoU-selected
checkpoints under `runs/deeplabv3plus_r50_voc_train/`. To continue an
interrupted run, pass the checkpoint with the greatest saved step (normally
`best.pt` or `last.pt`) and the same output directory:

```powershell
docker exec voc2012-gcb python tools/train_deeplabv3plus.py --resume runs/deeplabv3plus_r50_voc_train/last.pt --git-commit $vocCommit --git-worktree-clean
```

Before appending a resumed segment, the trainer archives the complete existing
JSONL under `history_archive/` and removes active records beyond the checkpoint
step. It rejects duplicate completed train/validation records at or before the
checkpoint, preventing ambiguous step histories after an interrupted run.

With AMP enabled, a verified GradScaler overflow safely skips the unsafe
optimizer update, reduces the scale, records an `amp_overflow` JSONL event,
and retries the same logical iteration. The trainer still stops on non-AMP
non-finite gradients, missing scale backoff, more than four consecutive
overflows, or more than 32 total overflows. The injection flag
`--diagnostic-force-amp-overflow-once` is accepted only in diagnostic mode.

## Transformer baseline diagnostics

Acquire and convert the fixed NVIDIA MiT-B1 ImageNet-1K encoder checkpoint:

```powershell
docker exec voc2012-gcb python tools/acquire_mit_b1_weights.py
```

The tool pins NVIDIA model revision
`13ddceec4e8bdf401e7cd7acf5aebc526222518c`, verifies the official LFS byte
count and full SHA-256, removes only the ImageNet classifier, converts the
Hugging Face encoder naming to the project's native MiT naming, strictly loads
all 176 encoder tensors, checks finite values and a finite forward pass, and
writes `artifacts/weights/mit_b1_imagenet1k/provenance.json`. The official
SegFormer license limits use to non-commercial research or evaluation; this
course project uses the weights only for educational evaluation.

Run the pretrained 384×384 CUDA optimizer-step smoke:

```powershell
docker exec voc2012-gcb python tools/smoke_segformer.py
```

It uses two official VOC-train samples, batch size 2, AMP, AdamW encoder and
decoder parameter groups, and the explicit hash-verified local checkpoint. It
writes `artifacts/transformer_baseline/smoke.json` with
`diagnostic_only=true`; its weights and metrics are never submission evidence.

Run the deterministic pretrained two-sample overfit gate:

```powershell
docker exec voc2012-gcb python tools/overfit_segformer.py
```

It fixes the same two official-train samples at 128x128, verifies the local
MiT-B1 checkpoint against its provenance, and runs 100 AMP updates with
separate encoder and decoder AdamW groups. The command accepts the gate only
when the final evaluation CE is at most half the initial training CE and all
recorded losses and gradients are finite. It writes a diagnostic-only
checkpoint, JSONL history, TensorBoard event, and `result.json` under
`artifacts/transformer_baseline/overfit_two_samples/`; none are eligible for
validation, model selection, or submission.

## Project layout

```text
configs/       Versioned experiment inputs
datasets/      VOC/SBD loading, synchronized transforms, boundary targets
models/        Baselines and GCB-Seg
losses/        CE, Lovász, auxiliary, and boundary losses
metrics/       Confusion-matrix and segmentation metrics
engine/        Trainer, evaluator, checkpoint handling
tools/         Data, training, evaluation, inference, and validation commands
tests/         Focused contract and behavior tests
project_docs/  Status, decisions, milestones, and immutable step records
```

Generated `artifacts/`, `runs/`, `checkpoints/`, `outputs/`, and `predictions/`
are gitignored. Preserve exact configs and evidence paths in `project_docs/`.
