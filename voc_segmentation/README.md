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
