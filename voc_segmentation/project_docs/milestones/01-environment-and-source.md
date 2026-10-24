# Milestone 01: Environment and source contracts

- State: **Passed**
- Last updated: 2026-10-24 19:16 +08:00

## Objective

Provide a clean, pinned, standalone PyTorch environment and enforce the VOC
label, split, model-output, and prediction contracts with focused tests.

## Required evidence

- Clean Docker build for `voc2012-gcb:dev`
- Python/PyTorch/torchvision/CUDA/cuDNN/dependency report
- Successful CUDA tensor allocation on the target GPU
- Passing focused unit tests
- No SegMAN/MMCV/MMSeg/NATTEN/Selective Scan runtime dependency

## Next action

Begin milestone 02: VOC/test400 audit and SBD acquisition/conversion.

## Current evidence

- `quick_validate.py` accepted the new project driver.
- Four focused config/project contract tests passed on the host and again
  inside the image build.
- Six Python files passed an in-memory syntax compile.
- `git diff --check` passed.
- First apt install against Ubuntu official mirrors failed with
  `Connection failed` while fetching `libllvm12`.
- Aliyun mirror rewrite reached `apt-get update` but package fetches returned
  `403 Forbidden`.
- Successful rebuild used `APT_MIRROR=mirrors.tuna.tsinghua.edu.cn` with apt
  retries; image tagged `voc2012-gcb:dev`.
- CUDA environment report:
  `voc_segmentation/artifacts/environment.json`
  (`cuda_available=true`, `cuda_allocation_ok=true`,
  device `NVIDIA GeForce RTX 4060 Laptop GPU`).
- Persistent development container `voc2012-gcb` exists. The user confirmed
  that `python tools/check_environment.py --require-cuda` and
  `python -m unittest discover -s tests -v` passed inside the container. This
  confirmation is recorded as user-reported because full console output was
  not supplied.
