# Step 0022: Build voc2012-gcb with Tsinghua apt

- Time: 2026-10-24T18:42:16+08:00
- Milestone: `01-environment-and-source`
- Status: **Passed**

## Objective and outcome

Switched Dockerfile apt to mirrors.tuna.tsinghua.edu.cn with Acquire::Retries, rebuilt voc2012-gcb:dev, passed in-image unit tests, and saved a CUDA allocation report on the RTX 4060 Laptop GPU. Aliyun had returned 403 Forbidden on this network.

## Commands

- `docker build --pull -t voc2012-gcb:dev .`
- `docker run --rm --gpus all --shm-size=4g --mount type=bind,source=voc_segmentation,target=/workspace/voc_segmentation voc2012-gcb:dev python tools/check_environment.py --require-cuda --output artifacts/environment.json`

## Changed files

- `Dockerfile`
- `voc_segmentation/README.md`
- `voc_segmentation/project_docs/STATUS.md`
- `voc_segmentation/project_docs/milestones/01-environment-and-source.md`

## Evidence

- `Docker image voc2012-gcb:dev`
- `voc_segmentation/artifacts/environment.json`

## Decisions and assumptions

- Default APT_MIRROR to mirrors.tuna.tsinghua.edu.cn; document --build-arg override for other mirrors.

## Next action

Start VOC/test400 audit and SBD acquisition for milestone 02.
