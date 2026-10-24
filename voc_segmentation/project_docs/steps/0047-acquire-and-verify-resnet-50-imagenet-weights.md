# Step 0047: Acquire and verify ResNet-50 ImageNet weights

- Time: 2026-10-24T23:50:44+08:00
- Milestone: `03-cnn-baseline`
- Status: **Passed**

## Objective and outcome

Downloaded the official torchvision 0.16.2 ResNet-50 ImageNet-1K V2 artifact, independently verified its bytes and full SHA-256, strictly loaded all tensors, checked finiteness, and passed a local-weight DeepLabV3+ CUDA forward.

## Commands

- `docker exec voc2012-gcb python tools/acquire_resnet50_weights.py`
- `Get-FileHash voc_segmentation/artifacts/weights/resnet50_imagenet1k_v2/resnet50-11ad3fa6.pth -Algorithm SHA256`
- `docker exec voc2012-gcb python -c [local-weight DeepLabV3+ CUDA forward integration check]`

## Changed files

- `voc_segmentation/artifacts/weights/resnet50_imagenet1k_v2/resnet50-11ad3fa6.pth (gitignored acquired weight)`
- `voc_segmentation/artifacts/weights/resnet50_imagenet1k_v2/provenance.json (gitignored evidence)`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`
- `voc_segmentation/project_docs/STATUS.md`

## Evidence

- `Official URL=https://download.pytorch.org/models/resnet50-11ad3fa6.pth; bytes=102540417; SHA256=11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca.`
- `torchvision=0.16.2; tensor_count=320; strict_resnet50_load_ok=true; finite_floating_tensors=true.`
- `Project DeepLabV3+ loaded the local file and produced finite CUDA logits of shape [1,21,64,80].`

## Decisions and assumptions

- Scope is ImageNet-1K classification initialization for the CNN encoder only.
- Torchvision code is BSD-3-Clause; pretrained-model use may inherit ImageNet terms, which describe non-commercial research/educational access. No broader rights are asserted.

## Next action

Implement formal training and full-resolution official-val evaluation with best-21-class-mIoU selection, then run a short pretrained smoke.
