# Step 0046: Implement provenance-checked ResNet-50 weight acquisition

- Time: 2026-10-24T23:34:28+08:00
- Milestone: `03-cnn-baseline`
- Status: **Passed**

## Objective and outcome

Implemented a fixed-version torchvision ResNet-50 ImageNet-1K V2 acquisition tool, explicit local encoder-weight loading, configuration contracts, strict state-dict validation, finiteness checks, and license-scope metadata.

## Commands

- `docker exec voc2012-gcb python -m unittest discover -s tests -v`
- `docker exec voc2012-gcb python -c [AST syntax check for acquire_resnet50_weights.py]`

## Changed files

- `voc_segmentation/tools/acquire_resnet50_weights.py`
- `voc_segmentation/models/baselines/deeplabv3plus.py`
- `voc_segmentation/configs/deeplabv3plus_r50.yaml`
- `voc_segmentation/tests/test_config_contract.py`
- `voc_segmentation/tests/test_deeplabv3plus.py`
- `voc_segmentation/README.md`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`
- `voc_segmentation/project_docs/STATUS.md`

## Evidence

- `15 tests passed in 4.519 seconds inside voc2012-gcb; acquisition tool AST parse passed.`
- `torchvision 0.16.2 enum URL=https://download.pytorch.org/models/resnet50-11ad3fa6.pth; expected SHA256 prefix=11ad3fa6.`

## Decisions and assumptions

- Formal training must load the local provenance-recorded file; runtime auto-download is forbidden for reportable runs.
- Torchvision code is BSD-3-Clause, while pretrained models may inherit dataset terms; record ImageNet educational/non-commercial terms without asserting broader rights.

## Next action

Run the acquisition tool, independently verify the downloaded bytes and state dict, then record full provenance.
