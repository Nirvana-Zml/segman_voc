# Step 0035: Implement and test CNN baseline interfaces

- Time: 2026-10-24T23:10:25+08:00
- Milestone: `03-cnn-baseline`
- Status: **Passed**

## Objective and outcome

Implemented synchronized VOC/trainaug loading, label-safe training/evaluation transforms, global confusion-matrix metrics, and a ResNet-50 DeepLabV3+ model; the complete container suite passed.

## Commands

- `docker exec voc2012-gcb python -m unittest discover -s tests -v`

## Changed files

- `voc_segmentation/datasets/voc.py`
- `voc_segmentation/datasets/__init__.py`
- `voc_segmentation/metrics/segmentation.py`
- `voc_segmentation/metrics/__init__.py`
- `voc_segmentation/models/baselines/deeplabv3plus.py`
- `voc_segmentation/models/baselines/__init__.py`
- `voc_segmentation/tests/test_voc_dataset.py`
- `voc_segmentation/tests/test_segmentation_metrics.py`
- `voc_segmentation/tests/test_deeplabv3plus.py`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`
- `voc_segmentation/project_docs/STATUS.md`

## Evidence

- `11 tests passed in 3.200 seconds inside voc2012-gcb.`
- `DeepLabV3+ test output shape=(1,21,64,80) with finite logits.`
- `Transform test preserved labels 0,1,3,4,20,255 and used ignore padding 255.`

## Decisions and assumptions

- Unit tests construct the encoder without pretrained weights; ImageNet weights require provenance before a reportable run.

## Next action

Implement and run a one-step CUDA smoke diagnostic on official VOC train.
