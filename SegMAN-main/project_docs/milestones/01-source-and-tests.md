# Milestone 01: Source, configuration, and focused tests

- State: **Reopened; replacement model contract and source pending**
- Last updated: 2026-10-24 17:17 +08:00

## Objective

Provide reproducible VOC2012 configurations, audit/inference/verification tools,
wrappers, a self-built semantic-segmentation model, and focused tests for the
project workflow.

## Implemented scope

- SegMAN-T and SegMAN-S VOC2012 configurations use 21 classes, ignore label
  255, and `reduce_zero_label=False`.
- Audit, test400 prediction, and strict prediction-verification tools exist.
- PowerShell wrappers cover audit, smoke, overfit, formal training, evaluation,
  and prediction.
- Focused tests cover configs, VOC audit behavior, and prediction format.

## Assignment alignment

The course brief explicitly requires Pascal VOC2012 semantic segmentation,
pixel-mask training, model validation, single-channel grayscale test masks,
TensorBoard visualization, and mIoU-oriented evaluation. The active SegMAN-S
configuration matches that scope: it uses `PascalVOCDataset`, reads
`SegmentationClass` masks, has a 21-class segmentation decode head, applies
pixelwise cross-entropy with ignore label 255, and preserves class IDs 0..20 in
single-channel prediction masks. Per-class IoU rows are pixel-level semantic
segmentation metrics, not image-classification output.

## Evidence

- `../../segmentation/local_configs/segman/voc/`
- `../../segmentation/tools/audit_voc.py`
- `../../segmentation/tools/predict_test400.py`
- `../../segmentation/tools/verify_predictions.py`
- `../../tests/`
- `../../scripts/`

## Verification rule

Run `python -m unittest discover -s tests -v` inside `/workspace/segman` after
changes to these files. Record each run as a separate step with its exact exit
status; do not infer a fresh pass from file presence.

## Replacement-model decision

The SegMAN architecture is no longer the active candidate. Existing data audit,
VOC dataset, evaluator, TensorBoard, prediction, and format-verification code
should be reused where their contracts remain valid. Before writing model code,
freeze the replacement model's blocks, tensor shapes, parameter/VRAM target,
normalization, initialization, loss, and focused acceptance tests. Preserve the
old SegMAN code and artifacts as diagnostic evidence.

## Next action

Write and approve the minimal replacement-model contract, then implement its
forward path and focused output/loss tests before any new training run.
