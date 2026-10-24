# Decision 0001: Standalone project reset

- Time: 2026-10-24 17:53 +08:00
- Status: Active

## Decision

The user deleted `SegMAN-main/` and adopted
`VOC2012_Semantic_Segmentation_Project_Plan.md` as the current plan. The active
application root is now `voc_segmentation/`, implemented in standalone PyTorch
without MMCV, MMSegmentation, NATTEN, or Selective Scan.

## Legacy handling

Do not restore the deleted tree unless explicitly requested. Tracked legacy
source and documentation remain in commit `111f8a0`; legacy runtime results do
not satisfy the new model's delivery gates.
