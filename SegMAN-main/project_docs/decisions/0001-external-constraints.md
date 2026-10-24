# Decision record 0001: External project constraints

- Status: **Open**
- Last updated: 2026-10-24 16:15 +08:00

## Known defaults

- Use only the official VOC2012 train split for supervised training and the
  official validation split for selection unless course rules say otherwise.
- Do not use extra segmentation datasets or segmentation-pretrained weights
  without explicit permission.
- A random-initialized baseline is permitted as the safe implementation path
  while ImageNet classification-pretraining permission remains unknown.
- Test400 has no labels and cannot support a test mIoU claim.

## Decisions still required

| Decision | Current value | When it becomes blocking |
| --- | --- | --- |
| ImageNet encoder pretraining allowed? | Unknown | Before using such a checkpoint |
| Extra segmentation data/weights allowed? | Default: no | Before any proposed use |
| Student name and ID | Unknown | Final report/package naming |
| Maximum training time and storage | Unknown | When a formal experiment would exceed available resources |

Update this record only when the user or an authoritative course document
provides the missing value. Include the source and date of the decision.
