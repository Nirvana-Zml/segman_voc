# Step 0027: Pass official VOC2012 and test400 audit

- Time: 2026-10-24T19:28:47+08:00
- Milestone: `02-data-audit-and-sbd`
- Status: **Passed**

## Objective and outcome

Audited all 2,913 official labeled samples and 400 test images; counts, split independence, files, dimensions, label values, hashes, class statistics, and 20 overlays all passed.

## Commands

- `docker exec voc2012-gcb python tools/check_dataset.py`

## Changed files

- `voc_segmentation/project_docs/STATUS.md`
- `voc_segmentation/project_docs/milestones/02-data-audit-and-sbd.md`

## Evidence

- `voc_segmentation/artifacts/data_check/split_summary.json: ok=true`
- `train=1464, val=1449, train_val_overlap=0, test=400`
- `missing/unreadable/mismatched/invalid-label counts: all 0`
- `voc_segmentation/artifacts/data_check/class_pixel_statistics.csv`
- `voc_segmentation/artifacts/data_check/invalid_samples.txt: 0 bytes`
- `voc_segmentation/artifacts/data_check/overlays/: 20 JPEG files`

## Decisions and assumptions

- Official VOC/test400 data gate is passed; milestone 02 remains partial until SBD provenance, conversion, and leak-free trainaug audit pass.

## Next action

Acquire SBD from an authoritative source, record source/license/SHA-256, and implement its conversion without touching official validation.
