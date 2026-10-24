# Step 0030: Pass SBD conversion tests

- Time: 2026-10-24T19:53:07+08:00
- Milestone: `02-data-audit-and-sbd`
- Status: **Passed**

## Objective and outcome

Implemented SBD provenance, GTcls segmentation loading, label-preserving PNG conversion, VOC-label preference, and val-excluded trainaug assembly; after correcting the fixture, all seven tests passed in the target container.

## Commands

- `docker exec voc2012-gcb python -m unittest discover -s tests -v`

## Changed files

- `voc_segmentation/datasets/sbd.py`
- `voc_segmentation/datasets/__init__.py`
- `voc_segmentation/tools/prepare_voc_sbd.py`
- `voc_segmentation/tests/test_sbd_conversion.py`
- `voc_segmentation/README.md`

## Evidence

- `7 tests passed in 1.567 seconds inside voc2012-gcb.`

## Decisions and assumptions

- Prefer official VOC masks for all 1,464 VOC-train IDs; convert only the 9,118 SBD extras; never shift labels.

## Next action

Run full conversion and validate the resulting trainaug artifacts.
