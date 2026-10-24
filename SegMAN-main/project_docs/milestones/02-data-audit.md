# Milestone 02: VOC2012 and test400 data audit

- State: **Passed**
- Last updated: 2026-10-24 16:15 +08:00

## Objective

Prove that the official train/validation splits and the unlabeled test400 input
are readable and comply with the project data contract.

## Observed result

- Train samples: 1464
- Validation samples: 1449
- Train/validation overlap: 0
- Missing or size-mismatched pairs: 0
- Observed labels: `0..20` and `255`
- Invalid labels: none
- Test images: 400 with 400 unique stems and no reported errors

## Evidence

- `../../work_dirs/data_audit.json`
- `../../segmentation/tools/audit_voc.py`

## Acceptance result

Passed. Rerun and record the audit if dataset contents, mounts, split files, or
the audit implementation change.

## Next action

Preserve the audited split and label contract during all experiments.
