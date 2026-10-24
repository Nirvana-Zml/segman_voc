# Step 0051: Fail first pretrained smoke artifact verification command

- Time: 2026-10-25T00:03:14+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The independent read-only verification one-liner did not read any artifact because PowerShell removed quotes around the Python Path literal, causing a NameError. No runtime evidence was changed.

## Commands

- `docker exec voc2012-gcb python -c <artifact verification one-liner>`

## Changed files

- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `Python NameError: name 'artifacts' is not defined`

## Decisions and assumptions

- Use PowerShell-compatible outer double quotes and Python inner single quotes.

## Next action

Rerun the same read-only hash, metadata, and TensorBoard verification.
