# Step 0195: Correct TTA selection artifact hash check

- Time: 2026-10-29T08:38:04+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

The first local inline-Python JSON/hash check failed from PowerShell quote mangling; it did not write or modify either selection artifact.

## Commands

- `python -c <inline JSON and SHA checker>`

## Changed files

- `voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md`

## Evidence

- `Python SyntaxError: unterminated string literal; project_status independently recognized formal selection`

## Decisions and assumptions

- Use PowerShell ConvertFrom-Json plus Get-FileHash for the corrected read-only verification.

## Next action

Verify both artifact JSON files and hashes, then run project_status.
