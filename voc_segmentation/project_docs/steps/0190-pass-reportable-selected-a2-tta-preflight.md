# Step 0190: Pass reportable selected A2 TTA preflight

- Time: 2026-10-29T08:20:54+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

The approved read-only preflight passed from clean commit b641874e: no competing process, idle 8 GB GPU, sufficient disk, absent output directory, matching locked hashes, and a working evaluator CLI.

## Commands

- `docker exec voc2012-gcb sh -lc <complete reportable TTA preflight>`

## Changed files

- `voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md`

## Evidence

- `GPU 0/8188 MiB at 0%; /workspace 756 GB free; hashes 064aa5da..., 5fcbfa7c..., d2e6fb31..., 2ce5a40d..., cdc1326d...`

## Decisions and assumptions

- Commit the preflight ledger before launch so the full evaluator can record a clean immutable source commit.

## Next action

Commit documentation, prepare a provenance-locked wrapper, and launch exactly one full 1449-image official-val TTA evaluation.
