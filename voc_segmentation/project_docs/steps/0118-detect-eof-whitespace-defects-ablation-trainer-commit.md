# Step 0118: Detect EOF whitespace defects during ablation trainer commit

- Time: 2026-10-26T09:20:39+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The staged whitespace check reported extra EOF blank lines in four new ablation configs and the GCB wrapper, but the semicolon-chained command continued and created commit f18e435.

## Commands

- `git add -- voc_segmentation; git diff --cached --check; git commit -m feat:add-controlled-GCB-ablation-trainer`

## Changed files

- `Commit f18e435 was created with five formatting-only EOF defects`

## Evidence

- `git diff --cached --check named lines 61 in all four ablation YAML files and line 24 in tools/train_gcb_seg.py as new blank line at EOF.`

## Decisions and assumptions

- Keep the substantive verified commit, remove only the five trailing blank lines, record both steps, and require a clean whitespace audit before the correction commit.

## Next action

Remove the five EOF blank lines with apply_patch and rerun git diff --check.
