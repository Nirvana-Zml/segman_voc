# Step 0176: Save selected A2 pause at step 13000

- Time: 2026-10-28T18:16:18+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Archived all records through train step 13478, reconciled active history to restart step 13000, and independently verified finite best/restart checkpoints and all pause-manifest hashes.

## Commands

- `python runs/gcb_selected_a2_trainaug/pause_and_save_20261028T181049+0800.py; python runs/gcb_selected_a2_trainaug/verify_pause_20261028T181049+0800.py`

## Changed files

- `voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md; ignored pause archive and manifest`

## Evidence

- `manifest af8155e2...; active history ecb18e96...; full archive ca185957...; last.pt step 13000 b227903d...; best.pt step 12000 mIoU 0.692692 fa3b628f...`

## Decisions and assumptions

- Future continuation must resume from last.pt step 13000 at unique step 13001; best.pt remains selection-only.

## Next action

Await user direction; run a recovery preflight before any continuation.
