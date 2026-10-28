# Step 0180: Pass selected A2 step-13000 recovery preflight

- Time: 2026-10-28T18:45:40+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Verified idle resources, every registered pause/config/data hash, finite step-13000 restart and step-12000 best checkpoints, and unique active history through step 13000.

## Commands

- `docker exec voc2012-gcb sh -lc '<resource checks>; python runs/gcb_selected_a2_trainaug/resume_preflight_step13000.py'`

## Changed files

- `voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md`

## Evidence

- `GPU 0/8188 MiB; 756 GB free; last.pt b227903d... step 13000; best.pt fa3b628f...; 13016 active records; pause manifest af8155e2...`

## Decisions and assumptions

- Resume only from last.pt step 13000 and require first continuation step 13001.

## Next action

Commit the preflight ledger, archive pause console/exit evidence, and launch one recovery.
