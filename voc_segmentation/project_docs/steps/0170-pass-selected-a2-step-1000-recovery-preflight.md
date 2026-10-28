# Step 0170: Pass selected A2 step-1000 recovery preflight

- Time: 2026-10-28T12:52:12+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Verified idle resources, all registered hashes, finite non-diagnostic step-1000 checkpoint, exact selected-A2/trainaug contract, preserved stop evidence, and unique reconciled history through step 1000.

## Commands

- `docker exec voc2012-gcb sh -lc '<resource checks>; cd /workspace/voc_segmentation && python runs/gcb_selected_a2_trainaug/resume_preflight.py'`

## Changed files

- `voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/06-formal-training-and-selection.md`

## Evidence

- `best.pt SHA-256 8005ad74... with 1036 finite tensors; history SHA-256 abb2bc8c...; GPU 0/8188 MiB; 756 GB free`

## Decisions and assumptions

- Resume only from best.pt at step 1000 and require first continued step 1001.

## Next action

Commit this preflight ledger, archive stop console/exit evidence, and launch one resume under the new clean commit.
