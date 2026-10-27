# Step 0153: Resume A4 from verified step-1000 checkpoint

- Time: 2026-10-28T00:03:38+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **In progress**

## Objective and outcome

Archived the paused console and exit code, launched A4 once from verified best.pt under clean commit f51af07, and confirmed exact unique continuation from step 1001 with finite composite losses.

## Commands

- `docker exec -d voc2012-gcb sh /workspace/voc_segmentation/runs/gcb_ablation_a4_lovasz_voc_train/resume.sh; independent history/process/GPU audit`

## Changed files

- `Ignored A4 interruption archive, resume wrapper, live console/history/TensorBoard files; STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `wrapper SHA 255e6000...; PIDs 125/132; reconciliation kept 1001 dropped 0; unique steps through 1070; first resumed step 1001; GPU 1794 MiB at 21 percent`

## Decisions and assumptions

- Keep A4 active toward 20000 steps; terminal comparison remains forbidden until result and artifacts pass audit.

## Next action

Monitor the step-2000 full official validation and eventual terminal result.
