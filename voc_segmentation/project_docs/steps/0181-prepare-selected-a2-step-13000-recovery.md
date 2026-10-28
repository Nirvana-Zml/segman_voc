# Step 0181: Prepare selected A2 step-13000 recovery

- Time: 2026-10-28T18:50:26+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Preserved pause evidence, verified archive hashes and idle GPU, and validated a wrapper that resumes last.pt under clean preflight commit fd1f6d3.

## Commands

- `sh -n runs/gcb_selected_a2_trainaug.launch.sh; sha256sum <wrapper and interruption archive artifacts>`

## Changed files

- `Ignored interruption archive and resume wrapper`

## Evidence

- `wrapper 8c42cd24...; paused console 02a004a5...; exit 9d9b1872...; prior wrapper 0fcca8f8...; GPU 0/8188 MiB`

## Decisions and assumptions

- Use last.pt step 13000, not best.pt, for recovery.

## Next action

Launch exactly once and audit the first continued records.
