# Step 0014: Estimate remaining baseline runtime

- Time: 2026-10-24T16:59:57+08:00
- Milestone: `05-formal-training`
- Status: **Information**

## Objective and outcome

Estimated runtime from actual epoch-to-epoch validation timestamps: about 10–15 minutes to the epoch-5 diagnostic gate and about 5–6 hours to finish all 55 epochs if continued.

## Commands

- `Inspect latest train record and validation summary timestamps in launcher.stderr.log and 20261024_083317.log.json`

## Changed files

- `SegMAN-main/project_docs/STATUS.md`
- `SegMAN-main/project_docs/milestones/05-formal-training.md`

## Evidence

- `Observed iteration 2900 at 2026-10-24 16:59 +08:00`
- `Validation summaries were approximately 6 minutes 20 seconds apart`

## Decisions and assumptions

- Use observed wall-clock epoch cycles rather than the inflated MMCV ETA that folds validation delay into iteration timing.

## Next action

Inspect the epoch-5 validation in about 10–15 minutes; continue the full run only if foreground learning emerges or diagnosis supports continuing.
