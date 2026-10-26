# Step 0122: Fail first A1 launch-wrapper container check

- Time: 2026-10-26T09:40:29+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

After creating the ignored A1 launch wrapper, its first container syntax and hash queries were denied by the Windows Docker named pipe; no background training command executed.

## Commands

- `git clean-state and full-commit check; create ignored launch.sh; docker exec sh -n launch.sh; docker exec sha256sum launch.sh`

## Changed files

- `Gitignored runs/gcb_ablation_a1_gcm_voc_train/launch.sh only`

## Evidence

- `Both docker exec checks returned permission denied; no started_at, source_commit, history, console, exit code, or result existed at that point.`

## Decisions and assumptions

- Require approved syntax and hash checks to pass before chaining the background launch.

## Next action

Retry wrapper checks with approved Docker access and launch only on zero exit.
