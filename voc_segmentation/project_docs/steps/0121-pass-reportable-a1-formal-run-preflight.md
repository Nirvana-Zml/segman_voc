# Step 0121: Pass reportable A1 formal-run preflight

- Time: 2026-10-26T09:36:37+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Verified a clean source revision, idle RTX 4060, sufficient disk, no competing Python trainer, absent A1 output directory, exact configuration and pretrained-weight hashes, and the parsed 20,000-step CE-only A1 contract.

## Commands

- `git status --short; git rev-parse HEAD; approved serial docker exec nvidia-smi, df, test, sha256sum, config assertions, and ps process table`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Clean commit 9386a12918cb01d0deee122047e655d6e823ea40; GPU 114/8188 MiB and 0 percent with no compute PID; 757 GB free; no trainer; output absent; config SHA bceb04f2d0a9ae6d35c7b65f6da0c33ead063399c9452c9043b9b49baeef4e66; weight SHA 8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b.`

## Decisions and assumptions

- Commit the preflight ledger first, then inject that new full clean commit into the reportable A1 launch.

## Next action

Commit the ledger, verify the new clean revision, create the ignored launch wrapper, start A1 in background, and inspect early records.
