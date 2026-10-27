# Step 0139: Pass reportable A3 formal-run preflight

- Time: 2026-10-27T08:03:49+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Verified clean source, idle RTX 4060, sufficient disk, no Python trainer, absent A3 output, exact config and pretrained-weight hashes, and the controlled A3 contract that adds only BAB plus 0.2 edge loss to A2.

## Commands

- `git status and revision; approved docker GPU, disk, process, output, hash, and parsed A2/A3 config checks`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Clean commit 38d68e2f011291dd8e0580bc4206abbd39a81315; GPU 846/8188 MiB at 0 percent with no compute PID; 757 GB free; no trainer; output absent; config SHA 11127f069b01f0464df539a3974aace591d2267868867f6aaf8aee4a6393354c; weight SHA 8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b.`

## Decisions and assumptions

- Commit the preflight ledger before creating the ignored launch wrapper so the launch can inject one exact clean source revision.

## Next action

Commit the ledger, create and verify the A3 wrapper, launch in background, and audit early composite-loss and boundary records.
