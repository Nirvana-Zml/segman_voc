# Step 0134: Pass reportable A2 formal-run preflight

- Time: 2026-10-26T18:59:15+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Verified clean source, idle RTX 4060, sufficient disk, no Python trainer, absent A2 output, exact config and pretrained-weight hashes, and the controlled 20,000-step CE-only A2 gated-GFF contract.

## Commands

- `git status and revision; approved docker GPU, disk, process, output, hash, and corrected parsed-config checks`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Clean commit 670709997733f99122da0ab0063ab218087e3725; GPU 285/8188 MiB at 0 percent with no compute PID; 757 GB free; no trainer; output absent; config SHA c711ec5498517081b9e6270c255986daef4061b3bea15a59c881e64ddf6ffc95; weight SHA 8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b.`

## Decisions and assumptions

- Commit the preflight ledger before creating the ignored wrapper so the launch can inject one exact clean source revision.

## Next action

Commit the ledger, create and verify the A2 launch wrapper, launch in background, and audit early records.
