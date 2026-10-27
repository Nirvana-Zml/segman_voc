# Step 0146: Pass reportable A4 formal-run preflight

- Time: 2026-10-27T21:54:46+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Verified clean source, idle RTX 4060, sufficient disk, no Python trainer, absent A4 output, exact config and pretrained-weight hashes, and the controlled A4 contract that adds only 0.5 Lovasz loss to A3.

## Commands

- `git status and revision; approved docker GPU, disk, process, output, hash, and parsed A3/A4 config checks`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Clean commit 6c8cfcb13261438c871aec992d1d0270bba3c0c5; GPU 1107/8188 MiB at 0 percent with no compute PID; 757 GB free; no trainer; output absent; config SHA f1d5532abf381ea2d31c7a60d84cc5d7e98def6fdd7553b43207b41303c0b596; weight SHA 8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b.`

## Decisions and assumptions

- Commit the preflight ledger before creating the ignored launch wrapper so the launch can inject one exact clean source revision.

## Next action

Commit the ledger, create and verify the A4 wrapper, launch in background, and audit early CE plus Lovasz plus edge records.
