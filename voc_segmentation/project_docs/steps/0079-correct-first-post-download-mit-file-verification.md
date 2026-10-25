# Step 0079: Correct first post-download MiT file verification

- Time: 2026-10-26T00:12:10+08:00
- Milestone: `04-transformer-baseline`
- Status: **Failed**

## Objective and outcome

The immediate first relative-path read after the verified download reported the final file absent, while the next directory listing showed the atomic rename had completed. Absolute-path verification then matched the official size and SHA-256.

## Commands

- `PowerShell relative-path Get-Item and Get-FileHash after download`

## Changed files

None.

## Evidence

- `Initial ItemNotFound; subsequent absolute path: 54732285 bytes and SHA-256 980b86b60db37b1b1528086f6c53d253d879e9e09ebe07397b40fd275738a3bf`

## Decisions and assumptions

- Retain the verified final file; treat the first read as a failed verification command, not a failed or partial download.

## Next action

Inspect the hash-verified checkpoint and implement deterministic conversion.
