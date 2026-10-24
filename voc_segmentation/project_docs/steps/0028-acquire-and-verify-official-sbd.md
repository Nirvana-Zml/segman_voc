# Step 0028: Acquire and verify official SBD

- Time: 2026-10-24T19:52:59+08:00
- Milestone: `02-data-audit-and-sbd`
- Status: **Passed**

## Objective and outcome

Downloaded the official Berkeley SBD archive to data/sbd, matched its published MD5, recorded SHA-256, and extracted 11,355 images plus 11,355 class-label MAT files.

## Commands

- `curl.exe -L --retry 5 --retry-delay 5 -o data/sbd/benchmark.tgz https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz`
- `Get-FileHash data/sbd/benchmark.tgz -Algorithm MD5`
- `Get-FileHash data/sbd/benchmark.tgz -Algorithm SHA256`

## Changed files

- `data/sbd/benchmark.tgz (gitignored dataset artifact)`
- `data/sbd/benchmark_RELEASE/dataset (gitignored extracted dataset)`

## Evidence

- `Archive bytes=1419539633; MD5=82b4d87ceb2ed10f6038a1cba92111cb; SHA256=6a5a2918d5c73ce032fdeba876574d150d9d04113ab87540a1304cbcc715be53`
- `Extracted images=11355; class MAT files=11355; SBD train=8498; SBD val=2857`

## Decisions and assumptions

- Use the authoritative Berkeley archive and retain source, checksum, citation, and license note before reportable use.

## Next action

Implement and test deterministic SBD mask conversion and leak-free trainaug assembly.
