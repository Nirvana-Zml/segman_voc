# Step 0132: Record incomplete first A2 Docker preflight

- Time: 2026-10-26T18:59:05+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The first batched read-only container preflight was incomplete because intermittent Docker Desktop named-pipe permission denial blocked GPU and process queries. Disk, output-absence, and SHA checks passed; no output or trainer was created.

## Commands

- `batched docker exec nvidia-smi, df, ps, output test, and sha256sum`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `GPU and ps returned permission denied; df showed 757 GB free; A2 output absent; config and weight hashes matched.`

## Decisions and assumptions

- Reject the incomplete attempt and repeat missing checks under approved serial Docker access.

## Next action

Repeat GPU and process checks with approved access; do not launch until the complete preflight passes.
