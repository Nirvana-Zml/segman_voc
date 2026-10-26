# Step 0124: Fail first A1 early-history audit invocation

- Time: 2026-10-26T09:40:40+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The first read-only Base64 history audit did not execute because nesting it inside sh -lc allowed PowerShell to truncate the Python command quotes; the active training process was unaffected.

## Commands

- `docker exec process and GPU queries; docker exec sh -lc nested Base64 Python audit`

## Changed files

- `None; audit was read-only and A1 continued running`

## Evidence

- `Process/GPU queries confirmed PIDs 2447/2454 and about 1134 MiB; shell reported unterminated quoted string and PowerShell treated base64.b64decode as a command.`

## Decisions and assumptions

- Run the same Base64 payload directly with docker exec python -c, without an intermediate shell.

## Next action

Repeat the early-history assertions directly in container Python.
