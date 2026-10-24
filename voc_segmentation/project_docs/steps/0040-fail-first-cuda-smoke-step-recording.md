# Step 0040: Fail first CUDA smoke step recording

- Time: 2026-10-24T23:18:08+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The first attempt to create the passed-smoke ledger entry failed in PowerShell parsing because a nested python -c command contained conflicting quotes. The already completed CUDA smoke and its artifact were unaffected.

## Commands

- `python record_step.py with nested python -c syntax-check text`

## Changed files

None.

## Evidence

- `PowerShell ParserError: MissingEndParenthesisInMethodCall before record_step.py executed.`

## Decisions and assumptions

- Record the syntax check as plain command text without nested shell quoting.

## Next action

Create the passed CUDA smoke record with shell-safe arguments.
