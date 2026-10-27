# Step 0142: Correct A3 resume source search invocation

- Time: 2026-10-27T12:29:48+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The first read-only resume-path source search passed a pattern beginning with -- to rg without the option terminator, so rg rejected it as an option. The corrected query confirmed the supported resume and history-reconciliation path; no state changed.

## Commands

- `rg -n --resume|resume trainer files; corrected rg -n -- pattern trainer files`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Initial output: rg unrecognized flag --resume|resume; corrected output located --resume parsing and reconcile_history_for_resume.`

## Decisions and assumptions

- Use rg's double-dash option terminator for patterns beginning with a hyphen.

## Next action

Complete the A3 recovery preflight before any runtime mutation.
