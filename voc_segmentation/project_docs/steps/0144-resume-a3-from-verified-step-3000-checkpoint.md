# Step 0144: Resume A3 from verified step-3000 checkpoint

- Time: 2026-10-27T12:36:31+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **In progress**

## Objective and outcome

Preserved the interrupted console and exit code, resumed A3 from the verified step-3000 best checkpoint under clean recovery commit c3638045039867aaa77addd8579b32b4d532ba6d, and verified unique finite composite-loss continuation from step 3001.

## Commands

- `archive original console and exit; sh -n and sha256sum resume wrapper; docker exec -d resume; process, GPU, provenance, history, formula, and terminal-artifact audits`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger; ignored A3 interruption archive and recovery runtime files`

## Evidence

- `Wrapper SHA 1588ffcd...; archived console SHA e8e1ca01... and exit 143; PIDs 97/104; reconciliation kept 3003 dropped 0; unique train steps through 3058; first resumed step 3001; all resumed values finite with nonzero edge and CE plus 0.2 edge formula; GPU 1675 MiB at 11 percent; no new exit or result.`

## Decisions and assumptions

- Treat A3 as active, not complete. Use step 4000 as the first post-resume full-validation acceptance gate.

## Next action

At step 4000, audit the complete 1449-image validation, checkpoint selection, unique history, process health, and terminal-artifact absence.
