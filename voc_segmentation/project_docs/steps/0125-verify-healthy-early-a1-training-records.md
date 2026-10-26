# Step 0125: Verify healthy early A1 training records

- Time: 2026-10-26T09:40:46+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Direct container-Python audit verified that the reportable A1 run is active with unique sequential finite early records, CE-only total loss, correct full runtime metadata, injected clean source commit, and no premature terminal artifacts.

## Commands

- `docker exec voc2012-gcb python -c Base64-decoded-A1-early-history-audit`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `51 unique train records for steps 1-51; step 51 CE 2.916139; gradient norm 26.750391; AMP scale 1024; no validation, exit_code, or result; resolved max 20000, full validation, diagnostic false; source commit c956520.`

## Decisions and assumptions

- Keep A1 running and wait for the first complete official validation before accepting any mIoU.

## Next action

Monitor step 1000, then verify all 1449 validation samples and the selected non-diagnostic checkpoint.
