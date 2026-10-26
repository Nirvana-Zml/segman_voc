# Step 0137: Correct A2 TensorBoard overflow-count assumption

- Time: 2026-10-27T07:56:17+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The second complete A2 audit advanced past tag-set verification but incorrectly required every train-related TensorBoard tag to contain exactly 20000 samples. AMP scale and overflow tags correctly contain 20006 because six recovered overflow events supplement the 20000 completed train steps.

## Commands

- `docker exec corrected full A2 audit; inspect SCALARS=0 count and step samples for all 15 tags`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Eleven ordinary train tags had 20000 points, three validation tags had 20, and AMP/scale plus System/amp_overflow had 20006; exactly six overflow values were nonzero.`

## Decisions and assumptions

- Validate normal train tags, validation tags, and AMP event multiplicities under separate contracts.

## Next action

Rerun the complete audit with 20000 completed-step points plus six explicit AMP events.
