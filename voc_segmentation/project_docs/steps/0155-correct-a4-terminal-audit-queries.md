# Step 0155: Correct A4 terminal audit queries

- Time: 2026-10-28T08:05:32+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The first Docker terminal query was denied; approved retry passed. Three read-only audit assumptions were then corrected: AMP scale field names, TensorBoard overflow-tag multiplicity, and the A0 config filename. No run evidence changed.

## Commands

- `docker terminal snapshot; a4_terminal_audit.py attempts; A0-A4 result/config hash query`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger only`

## Evidence

- `Docker permission denial; KeyError scale_before; TensorBoard assertion showing per-step zeros plus six duplicate overflow events; missing segformer_b1_voc_train.yaml query`

## Decisions and assumptions

- Reject incomplete audit attempts and retain only the corrected full audit and verified segformer_b1.yaml hash.

## Next action

Run and record the corrected complete A4 terminal audit.
