# Step 0128: Expose TensorBoard reservoir assumption in first A1 terminal audit

- Time: 2026-10-26T18:45:39+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The first complete A1 terminal audit passed history, all 20 confusion-matrix recomputations, best selection, checkpoints, provenance, and file hashes, then failed because it assumed the default TensorBoard EventAccumulator would retain all 20000 scalar samples instead of its 10000-sample reservoir.

## Commands

- `docker exec Base64-decoded full A1 terminal audit with default EventAccumulator; read scalar tag counts`

## Changed files

- `None; both commands were read-only`

## Evidence

- `Assertion failed at the per-training-tag count; every training tag showed 10000 retained samples spanning steps 3-20000, while validation tags correctly showed 20. This matches the accumulator default cap rather than event-file loss.`

## Decisions and assumptions

- Set TensorBoard scalar size guidance to zero and repeat the complete audit without weakening any artifact assertion.

## Next action

Rerun the full terminal audit with EventAccumulator size_guidance SCALARS=0.
