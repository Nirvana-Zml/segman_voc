# Step 0136: Reject ordered TensorBoard tag assumption in A2 audit

- Time: 2026-10-27T07:56:11+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Failed**

## Objective and outcome

The first A2 terminal audit passed history, all 20 confusion-matrix recomputations, best selection, checkpoint tensors, provenance, and hashes before incorrectly requiring EventAccumulator tag order to match the sorted result manifest. Both contained the same 15-tag set; no training artifact changed.

## Commands

- `docker exec voc2012-gcb python ignored A2 terminal audit; compare EventAccumulator and result tag lists`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Assertion failed only at ordered tag-list equality; same_set=true for all 15 tags.`

## Decisions and assumptions

- Treat TensorBoard tag ordering as non-semantic and compare exact sets instead.

## Next action

Rerun the complete audit with set equality while retaining all scalar samples.
