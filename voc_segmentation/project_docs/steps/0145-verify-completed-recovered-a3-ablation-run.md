# Step 0145: Verify completed recovered A3 ablation run

- Time: 2026-10-27T21:50:10+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Independently verified recovered A3 terminal exit, unique train steps 1-20000, 20 complete validations, six recovered AMP overflows, composite-loss formulas, all global-confusion metrics, checkpoint selection, two TensorBoard sessions, interruption evidence, provenance, and artifact hashes.

## Commands

- `docker exec process and GPU checks; inspect result and checkpoint roles; full A3 recovery-aware terminal audit with SCALARS=0; compare A0-A3 results`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Exit 0; best and terminal step 20000 mIoU 0.726746; delta versus A2 -0.028897; 20026 history records; six AMP recoveries; best SHA 41ab4739...; result SHA 85a691f8...; resumed elapsed 30096.928 seconds; peak CUDA 1350610432 bytes.`

## Decisions and assumptions

- Accept A3 as the completed BAB plus edge-loss result for seed 42 and the common 20k budget. A3 underperforms A2, so A2 remains best. Select best.pt at step 20000; do not use stale step-16000 last.pt.

## Next action

Commit the terminal ledger, then pass A4 Lovasz formal preflight from the new clean revision.
