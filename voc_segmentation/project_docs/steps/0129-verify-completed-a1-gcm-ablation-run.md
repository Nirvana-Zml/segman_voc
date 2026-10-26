# Step 0129: Verify completed A1 GCM ablation run

- Time: 2026-10-26T18:45:44+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Independently verified A1 terminal exit, all 20000 train steps, 20 full validations, five recovered AMP overflows, global-confusion metrics, validation-selected best checkpoint, terminal checkpoint, TensorBoard series, source/data/weight provenance, and artifact hashes.

## Commands

- `docker exec ps and nvidia-smi; docker exec Base64-decoded full A1 audit with TensorBoard SCALARS=0; compare A0 and A1 result JSON`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Exit 0; best step 19000 mIoU 0.736626 versus A0 0.733889, delta +0.002738; terminal mIoU 0.736464; 20025 history records; five AMP recoveries; best/last SHA 55839b61... / c8c93dd7...; result SHA 1813743c...; elapsed 34049.165 seconds; peak CUDA 828434432 bytes.`

## Decisions and assumptions

- Accept A1 as the completed GCM ablation result for seed 42 and the common 20k budget; use best.pt at step 19000, not last.pt. Do not generalize statistical significance from one seed.

## Next action

Commit the terminal ledger, then pass A2 gated-GFF formal preflight from the new clean revision.
