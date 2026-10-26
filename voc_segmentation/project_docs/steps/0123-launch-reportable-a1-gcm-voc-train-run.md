# Step 0123: Launch reportable A1 GCM VOC-train run

- Time: 2026-10-26T09:40:35+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **In progress**

## Objective and outcome

Passed the A1 launch-wrapper checks and started the 20,000-step reportable GCM experiment in the background from clean commit c956520 with explicit host provenance.

## Commands

- `docker exec sh -n launch.sh; docker exec sha256sum launch.sh; docker exec -d voc2012-gcb sh runs/gcb_ablation_a1_gcm_voc_train/launch.sh`

## Changed files

- `Gitignored runs/gcb_ablation_a1_gcm_voc_train runtime files; STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `Launch wrapper SHA-256 3cd4fce521998a761c982f92ac0ebce809dd9ebf355a54748d45a67f3922fbd3; start 2026-10-26 09:38:08 +08:00; wrapper PID 2447; trainer PID 2454; source c956520ca370a892fe4d63d9d181d2e0818315e9; GPU approximately 1134 MiB and 14 percent.`

## Decisions and assumptions

- Treat A1 as in progress; accept no ablation metric until a complete 1449-image validation and selected non-diagnostic checkpoint are independently verified.

## Next action

Audit early history, then monitor the first full step-1000 validation.
