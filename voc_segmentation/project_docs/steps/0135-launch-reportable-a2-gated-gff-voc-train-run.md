# Step 0135: Launch reportable A2 gated-GFF VOC-train run

- Time: 2026-10-26T19:05:05+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **In progress**

## Objective and outcome

Launched A2 from clean preflight commit 7ca7d0d892fec7529fc5193b078c6c9d30cb27b7 and independently verified active processes, the formal resolved configuration, 91 sequential finite CE-only records, matching provenance, and absence of terminal artifacts.

## Commands

- `sh -n and sha256sum launch wrapper; docker exec -d launch; process, GPU, artifact, and direct-Python early-history audits`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger; ignored A2 run directory`

## Evidence

- `Wrapper SHA 3daacd10132943540e2571bc1f54f68a47ab6218c26688f3a9323d1b1820c5e8; PIDs 2784/2791; started 2026-10-26T11:01:56+00:00; 91 records through step 91; latest loss 1.848219; GPU 1684 MiB at 50 percent; no exit or result.`

## Decisions and assumptions

- Treat the run as active, not complete; do not compare mIoU until a full official validation is independently verified.

## Next action

At step 1000, audit the complete 1449-image official validation, checkpoint, history continuity, process health, and terminal-artifact absence.
