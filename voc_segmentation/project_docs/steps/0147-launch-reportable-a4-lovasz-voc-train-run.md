# Step 0147: Launch reportable A4 Lovasz VOC-train run

- Time: 2026-10-27T21:59:28+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **In progress**

## Objective and outcome

Launched A4 from clean preflight commit fb2b376e43a04f5f351f83b33bfcb71c8349bbfb and independently verified active processes, formal resolved configuration, 52 sequential finite records, nonzero Lovasz and boundary supervision, the exact CE plus 0.5 Lovasz plus 0.2 edge formula, matching provenance, and absence of terminal artifacts.

## Commands

- `sh -n and sha256sum launch wrapper; docker exec -d launch; process, GPU, artifact, and direct-Python early-history audits`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger; ignored A4 run directory`

## Evidence

- `Wrapper SHA b6cefef8...; PIDs 409/416; started 2026-10-27T13:56:48+00:00; 52 records through step 52; latest total/CE/Lovasz/edge 2.211768/1.633973/0.829992/0.813998; GPU 2642 MiB at 11 percent; no exit or result.`

## Decisions and assumptions

- Treat A4 as active, not complete; do not compare mIoU until a full official validation is independently verified.

## Next action

At step 1000, audit the complete 1449-image official validation, checkpoint, composite-loss history, process health, and terminal-artifact absence.
