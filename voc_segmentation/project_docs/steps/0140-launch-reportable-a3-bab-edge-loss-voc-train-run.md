# Step 0140: Launch reportable A3 BAB edge-loss VOC-train run

- Time: 2026-10-27T08:13:14+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **In progress**

## Objective and outcome

Launched A3 from clean preflight commit c3ff587f59396890816dfa6a17fdb7a1ed7bbdfa and independently verified active processes, formal resolved configuration, 341 sequential finite records, nonzero boundary supervision, the exact CE plus 0.2 edge formula, matching provenance, and absence of terminal artifacts.

## Commands

- `sh -n and sha256sum launch wrapper; docker exec -d launch; process, GPU, artifact, and direct-Python early-history audits`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger; ignored A3 run directory`

## Evidence

- `Wrapper SHA af20a507b75c401ef458531286c9ad010349092e74af3bedb3dd2ad02b64d36b; PIDs 3246/3253; started 2026-10-27T00:05:24+00:00; 341 records through step 341; latest total/CE/edge 0.927822/0.804647/0.615879; GPU 2379 MiB at 54 percent; no exit or result.`

## Decisions and assumptions

- Treat A3 as active, not complete; do not compare mIoU until a full official validation is independently verified.

## Next action

At step 1000, audit the complete 1449-image official validation, checkpoint, composite-loss history, process health, and terminal-artifact absence.
