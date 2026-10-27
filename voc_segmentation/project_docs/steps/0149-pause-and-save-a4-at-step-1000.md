# Step 0149: Pause and save A4 at step 1000

- Time: 2026-10-27T23:01:08+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Information**

## Objective and outcome

At the user's request, terminated exact A4 trainer PID 416 with SIGTERM, preserved the complete pre-pause history through attempted step 1788, reconciled active history to the verified step-1000 checkpoint, and created a hash-verified pause manifest. A4 remains incomplete and resumable.

## Commands

- `docker exec voc2012-gcb kill -TERM 416; reconcile_history_for_resume to step 1000; verify checkpoint, validation, histories, manifest, process, exit, and GPU evidence`

## Changed files

- `Ignored A4 exit code, history archive, reconciled history, and user-pause manifest; STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `PID absent; exit 143; result absent; GPU 0 percent; active history 1001 records SHA 38404298...; full 1789-record archive SHA 8d017947...; step-1000 best checkpoint mIoU 0.188642 SHA 35aa3e4d...; pause manifest SHA 22826e40....`

## Decisions and assumptions

- Treat this as a saved user pause, not a completed A4 result. Resume only from verified step-1000 best.pt and require unique continuation from step 1001.

## Next action

Await user direction; if requested, pass recovery preflight and resume A4 from the verified step-1000 checkpoint.
