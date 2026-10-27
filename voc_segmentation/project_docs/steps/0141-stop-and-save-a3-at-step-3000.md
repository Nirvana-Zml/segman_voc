# Step 0141: Stop and save A3 at step 3000

- Time: 2026-10-27T09:45:01+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Information**

## Objective and outcome

At the user's request, terminated exact A3 trainer PID 3253 with SIGTERM, preserved the complete pre-stop history through attempted step 3314, reconciled active history to the verified step-3000 checkpoint, and created a hash-verified stop manifest. A3 remains incomplete and resumable.

## Commands

- `docker exec voc2012-gcb kill -TERM 3253; reconcile_history_for_resume to step 3000; verify checkpoint, validation, histories, manifest, process, exit, and GPU evidence`

## Changed files

- `Ignored A3 exit code, history archive, reconciled history, and user-stop manifest; STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `PID absent; exit 143; result absent; GPU 0 percent; active history 3003 records SHA 0d9b0330...; full 3317-record archive SHA 02f6bb11...; step-3000 best checkpoint mIoU 0.477465 SHA 89a3a2af...; stop manifest SHA 93aea8eb....`

## Decisions and assumptions

- Treat this as a saved user interruption, not a completed A3 result. Resume only from verified step-3000 best.pt and require unique continuation from step 3001.

## Next action

Await user direction; if requested, pass recovery preflight and resume A3 from the verified step-3000 checkpoint.
