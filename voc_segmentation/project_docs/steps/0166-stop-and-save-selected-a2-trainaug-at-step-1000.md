# Step 0166: Stop and save selected A2 trainaug at step 1000

- Time: 2026-10-28T08:58:19+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Information**

## Objective and outcome

At user request, stopped exact trainer PID 792, preserved full history through train step 1066, reconciled active history to the verified step-1000 checkpoint, and wrote a hash-verified stop manifest.

## Commands

- `docker exec voc2012-gcb kill -TERM 792; reconcile_history_for_resume at step 1000; verify checkpoint, validation, histories, manifest, process, exit and GPU evidence`

## Changed files

- `Ignored exit code, full history archive, reconciled history and stop manifest; STATUS.md; milestone 06; immutable step ledger`

## Evidence

- `exit 143; best.pt step 1000 mIoU 0.154537 SHA 8005ad74...; active history 1001 records SHA abb2bc8c...; full 1067-record archive SHA ced17718...; manifest SHA cf208b9a...`

## Decisions and assumptions

- Treat this as an incomplete saved user stop; resume only from best.pt step 1000 and require unique continuation from 1001.

## Next action

Await user direction; if continuation is requested, pass recovery preflight and preserve stop evidence before launch.
