# Step 0066: Stop and save formal CNN run at step 10000

- Time: 2026-10-25T14:33:28+08:00
- Milestone: `03-cnn-baseline`
- Status: **Information**

## Objective and outcome

At the user's request, terminated exact PID 422 with SIGTERM after attempted step 10047. Step 10000 had already completed full validation and checkpointing. Preserved all 10058 pre-stop history records in a hash-verified archive, reconciled the active history to the saved checkpoint, verified best/last metadata and hashes, and wrote a machine-readable stop manifest. The run remains incomplete and resumable.

## Commands

- `docker exec voc2012-gcb kill -TERM 422`
- `docker exec voc2012-gcb python -c <reconcile active history to step 10000 and archive full history>`
- `docker exec voc2012-gcb python -c <best/last checkpoint metadata verification>`
- `PowerShell SHA-256 verification for manifest artifacts`

## Changed files

- `voc_segmentation/runs/deeplabv3plus_r50_voc_train/user_stop_20261025T142718+0800.json (gitignored runtime manifest)`
- `voc_segmentation/project_docs/STATUS.md`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `PID 422 absent; exit_code.txt=143; result.json absent`
- `Active history 10011 records through step 10000, SHA-256 e50f3344fdf55ca8b28da384b1996138c4ee6cb0bf1559393ba311294735aaa5`
- `Full 10058-record archive SHA-256 0ff6ae540aa0312792644a4356fdf73125309f099aa6701e4d1d3a82cd47beec`
- `Best step 9000 mIoU 0.5620939223717769 SHA-256 07e2364f7b34490f1d42515580dd73c9c54db79143967db943e5de0f1dca311f`
- `Last step 10000 mIoU 0.5403878709449588 SHA-256 1e46f3bce6fa88eaec822ae804c88793e189cf14adf939fe0ceb62a543edde81`
- `All seven hashes in user_stop_20261025T142718+0800.json independently matched`

## Decisions and assumptions

- Treat the user-requested stop as a saved interruption, not a successful terminal run or passed CNN gate.

## Next action

Await user direction; resume only from the verified step-10000 last.pt if continuation is requested.
