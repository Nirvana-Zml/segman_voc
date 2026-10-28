# Step 0165: Retry selected A2 trainaug stop inspections

- Time: 2026-10-28T08:58:19+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

Three ordinary Docker observation batches around the user stop were denied by the Windows named pipe. Approved retries identified exact PID 792, confirmed the saved step-1000 checkpoint, and verified final process/GPU/exit state; the stop signal was sent once.

## Commands

- `docker ps/checkpoint inspection; kill -TERM 792; post-stop ps/nvidia-smi/exit/history checks; approved retries`

## Changed files

- `STATUS.md; milestone 06; immutable step ledger only`

## Evidence

- `Three permission-denied batches followed by PID 792, exit 143, no trainer, no compute PID, and active-history verification`

## Decisions and assumptions

- Treat these as observation-layer failures; retain the single successful SIGTERM and saved runtime evidence.

## Next action

Archive full history, reconcile to checkpoint step 1000, and create a stop manifest.
