# Step 0059: Terminate formal CNN run on unhandled AMP overflow

- Time: 2026-10-25T12:38:18+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The reportable DeepLabV3+ run exited with code 1 at attempted iteration 8,039. Loss remained finite, but AMP scale had grown from 8,192 to 16,384 at step 8,000 and a later non-finite gradient norm triggered the trainer's fatal hard gate instead of GradScaler's safe skip-and-backoff path. Eight full validations and verified step-7,000 best/step-8,000 last checkpoints remain valid intermediate evidence; no terminal result exists.

## Commands

- `Read exit_code.txt, console.log, history.jsonl, and validation/step_*.json`
- `docker exec voc2012-gcb python -c <checkpoint metadata and SHA-256 verification>`

## Changed files

- `voc_segmentation/project_docs/STATUS.md`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `exit_code.txt=1; last recorded train step=8038; result.json absent`
- `AMP scale 8192 at step 7999, 16384 at step 8000; non-finite gradient norm at attempted step 8039`
- `Best official-val mIoU 0.5550555638663238 at step 7000`
- `best.pt SHA-256 b271ebb2682450c69d21ee3cc5339c59e1d9fd2c826d3567811952ccc2dd9dea`
- `last.pt step 8000 SHA-256 57fcb009202f2c9ff9c073c3e512e4e9eec356d6828447ab1a22c6c85d3c8d67`

## Decisions and assumptions

- Reject the run as complete but retain its eight full validations and non-diagnostic checkpoints as intermediate evidence.
- Fix bounded AMP-overflow handling before any resume or restart.

## Next action

Implement and test safe AMP overflow skip/backoff, then choose documented resume from step 8000 or a clean restart.
