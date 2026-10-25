# Step 0069: Pause and save formal CNN run at step 18000

- Time: 2026-10-25T19:48:26+08:00
- Milestone: `03-cnn-baseline`
- Status: **Information**

## Objective and outcome

At the user's request, terminated PID 66 after attempted step 18098, preserved the complete pre-pause history, reconciled active history to the verified step-18000 checkpoint, and wrote a hash-verified pause manifest. The run remains incomplete and resumable.

## Commands

- `docker exec voc2012-gcb kill -TERM 66; reconcile_history_for_resume(... start_step=18000 ...); PowerShell SHA-256 and manifest verification`

## Changed files

- `voc_segmentation/runs/deeplabv3plus_r50_voc_train/user_pause_20261025T194635+0800.json (gitignored); history archive and active history (gitignored); voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `PID absent; exit_code.txt=143; result.json absent; GPU 0 MiB/0%; last.pt step 18000 mIoU 0.499866 SHA-256 0af35cef20f83c5f0ac6eebde113f778edfc574befd5933adb3cf553e7127b67; best.pt step 9000 mIoU 0.562094 SHA-256 07e2364f7b34490f1d42515580dd73c9c54db79143967db943e5de0f1dca311f; active history 18023 records with 0 duplicate train steps; archive 18121 records; all 7 manifest hashes matched`

## Decisions and assumptions

- Treat this as a saved user pause, not terminal success. Preserve 98 uncheckpointed train records only in the full archive and resume later from step 18000. A read-only source inspection first queried nonexistent training/checkpointing.py, then correctly located engine/training.py; runtime state was unaffected.

## Next action

Await user direction; resume only from verified step-18000 last.pt and train through terminal completion.
