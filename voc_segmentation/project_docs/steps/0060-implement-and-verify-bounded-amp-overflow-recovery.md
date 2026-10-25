# Step 0060: Implement and verify bounded AMP overflow recovery

- Time: 2026-10-25T12:47:54+08:00
- Milestone: `03-cnn-baseline`
- Status: **Passed**

## Objective and outcome

Changed formal training so a non-finite gradient is recoverable only when CUDA AMP is enabled and GradScaler proves a scale backoff. The unsafe update is skipped, the event is logged, and the same logical iteration retries; bounded consecutive/total limits preserve the hard failure gate. All 21 tests and a forced-overflow CUDA diagnostic passed.

## Commands

- `docker exec voc2012-gcb python -m unittest discover -s tests -v`
- `docker exec voc2012-gcb python tools/train_deeplabv3plus.py --diagnostic-only --diagnostic-force-amp-overflow-once --output-dir artifacts/cnn_baseline/amp_overflow_recovery_smoke --max-iterations 1 --validate-every 1 --validation-limit 2`
- `docker exec voc2012-gcb python -c <independent overflow artifact verification>`

## Changed files

- `voc_segmentation/configs/deeplabv3plus_r50.yaml`
- `voc_segmentation/engine/training.py`
- `voc_segmentation/engine/__init__.py`
- `voc_segmentation/tools/train_deeplabv3plus.py`
- `voc_segmentation/tests/test_training.py`
- `voc_segmentation/tests/test_config_contract.py`
- `voc_segmentation/README.md`
- `voc_segmentation/project_docs/STATUS.md`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `21 in-container tests passed in 3.762 seconds`
- `Forced overflow: attempt 1 scale 1024 to 512 and retry; attempt 2 completed optimizer step`
- `Diagnostic result ok=true, diagnostic_only=true, amp_overflow_count=1`
- `Checkpoint SHA-256 35ea803109d352ae1a9e1f10f0162b9dbec670353078f9531e6e640071414bfc; scale 512; 188 optimizer-state entries`

## Decisions and assumptions

- Retry the same logical iteration so max_iterations still counts completed optimizer updates.
- Abort after more than four consecutive or 32 total AMP overflows, and keep all non-AMP non-finite gradients fatal.

## Next action

Commit the verified recovery and resume the formal run from the validated step-8000 last.pt checkpoint.
