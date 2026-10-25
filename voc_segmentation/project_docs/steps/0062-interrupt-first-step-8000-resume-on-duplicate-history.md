# Step 0062: Interrupt first step-8000 resume on duplicate history

- Time: 2026-10-25T12:59:23+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The first resume under clean commit a4aeefe loaded the valid step-8000 checkpoint and trained through step 8041, but append mode preserved uncheckpointed records 8001-8038 from the failed segment and created duplicate step numbers. The process was intentionally terminated with SIGTERM before validation or checkpointing; last.pt is unchanged.

## Commands

- `docker exec -d voc2012-gcb sh runs/deeplabv3plus_r50_voc_train/resume_from_008000.sh`
- `failed PowerShell-expanded pgrep/kill shell one-liner`
- `docker exec voc2012-gcb pgrep -af train_deeplabv3plus.py`
- `docker exec voc2012-gcb kill -TERM 270`

## Changed files

- `voc_segmentation/project_docs/STATUS.md`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `Resume reached step 8041, then exited 143; no step-9000 validation or new checkpoint`
- `history.jsonl contains 8087 lines including duplicate train steps after 8000`
- `last.pt remains SHA-256 57fcb009202f2c9ff9c073c3e512e4e9eec356d6828447ab1a22c6c85d3c8d67`

## Decisions and assumptions

- Preserve the entire ambiguous history as an archive, but reconcile the active history to records at or before the restart checkpoint.

## Next action

Implement/test automatic resume-history archival and truncation, commit it, and resume again from step 8000.
