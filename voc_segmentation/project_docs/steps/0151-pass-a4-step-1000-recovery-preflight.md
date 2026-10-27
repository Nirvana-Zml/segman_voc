# Step 0151: Pass A4 step-1000 recovery preflight

- Time: 2026-10-27T23:42:46+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Verified the paused A4 checkpoint, configuration, histories, hashes, finite tensors and loss formula; confirmed no competing trainer and sufficient GPU/disk resources. A4 is eligible to continue uniquely from step 1001.

## Commands

- `docker exec voc2012-gcb python .test_tmp/a4_recovery_audit.py; docker exec voc2012-gcb ps/nvidia-smi/df inspections`

## Changed files

- `STATUS.md; milestone 05; immutable step ledger`

## Evidence

- `best.pt step 1000 mIoU 0.188642 SHA 35aa3e4d...; 1068 finite tensors; history 1001 records SHA 38404298...; no trainer; GPU 120/8188 MiB at 0 percent; 757 GB free`

## Decisions and assumptions

- Resume only from verified best.pt at step 1000 after committing this clean preflight ledger.

## Next action

Commit the recovery preflight documentation, archive paused console/exit, and launch A4 resume.
