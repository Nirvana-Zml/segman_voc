# Step 0072: Verify completed formal DeepLabV3+ run

- Time: 2026-10-25T23:52:38+08:00
- Milestone: `03-cnn-baseline`
- Status: **Passed**

## Objective and outcome

Independently verified terminal exit 0, result ok=true, all 20000 unique train steps, 20 full validations, bounded AMP recovery, validation-selected best checkpoint, final checkpoint, TensorBoard scalars, and ten artifact hashes. The CNN baseline gate now passes.

## Commands

- `docker exec voc2012-gcb <terminal, result, history, checkpoint, TensorBoard, and finite-value verification>; PowerShell SHA-256 verification`

## Changed files

- `voc_segmentation/runs/deeplabv3plus_r50_voc_train/formal_completion_20261025T235047+0800.json (gitignored); voc_segmentation/project_docs/STATUS.md; voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `exit=0; result ok=true diagnostic_only=false; history 20000 train + 20 validation + 6 recovered overflows with 0 missing/duplicate train steps; best step 9000 mIoU 0.5620939223717769 SHA-256 07e2364f7b34490f1d42515580dd73c9c54db79143967db943e5de0f1dca311f; last step 20000 mIoU 0.5217021924435665 SHA-256 653dcb628aa6231631b2594b6075f6c9720fc9ac0802bfb0d3bfc5172739ec38; all 10 completion-manifest hashes matched`

## Decisions and assumptions

- Pass the CNN baseline gate. Two read-only finite-value checks first used incorrect history field assumptions; corrected schema-aware checks proved all completed train/validation values finite and all six AMP backoffs valid.

## Next action

Implement and verify the SegFormer baseline with the same split, evaluator, and validation-mIoU selection contract.
