# Step 0054: Implement and verify formal DeepLabV3+ trainer

- Time: 2026-10-25T00:13:54+08:00
- Milestone: `03-cnn-baseline`
- Status: **Passed**

## Objective and outcome

Implemented the formal VOC-train DeepLabV3+ training path with provenance-checked ImageNet initialization, accumulated AMP training, separate encoder/decoder learning rates, warmup-poly scheduling, native-resolution global validation, best-mIoU checkpoint selection, restart support, resolved evidence, and reportable-run provenance gates. Twenty tests and the final v4 GPU diagnostic passed; the diagnostic remains ineligible as a baseline result.

## Commands

- `docker exec voc2012-gcb python -m unittest discover -s tests -v`
- `docker exec voc2012-gcb python tools/train_deeplabv3plus.py --diagnostic-only --git-commit 7496a7a0a641e9039f4ada378afc81f3c5b38c2e --output-dir artifacts/cnn_baseline/pretrained_formal_smoke_v4 --max-iterations 1 --validate-every 1 --validation-limit 2`
- `docker exec voc2012-gcb python -c <independent v4 artifact verification>`

## Changed files

- `voc_segmentation/configs/deeplabv3plus_r50.yaml`
- `voc_segmentation/engine/evaluation.py`
- `voc_segmentation/engine/training.py`
- `voc_segmentation/engine/__init__.py`
- `voc_segmentation/tools/train_deeplabv3plus.py`
- `voc_segmentation/tests/test_evaluation.py`
- `voc_segmentation/tests/test_training.py`
- `voc_segmentation/tests/test_config_contract.py`
- `voc_segmentation/README.md`
- `voc_segmentation/project_docs/STATUS.md`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `20 in-container tests passed in 5.722 seconds`
- `artifacts/cnn_baseline/pretrained_formal_smoke_v4/result.json: ok=true, diagnostic_only=true, validation_is_full_official_split=false`
- `Step-1 checkpoint SHA-256 ee43b5f640535fef43a89d233b7b4c2a60cb9d4c582df9dee11c85436a12a9b6`
- `Independent verification matched checkpoint/event hashes, two optimizer groups, Git base commit, environment versions, and train/val split hashes`

## Decisions and assumptions

- A one-step schedule uses multiplier 1.0 so the diagnostic performs a real update.
- Reportable runs reject validation limits and require explicit host commit plus a host-clean assertion because .git is not mounted in the container.
- The two-image validation mIoU is pipeline evidence only and must not enter the baseline table.

## Next action

Commit this verified trainer, then launch the reportable VOC-train run with all 1,449 official validation images.
