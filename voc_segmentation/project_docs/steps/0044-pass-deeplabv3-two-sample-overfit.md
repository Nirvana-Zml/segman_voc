# Step 0044: Pass DeepLabV3+ two-sample overfit

- Time: 2026-10-24T23:28:05+08:00
- Milestone: `03-cnn-baseline`
- Status: **Passed**

## Objective and outcome

Completed and independently validated the deterministic two-sample CUDA overfit gate, including a diagnostic checkpoint, 100-step history, TensorBoard scalars, artifact hashes, and a clear finite loss decrease.

## Commands

- `docker exec voc2012-gcb python tools/overfit_deeplabv3plus.py`
- `Get-FileHash artifacts/cnn_baseline/overfit_two_samples/{diagnostic.pt,history.jsonl,event} -Algorithm SHA256`
- `docker exec voc2012-gcb python -c [load checkpoint and TensorBoard event validation]`

## Changed files

- `voc_segmentation/artifacts/cnn_baseline/overfit_two_samples/result.json (gitignored evidence)`
- `voc_segmentation/artifacts/cnn_baseline/overfit_two_samples/history.jsonl (gitignored evidence)`
- `voc_segmentation/artifacts/cnn_baseline/overfit_two_samples/diagnostic.pt (gitignored diagnostic checkpoint)`
- `voc_segmentation/artifacts/cnn_baseline/overfit_two_samples/tensorboard/... (gitignored evidence)`
- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`
- `voc_segmentation/project_docs/STATUS.md`

## Evidence

- `Initial/final CE=3.002532720565796/0.017126988619565964; ratio=0.005704180508094034 <= required 0.5; 100 steps.`
- `Same-two-sample pixel accuracy=0.9948263352156219; elapsed=6.455 seconds; peak allocated CUDA memory=891774464 bytes.`
- `Checkpoint bytes=484746098, step=100, diagnostic_only=true, SHA256=925a94a271ee13682f5d49477602355b210ce2cb3107838dc663b58ee5215e5b.`
- `History has 100 lines and SHA256=12d5acfc7b4d0928c49b1e77ef59f009eb7061a5c18a1d0899f55a1b03d5600b; all three TensorBoard tags have 100 points.`

## Decisions and assumptions

- Overfit evidence proves pipeline learnability only; the checkpoint is not a validation-selected or submission weight.

## Next action

Acquire provenance-recorded torchvision ResNet-50 ImageNet weights, then implement formal training and official-val best-mIoU selection.
