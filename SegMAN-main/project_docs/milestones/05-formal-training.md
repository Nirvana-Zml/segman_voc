# Milestone 05: Formal training and validation

- State: **Stopped after failed diagnostic gate**
- Last updated: 2026-10-24 17:17 +08:00

## Objective

Train a reportable self-built baseline, preserve its resolved configuration and
runtime evidence, and obtain reproducible VOC2012 validation metrics.

## Required evidence

- Formal work directory distinct from smoke/overfit diagnostics
- Resolved configuration, seed, software/GPU information, and TensorBoard events
- Restorable checkpoints and the checkpoint selected by validation mIoU
- Validation mIoU, mAcc, aAcc, and per-class IoU when available
- Physical/effective batch size, crop size, elapsed time, and peak memory
- Separate step records for launch, interruption/resume, completion, and
  evaluation

## Current evidence

The 2026-10-24 16:26 +08:00 preflight passed: `segman_v2` was running with no
existing `tools/train.py` process; the RTX 4060 Laptop GPU reported 7869 MiB
free, the mounted workspace reported 102 GB free, the dedicated target
directory was absent, and all 5 focused unit tests passed. No non-diagnostic
formal checkpoint or validation record exists yet.

The first launch at 2026-10-24 16:27 +08:00 failed before iteration 1. The
DataLoader reported `No space left on device` while allocating a PyTorch shared
tensor and identified insufficient shared memory. The container has a 64 MiB
`/dev/shm`; the launcher and all trainer processes exited, and the JSON log has
zero training records. This attempt is not a formal result.

The worker-count fix changed the T/S training and validation DataLoader worker
count to 1. The first focused-test invocation used a package-style module name,
but `tests/` is not a Python package, so it failed during test discovery without
executing the configuration test. The corrective command must use discovery
with a filename pattern.

The corrected discovery command passed the configuration contract test after
both formal VOC configs and the test contract were updated to require one
training worker and one validation worker.

The corrected random-initialized SegMAN-S run started at 2026-10-24 16:33
+08:00 in `work_dirs/segman_s_voc12_random_seed42_r1` with seed 42. The hidden
host launcher PID is 46392. At handoff check iteration 550 was present with
loss 1.75146, decode accuracy 60.4916, iteration time 0.26192 seconds, and
reported GPU memory 3013 MiB. The resolved config, JSON log, and TensorBoard
event exist. This is an active run, not yet a passed formal baseline.

At 2026-10-24 16:40 +08:00 the launcher was still alive and the trainer was
actively consuming CPU/GPU resources. The latest periodic training record was
iteration 700 with loss 1.58806; the run had entered the silent full-validation
window around iteration 732, but no validation summary had been written yet.

The first validation completed at training iteration 732. Its full 1449-image
summary was mIoU 0.0349, mAcc 0.0476, and aAcc 0.7332, and it created
`best_mIoU_iter_732.pth`. These are early baseline metrics, not terminal model
selection evidence. At 2026-10-24 16:44 +08:00 the active run had continued to
iteration 1000 with training loss 1.69255 and reported memory 3013 MiB.

The second validation after training iteration 1464 repeated mIoU 0.0349,
mAcc 0.0476, and aAcc 0.7332. Background IoU was 0.7332 with background
accuracy 1.0 while every foreground class had zero IoU and accuracy, proving
the model predicted background for every validation pixel. At 2026-10-24
16:50 +08:00 training was still active at iteration 1800; the latest eight
logged losses ranged from 1.54246 to 1.81056. Since this is a random-initialized
run observed immediately after warmup, treat the repeated background-only
prediction as an early warning rather than a terminal failure.

At 2026-10-24 16:59 +08:00 the active run had reached iteration 2900 and had
completed three validations. Successive validation summaries were about 6
minutes 20 seconds apart, including the intervening training epoch. This gives
an evidence-based estimate of roughly 10--15 minutes to the epoch-5 diagnostic
result and roughly 5--6 additional hours to finish all 55 epochs if the run is
allowed to continue. The MMCV log ETA near 14 hours is inflated by validation
time being folded into its iteration-time estimator.

All five validations through epoch 5 remained background-only at mIoU 0.0349,
mAcc 0.0476, and aAcc 0.7332. At the user's explicit request, the main trainer
received SIGTERM at 2026-10-24 17:11 +08:00. The host launcher and all container
trainer/DataLoader processes exited; the final periodic training record is
iteration 4350 with loss 1.46453. `iter_3660.pth`, `best_mIoU_iter_732.pth`,
logs, resolved config, and TensorBoard events remain preserved. This
interrupted run is diagnostic evidence only and is not eligible for model
selection or test400 inference.

## Decision boundary

Use random initialization while ImageNet pretraining permission is unknown.
Never silently use a segmentation-pretrained checkpoint.

The SegMAN architecture is retired as the active candidate. Preserve the
interrupted run as diagnostic evidence, but do not use its checkpoint for model
selection or test400 inference.

## Next action

Return to milestone 01 and freeze the self-built model contract before coding.
After implementation, pass focused tests, smoke, overfit, formal validation,
and model-selection gates before test400 inference.

## Planned execution

1. Freeze the replacement model contract and tensor interfaces.
2. Implement the minimal forward path and focused shape/loss tests.
3. Pass a real one-iteration smoke run and deterministic two-sample overfit
   diagnostic with the replacement model.
4. Preflight the container, CUDA, disk, and absence of another trainer.
5. Launch the random-initialized formal run in a new dedicated work directory.
6. Preserve the resolved config, seed, logs, TensorBoard events, checkpoints,
   GPU memory, and validation metrics.
7. Select only by independent validation mIoU, then run a separate reproducible
   evaluation before test400 inference.
