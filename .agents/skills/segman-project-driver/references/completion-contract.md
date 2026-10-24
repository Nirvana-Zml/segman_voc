# Completion contract

Use this reference when deciding what to do next, whether a milestone passed,
what may appear in the report, or whether the project is ready to submit.

## Stable project facts

- Application root: `SegMAN-main/`
- Runtime container: `segman_v2`
- Container source root: `/workspace/segman`
- Container training working directory: `/workspace/segman/segmentation`
- Host VOC root: `data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012`
- Container VOC root used by configs: `../data/VOCdevkit/VOC2012`
- Host test input: `data/test400/test400/images` (400 images)
- Preferred formal model: SegMAN-S; diagnostic/fallback model: SegMAN-T
- Expected environment: Python 3.10, PyTorch 2.1.2 + CUDA 12.1,
  torchvision 0.16.2, MMCV 1.7.2, MMSegmentation 0.30.0-compatible
  source, NATTEN 0.17.3, and the local selective-scan CUDA extension.

## Gates and sufficient evidence

| Gate | Pass evidence |
| --- | --- |
| Documentation | `project_docs/` has a current status ledger, milestone records, an index, and an immutable step record for every performed logical step. |
| Source | VOC T/S configs and audit/predict/verify tools exist; focused tests pass. |
| Data | `SegMAN-main/work_dirs/data_audit.json` has `ok: true`, train=1464, val=1449, no overlap/invalid labels, and test count=400. |
| Smoke | SegMAN-S completes a real one-iteration train step and writes a checkpoint/log/TensorBoard event. |
| Overfit | SegMAN-T completes 100 deterministic iterations on two real samples; loss falls materially and `iter_100.pth` exists. This is diagnostic evidence only. |
| Formal baseline | A non-diagnostic run saves its resolved config, logs, TensorBoard events, restorable checkpoint, and validation mIoU/mAcc/aAcc. |
| Model selection | Candidate runs use the same split/evaluator; the chosen checkpoint has the best supported validation mIoU. Any CE+Dice or TTA claim is backed by a controlled comparison. |
| Predictions | The chosen non-diagnostic checkpoint generates 400 masks and `verify_predictions.py` passes filename, count, size, mode, dtype, and value-range checks. Save its JSON report. |
| Report | Every table, curve, metric, hardware fact, and prediction example traces to saved project artifacts. The report explains data, architecture, training, validation, failure cases, and final selection. |
| Package | Real student name/ID are present; source is reproducible; the prediction ZIP contains only the 400 validated PNG masks; no caches, unrelated artifacts, or diagnostic weights are submitted. |

File presence alone is not enough when a gate requires successful contents.
Parse audit/validation JSON and training logs. Distinguish a partial or failed
run from a pass.

Every gate transition must also update its milestone document, `STATUS.md`, and
the step index. A technical artifact without its documentation record is
incomplete; a document without the referenced artifact is not evidence.

## Existing command map

Run from the repository root unless noted:

```powershell
./SegMAN-main/scripts/audit_voc.ps1
./SegMAN-main/scripts/smoke_voc_s.ps1
./SegMAN-main/scripts/overfit_voc_t.ps1
./SegMAN-main/scripts/train_voc_t.ps1
./SegMAN-main/scripts/train_voc_s.ps1
./SegMAN-main/scripts/eval_voc.ps1 -Model s -Checkpoint <container-path>
./SegMAN-main/scripts/predict_test400.ps1 -Model s -Checkpoint <container-path>
```

Inside the container, run focused tests from `/workspace/segman`:

```text
python -m unittest discover -s tests -v
```

For a durable prediction-format artifact, run the verifier with
`--output /workspace/segman/work_dirs/prediction_validation.json` after
inference if the wrapper did not save one.

## External decisions

These decisions must not be invented:

- Whether ImageNet classification pretraining is allowed.
- Whether extra segmentation data or segmentation-pretrained weights are
  allowed. Default to not using either.
- Student name and student ID for final filenames.
- A hard training-time or storage limit when it changes the experiment plan.

Unknown weight rules do not block audit, tests, diagnostics, or a
random-initialized baseline. Unknown student identity blocks only final naming
and packaging.

## Experiment integrity

- Keep validation independent for all model/parameter choices.
- Save the exact resolved config for every reportable run.
- Record seed, GPU, versions, crop size, physical/effective batch size,
  optimizer, learning-rate schedule, elapsed time, and peak memory when known.
- Label incomplete/interrupted runs as such; they may inform diagnosis but not
  final comparisons.
- Do not describe TensorBoard training loss as validation loss. The current
  validation evidence is mIoU, mAcc, and aAcc unless code is added to compute a
  true validation loss.
- Use TTA for final inference only after it improves validation mIoU under the
  same checkpoint and evaluator.
