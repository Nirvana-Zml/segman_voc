# VOC2012 semantic-segmentation repository guidance

## Mandatory workflow

- Use `$voc-segmentation-project-driver` when a request asks to assess,
  continue, finish, train, evaluate, infer, document, or package this course
  project.
- Every logical project step must update documentation under
  `voc_segmentation/project_docs/`. Record successful, failed, blocked, and
  interrupted steps with commands and artifact paths.
- Treat `VOC2012_Semantic_Segmentation_Project_Plan.md` as the current
  experiment plan. Older SegMAN plans and artifacts are legacy evidence only.

## Project shape

- `voc_segmentation/` is the standalone PyTorch application root. Container
  commands run from `/workspace/voc_segmentation`.
- Host datasets remain under `data/` and are mounted read-only at `/data`.
- Generated data audits, runs, checkpoints, outputs, and predictions stay under
  `voc_segmentation/` in gitignored directories.
- Read `voc_segmentation/README.md` for commands and
  `voc_segmentation/project_docs/STATUS.md` for current evidence.
- The deleted `SegMAN-main/` tree is preserved only in Git history at commit
  `111f8a0`; do not restore it unless the user explicitly requests recovery.

## Non-negotiable invariants

- Pascal VOC masks have 21 classes (`0..20`), ignore label `255`, and no label
  reduction. Resize masks only with nearest-neighbor behavior.
- Keep official train and validation splits independent during development and
  model selection. Never tune on test400.
- The current plan states that ImageNet classification pretraining, SBD,
  validation-proven TTA, and validation-proven ensembles are allowed. Record
  source, scope, license where relevant, and SHA-256 for every acquired data or
  weight artifact before it enters a reportable run.
- Prefer the 8 GB-safe MiT-B1/384 configuration until a saved memory profile
  proves MiT-B2/512 is viable. This resource adaptation does not change the
  planned GCB-Seg architecture or its required ablations.
- Never invent metrics, curves, hardware results, or report content. Derive
  every claim from saved configs, logs, checkpoints, and outputs.
- Final test predictions must be exactly 400 single-channel `uint8` PNG files,
  preserve source dimensions and names, and contain only class IDs `0..20`.
- Smoke and overfit checkpoints are diagnostic only and never submission
  weights.

## Verification

- Run the cheapest relevant check first.
- Run `python -m unittest discover -s tests -v` inside
  `/workspace/voc_segmentation` after changes to configs, datasets, metrics,
  model interfaces, losses, or prediction tooling.
- A Dockerfile change is not verified until a clean image build and the
  container environment check pass.
- Before handoff, update the applicable milestone and `STATUS.md`, then create
  a step record with the driver skill's `record_step.py` helper.
