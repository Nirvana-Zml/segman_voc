# SegMAN VOC2012 repository guidance

## Mandatory workflow

- Use `$segman-project-driver` when a request asks to assess, continue, finish,
  train, evaluate, infer, document, or package this course project.
- Every logical project step must produce or update documentation under
  `SegMAN-main/project_docs/`. Record successful, failed, blocked, and
  interrupted steps; commands and artifact paths are part of the record.
- For a narrow unrelated edit to the upstream SegMAN/MMSeg code, handle the
  edit directly and still preserve the project invariants below.

## Project shape

- `SegMAN-main/` is the working source tree. Run training commands from its
  `segmentation/` directory inside the `segman_v2` container.
- Host datasets live under `data/`; runtime outputs live under
  `SegMAN-main/work_dirs/` and `SegMAN-main/predictions/` and are gitignored.
- Read `SegMAN-main/README_PROJECT.md` for operational commands and
  `实现方案.md` for the full delivery contract. Treat the live code and
  generated evidence as authoritative when prose is stale.
- `SegMAN-main/project_docs/STATUS.md` is the human-readable progress ledger;
  `project_docs/INDEX.md` indexes immutable step records.

## Non-negotiable invariants

- Pascal VOC masks have 21 classes (`0..20`), ignore label `255`, and
  `reduce_zero_label=False`. Resize masks only with nearest-neighbor behavior.
- Keep the official train and validation splits independent during model
  selection. Select the final checkpoint by validation mIoU, not test400.
- Do not use extra segmentation data, segmentation-pretrained weights, or an
  ImageNet encoder checkpoint unless the course rule and provenance are known.
- Never invent metrics, curves, hardware results, or report content. Derive
  every experimental claim from saved configs, logs, checkpoints, and outputs.
- Final test predictions must be exactly 400 single-channel `uint8` PNG files,
  preserve source dimensions/names, and contain only class IDs `0..20`.
- Diagnostic smoke/overfit checkpoints are never valid submission weights.

## Verification

- Run the cheapest relevant checks first. Use the repository PowerShell
  wrappers for containerized audit, smoke, overfit, training, evaluation, and
  prediction workflows.
- Run `python -m unittest discover -s tests -v` inside `/workspace/segman`
  after changes to VOC configs, datasets, or prediction tooling.
- A passing command is evidence only for what it actually checked. Do not call
  the project complete until the driver skill's delivery gates all have saved
  evidence.
- Before handoff, update the applicable milestone document and `STATUS.md`,
  then create a step record with the driver skill's `record_step.py` helper.
