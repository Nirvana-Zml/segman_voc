---
name: voc-segmentation-project-driver
description: Drive and document this repository's standalone Pascal VOC2012 semantic-segmentation project through environment setup, data preparation, baselines, GCB-Seg implementation and ablations, training, evaluation, test400 inference, reporting, and packaging. Use for project status or any request to continue, train, evaluate, infer, document, or finish this course project.
---

# VOC2012 Semantic-Segmentation Project Driver

Work from the repository root. Treat `voc_segmentation/` as the application
root and `VOC2012_Semantic_Segmentation_Project_Plan.md` as the current design
authority.

## Start from evidence

1. Run `python .agents/skills/voc-segmentation-project-driver/scripts/project_status.py`.
2. Inspect `git status --short` and preserve user changes.
3. Read `voc_segmentation/project_docs/STATUS.md` and the applicable milestone.
4. Read only the source/configuration needed for the next incomplete gate.
5. Read [references/completion-contract.md](references/completion-contract.md)
   before deciding that a gate or the project is complete.

Trust live configs, logs, checkpoints, metrics, and validation reports over
filenames or prose. The legacy SegMAN tree exists only in Git history and must
not be restored unless the user explicitly requests recovery.

## Record every logical step

Read [references/documentation-protocol.md](references/documentation-protocol.md)
before the first mutation or runtime command in a project-driving turn.

For each bounded step, including failures and interruptions:

1. perform the action and collect its exact result;
2. update the applicable milestone;
3. update `STATUS.md` when state or next action changes;
4. run `scripts/record_step.py`;
5. verify the new record appears in `project_docs/INDEX.md`.

Record long-running launch and terminal exit separately. Never mark a gate
passed before its command and required artifact checks finish.

## Drive the next gate

Use this order unless the user requests a narrower task:

1. reproducible Docker environment and source contracts;
2. VOC data audit and SBD acquisition/conversion with provenance;
3. DeepLabV3+ pipeline baseline;
4. SegFormer baseline;
5. GCB-Seg forward/loss tests, smoke, and two-sample overfit;
6. controlled GCM/GFF/BAB/Lovasz ablations;
7. trainaug formal training and validation-based selection;
8. validation-proven TTA/ensemble and optional final all-label retraining;
9. test400 inference, strict validation, report, and package.

Use the 8 GB-safe MiT-B1/384 configuration until a saved memory profile proves
MiT-B2/512 is viable. Diagnose failures from the first causal error and rerun
the narrow gate before broader work.

## Integrity gates

- VOC labels are `0..20`, ignore is `255`, and masks resize only by nearest
  neighbor.
- Official validation remains independent during all development and model
  selection; test400 is never a labeled evaluation set.
- The current plan permits ImageNet classification pretraining, SBD, TTA, and
  ensembles. Record source, scope, relevant license, and SHA-256 for acquired
  data and weights before reportable use.
- Select checkpoints and inference enhancements only by saved validation mIoU.
- Diagnostic checkpoints are never submission weights.
- Ask for student identity only when packaging is the next unblocked gate.

## Handoff

State the highest verified gate, evidence paths, the next action, and genuine
blockers. Claim completion only when every delivery gate has saved evidence and
the documentation ledger is current.
