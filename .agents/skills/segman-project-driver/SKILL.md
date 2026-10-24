---
name: segman-project-driver
description: Drive and document this repository's Pascal VOC2012 SegMAN course project from its current evidence-backed state through implementation, training, evaluation, test400 inference, reporting, and submission packaging. Use for project status, continuing or finishing the project, running experiments, selecting checkpoints, generating predictions, or preparing final deliverables. Do not use for generic segmentation questions or unrelated upstream SegMAN edits.
---

# SegMAN Project Driver

Work from the repository root and treat `SegMAN-main/` as the application root.

## Start from evidence

1. Run `python .agents/skills/segman-project-driver/scripts/project_status.py`.
2. Inspect `git status --short` and preserve unrelated user changes.
3. Read `SegMAN-main/project_docs/STATUS.md` and the applicable milestone
   document before acting.
4. Read only the source needed for the next milestone. Use
   `SegMAN-main/README_PROJECT.md` for commands and `实现方案.md` for project
   intent. When deciding milestone completion, report content, or packaging,
   also read [references/completion-contract.md](references/completion-contract.md).
5. Trust actual configs, logs, checkpoints, metrics, and validation reports over
   filenames or claims in prose. Do not redo a passed expensive stage unless its
   inputs changed or the user requests it.

## Record every step

Documentation is part of the work, not a handoff-only summary. Read
[references/documentation-protocol.md](references/documentation-protocol.md)
before the first mutation or runtime command in a project-driving turn.

For each logical step, including a failed or blocked attempt:

1. perform the bounded action and collect its exact command/result;
2. update the applicable file under `SegMAN-main/project_docs/milestones/`;
3. update `SegMAN-main/project_docs/STATUS.md` if state or next action changed;
4. create one immutable record with
   `python .agents/skills/segman-project-driver/scripts/record_step.py`;
5. verify that the new record appears in `project_docs/INDEX.md`.

Record long-running job launch and final exit as separate steps. Never mark a
step passed before the command finishes and its required artifacts are checked.
Do not delete or rewrite earlier step records to hide a failed attempt; add a
later corrective record.

## Drive the next milestone

Choose the earliest incomplete prerequisite in this order:

1. source/configuration and unit checks;
2. VOC/test400 data audit;
3. SegMAN-S one-iteration smoke run;
4. SegMAN-T two-sample overfit diagnostic;
5. formal baseline training and validation;
6. controlled improvements, only when the baseline is reproducible;
7. best-checkpoint test400 inference and strict format validation;
8. evidence-backed report and final submission package.

For an explicit "continue" or "finish" request, keep progressing through safe,
unblocked milestones rather than stopping after analysis. Use the existing
PowerShell wrappers and container environment. Monitor long-running training,
retain its config/log/checkpoint evidence, and resume from a valid checkpoint
instead of silently restarting when possible.

If a failure occurs, diagnose it from the first causal error, make the smallest
project-scoped fix, rerun the focused check, then rerun the broader gate that
failed. Do not weaken tests or completion criteria to obtain a pass.

## Decision gates

Continue with random initialization while pretrained-weight rules are unknown.
Before using any external weight or dataset, require the course rule, source,
scope, and SHA-256 provenance. Ask for student name/ID only when packaging is the
next unblocked step. Ask about time or disk limits only when they materially
affect an imminent formal run.

Do not treat test400 as an evaluation set: it has no labels. Never infer or
claim test mIoU from its predictions. Diagnostic smoke and overfit checkpoints
must not be selected for final inference.

## Handoff

At each handoff, state the highest verified milestone, concrete evidence paths,
the next action, and any decision that genuinely blocks it. Claim completion
only when every required gate in the completion contract has evidence and the
documentation ledger is current.
