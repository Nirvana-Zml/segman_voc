# Documentation protocol

Read this reference before changing project state or starting a runtime command.

## Package ownership

`SegMAN-main/project_docs/` is the versioned project ledger:

- `STATUS.md`: current state, latest evidence, next action, and blockers.
- `INDEX.md`: ordered index of immutable step records.
- `milestones/`: one maintained document per completion stage.
- `steps/`: one immutable document per performed logical step.
- `decisions/`: external constraints and decisions with provenance.
- `templates/`: the required shape for step documents.

Large generated logs, checkpoints, TensorBoard events, masks, and archives stay
in their existing runtime directories. Documentation links to them; it does not
copy binary artifacts into `project_docs/`.

## What counts as one step

A step is one bounded action with one observable outcome, for example:

- inspect current status;
- change one coherent set of files;
- run one focused test command;
- launch a formal training job;
- observe a job's final exit and artifacts;
- evaluate one checkpoint;
- validate one prediction directory;
- make one project decision.

Do not create a document for every shell subcommand when several subcommands
form one atomic check. Do not merge unrelated implementation, training, and
evaluation work into one vague record.

## Required content

Each step record must state:

- Asia/Shanghai timestamp;
- milestone and status (`passed`, `failed`, `blocked`, `in-progress`, or
  `information`);
- objective and concise outcome;
- exact commands, if any;
- changed files, if any;
- evidence paths and key observed values;
- decisions/assumptions;
- next action.

Use repository-relative paths where possible. Preserve exact metrics and exit
codes. Mark unavailable facts as unknown rather than guessing.

## Recording mechanics

After a logical step, run:

```powershell
python .agents/skills/segman-project-driver/scripts/record_step.py `
  --milestone 05-formal-training `
  --title "Launch SegMAN-S baseline" `
  --status in-progress `
  --summary "Started the random-initialized baseline in segman_v2." `
  --command "./SegMAN-main/scripts/train_voc_s.ps1" `
  --evidence "SegMAN-main/work_dirs/segman_s_voc12" `
  --next "Monitor the process and record its final exit separately."
```

The helper creates a numbered file in `steps/` and appends it to `INDEX.md`.
Use `--changed`, `--evidence`, `--command`, and `--decision` more than once when
needed. Use `--dry-run` to inspect output without modifying files.

Update the milestone document and `STATUS.md` before creating the immutable step
record so that the record can truthfully state the post-step project state.

## Corrections and failures

Never erase historical failures. If a record contains a factual mistake, add a
new correction record naming the superseded record. If a command fails, capture
the causal error and exit code, then document the later repair as a separate
step. A retry is a new step.
