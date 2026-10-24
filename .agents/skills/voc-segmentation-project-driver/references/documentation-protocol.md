# Documentation protocol

Read this reference before changing project state or starting a runtime command.

## Ledger ownership

`voc_segmentation/project_docs/` is the versioned evidence ledger:

- `STATUS.md`: current state, evidence, next action, and blockers;
- `INDEX.md`: active immutable step index and legacy-ledger pointer;
- `milestones/`: maintained summaries for delivery gates;
- `steps/`: immutable records created after the standalone-project reset;
- `decisions/`: external rules and architectural decisions.

Runtime logs, checkpoints, TensorBoard events, converted datasets, masks, and
archives stay in their gitignored runtime directories. Documentation links to
them without copying binaries.

## One step

A step is one bounded action with one observable result: a source change, test,
environment build, data audit, job launch, job exit, evaluation, prediction
validation, or project decision. Do not merge unrelated implementation,
training, and evaluation into one record.

Each record must include the Asia/Shanghai time, milestone, status, objective,
outcome, exact commands, changed files, evidence, decisions, and next action.
Unavailable facts stay explicitly unknown.

After updating the milestone and `STATUS.md`, record the step:

```powershell
python .agents/skills/voc-segmentation-project-driver/scripts/record_step.py `
  --milestone 01-environment-and-source `
  --title "Verify standalone container" `
  --status passed `
  --summary "Built the pinned image and passed the CUDA environment check." `
  --command "docker build --pull -t voc2012-gcb:dev ." `
  --evidence "Docker image voc2012-gcb:dev" `
  --next "Implement and test the official VOC data loader."
```

Never erase historical failures. Add a later corrective record. The legacy
steps 0001–0017 remain available in Git commit `111f8a0` under
`SegMAN-main/project_docs/`; active numbering continues at 0018.
