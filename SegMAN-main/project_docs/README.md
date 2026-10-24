# Project documentation package

This directory is the auditable development record for the Pascal VOC2012
SegMAN course project. It records what was attempted, what actually happened,
which artifacts support each claim, and what should happen next.

## Structure

```text
project_docs/
├── README.md                 Documentation rules and package layout
├── STATUS.md                 Current project state and next action
├── INDEX.md                  Ordered index of immutable step records
├── milestones/              Maintained stage-level records (00–08)
├── steps/                   One immutable Markdown file per logical step
├── decisions/               External rules and project decisions
└── templates/               Required step-document structure
```

Runtime artifacts remain in `work_dirs/` and `predictions/`. This package links
to them with repository-relative paths instead of duplicating checkpoints,
events, images, or archives.

## Recording rule

Every logical project action must be documented, including failures, blocked
attempts, retries, long-job launches, and long-job completions. Update the
applicable milestone file and `STATUS.md`, then create an immutable step record
using the project agent's `record_step.py` helper. Historical step records are
append-only; corrections are new records.

No metric, completion claim, or report statement is valid merely because it is
written here. Each claim must reference inspectable source, command output,
logs, checkpoints, validation JSON, or generated deliverables.
