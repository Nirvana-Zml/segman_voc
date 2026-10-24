# Project evidence ledger

This directory records the standalone VOC2012 project's current state and every
bounded project step. Runtime binaries remain in gitignored directories; this
ledger links to them and records exact commands and observed results.

The legacy SegMAN ledger for steps 0001–0017 is preserved in Git commit
`111f8a0` at `SegMAN-main/project_docs/`. Those records remain historical
evidence but do not prove any gate for the new standalone model.

- `STATUS.md`: current human-readable state and next action
- `INDEX.md`: append-only active step index
- `decisions/`: external constraints and architecture decisions
- `milestones/`: maintained delivery-gate summaries
- `steps/`: immutable standalone-project records
- `templates/`: step-record format
