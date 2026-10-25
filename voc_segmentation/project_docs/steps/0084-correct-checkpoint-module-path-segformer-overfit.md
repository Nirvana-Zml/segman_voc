# Step 0084: Correct checkpoint module path during SegFormer overfit planning

- Time: 2026-10-26T00:19:34+08:00
- Milestone: `04-transformer-baseline`
- Status: **Failed**

## Objective and outcome

A read-only reference inspection requested nonexistent engine/checkpoint.py; repository search located the exported implementation at engine/checkpointing.py and confirmed the existing engine API could be reused.

## Commands

- `Read tests/test_project_contract.py, engine/checkpoint.py, tests/test_checkpointing.py, and README; search engine package for checkpoint functions`

## Changed files

- `project_docs/INDEX.md; project_docs/steps/0084-correct-checkpoint-module-path-segformer-overfit.md`

## Evidence

- `PowerShell ItemNotFound for engine/checkpoint.py; rg found save_checkpoint and load_checkpoint in engine/checkpointing.py and exports in engine/__init__.py.`

## Decisions and assumptions

- Reuse the existing engine exports; do not add a duplicate checkpoint module.

## Next action

Implement the provenance-gated two-sample SegFormer overfit diagnostic.
