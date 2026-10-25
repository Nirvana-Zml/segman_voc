# Step 0095: Verify SegFormer reportable-run provenance guard

- Time: 2026-10-26T00:28:08+08:00
- Milestone: `04-transformer-baseline`
- Status: **Passed**

## Objective and outcome

A negative reportable-mode probe omitted the required host commit and clean-worktree assertion and was rejected before creating any output directory.

## Commands

- `docker exec voc2012-gcb python tools/train_segformer.py --output-dir artifacts/transformer_baseline/reportable_guard_probe --max-iterations 1 --validate-every 1`

## Changed files

- `project_docs/INDEX.md; project_docs/steps/0095-verify-segformer-reportable-run-provenance-guard.md`

## Evidence

- `Expected RuntimeError requires --git-commit and --git-worktree-clean; host check confirmed no artifact directory was created.`

## Decisions and assumptions

- Commit the trainer before launching; reportable runs must inject the exact clean host revision and use full validation.

## Next action

Commit the verified trainer and then launch the 20000-step official-train SegFormer baseline from that clean revision.
