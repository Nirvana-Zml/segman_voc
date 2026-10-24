# Project status

- Last updated: 2026-10-24 17:20 +08:00
- Overall state: **In progress**
- Highest evidence-backed milestone: **04 — SegMAN-T overfit diagnostic**
- Next milestone: **05 — Replacement formal baseline**
- Preferred formal model: self-built lightweight encoder-decoder; exact architecture pending a written model contract
- Safe default while pretraining rules are unknown: random initialization

## Milestone summary

| ID | Milestone | State | Primary evidence |
| --- | --- | --- | --- |
| 00 | Agent and documentation | Complete | `AGENTS.md`, `.agents/skills/segman-project-driver/`, `project_docs/` |
| 01 | Source and focused tests | Reopened for replacement model | Existing VOC tooling remains valid; self-built model source and contract are not yet implemented |
| 02 | Data audit | Passed | `work_dirs/data_audit.json` |
| 03 | SegMAN-S smoke | Passed | `work_dirs/smoke_s/iter_1.pth` and logs/events |
| 04 | SegMAN-T overfit | Passed (diagnostic only) | `work_dirs/overfit_t_complete/iter_100.pth`; loss 2.97378 → 0.15274 |
| 05 | Formal training | Stopped; not eligible | SegMAN-S stopped at iteration 4350 after five background-only validations; `iter_3660.pth` retained |
| 06 | Model selection | Not started | Requires comparable validation experiments |
| 07 | Test400 inference | Not started | No validated 400-mask prediction directory found |
| 08 | Report and package | Not started | Requires final metrics, predictions, student name, and student ID |

## Current blockers and decisions

- ImageNet classification pretraining permission is unknown. This does not
  block a random-initialized baseline.
- The SegMAN architecture is retired as the active model candidate. Its source,
  stopped-run artifacts, and project tooling remain preserved as diagnostic
  evidence; this decision does not require deleting the `SegMAN-main/` tree.
- Extra segmentation data and segmentation-pretrained weights default to
  disallowed unless the course rule explicitly permits them.
- Student name and ID are unknown; they block only final naming/packaging.
- A hard training-time or disk limit has not been provided; ask only when it
  would change the imminent experiment plan.

## Next action

Freeze a one-page contract for the self-built semantic-segmentation baseline
before implementation: architecture blocks and tensor shapes, 21-class output,
ignore-label handling, normalization/initialization, parameter and VRAM target,
loss, and smoke/overfit acceptance checks. Then implement only that minimal
model and reconnect the existing VOC audit, validation, and prediction tooling.

## Latest repository check

All current tracked and untracked project changes were reviewed for a local Git
snapshot. The pre-commit whitespace check passed, no candidate file exceeded
11 KiB, and the targeted private-key/token signature scan found no match.
