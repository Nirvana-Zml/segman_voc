# Step 0001: Create the repository project driver

- Time: 2026-10-24T16:03:51+08:00
- Milestone: `00-agent-and-documentation`
- Status: **Passed**

## Objective and outcome

Created a repository-scoped Codex skill that assesses existing evidence and
drives the SegMAN VOC2012 project through training, validation, prediction,
reporting, and packaging. Added root repository guidance so relevant requests
route to the skill.

## Commands

- `python .../init_skill.py segman-project-driver --path .agents/skills --resources scripts,references ...`
- `python .../quick_validate.py .agents/skills/segman-project-driver`
- `python -B .agents/skills/segman-project-driver/scripts/project_status.py`

## Changed files

- `AGENTS.md`
- `.agents/skills/segman-project-driver/SKILL.md`
- `.agents/skills/segman-project-driver/agents/openai.yaml`
- `.agents/skills/segman-project-driver/references/completion-contract.md`
- `.agents/skills/segman-project-driver/scripts/project_status.py`

## Evidence

- Skill validator result: `Skill is valid!`
- Status scanner result: highest verified stage `segman_t_overfit`; next stage
  `formal_training_and_validation`.

## Decisions and assumptions

- Use a repository-local skill so the workflow travels with this project.
- Keep implicit invocation enabled and also support explicit
  `$segman-project-driver` invocation.

## Next action

Add a mandatory, structured documentation ledger so every future logical step
has an auditable Markdown record.
