# Step 0174: Pause selected A2 trainaug on user request

- Time: 2026-10-28T18:16:18+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Information**

## Objective and outcome

Sent one SIGTERM to the unique trainer PID 201; wrapper exited 143 after train step 13478, no trainer/result remains, and GPU returned to idle.

## Commands

- `docker exec voc2012-gcb sh -lc 'kill -TERM 201'`

## Changed files

- `Ignored live console and exit-code artifacts`

## Evidence

- `exit_code.txt=143; last logged train step 13478; GPU 0/8188 MiB at 0%; no trainer process`

## Decisions and assumptions

- Use the latest complete step-13000 checkpoint as the recovery boundary.

## Next action

Archive full history, reconcile active history to step 13000, and verify checkpoints.
