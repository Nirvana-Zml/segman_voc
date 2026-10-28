# Step 0178: Correct Docker Desktop launch path

- Time: 2026-10-28T18:45:40+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Failed**

## Objective and outcome

The first Docker Desktop start request used a nonexistent system-wide path; registry inspection found the actual per-user installation.

## Commands

- `Start-Process 'C:\Program Files\Docker\Docker\Docker Desktop.exe' -WindowStyle Hidden`

## Changed files

- `None`

## Evidence

- `System-wide executable absent; uninstall registry InstallLocation=C:\Users\Nirvanaml\AppData\Local\Programs\DockerDesktop`

## Decisions and assumptions

- Use the exact registered per-user Docker Desktop executable.

## Next action

Start Docker Desktop from the registered location and wait for the engine.
