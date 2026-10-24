# Step 0021: Fail Docker apt on Ubuntu mirrors

- Time: 2026-10-24T18:42:16+08:00
- Milestone: `01-environment-and-source`
- Status: **Failed**

## Objective and outcome

Docker build failed while installing git/libglib/libgl1: apt could not fetch libllvm12 from security.ubuntu.com (Connection failed after slow ~89 kB/s downloads). This is a network/mirror reachability issue, not a Dockerfile syntax error.

## Commands

- `docker build --pull -t voc2012-gcb:dev .`

## Changed files

None.

## Evidence

- `Build log: Failed to fetch http://security.ubuntu.com/.../libllvm12_...deb Connection failed [IP: 91.189.92.24 80]`

## Decisions and assumptions

- Keep the pinned pytorch/pytorch:2.1.2 base image; harden apt with a configurable China-reachable mirror and retries.

## Next action

Rewrite apt sources in the Dockerfile and rebuild.
