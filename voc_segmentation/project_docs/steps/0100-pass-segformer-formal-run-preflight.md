# Step 0100: Pass SegFormer formal-run preflight

- Time: 2026-10-26T00:35:28+08:00
- Milestone: `04-transformer-baseline`
- Status: **Passed**

## Objective and outcome

The approved retry verified a clean source revision, running container, idle GPU, sufficient disk, no competing trainer, absent formal output directory, and the accepted MiT-B1 hash.

## Commands

- `git status --short; git rev-parse HEAD; docker inspect voc2012-gcb; docker exec voc2012-gcb sh -lc [GPU, disk, process, output and weight checks]`

## Changed files

- `gitignored runs/segformer_b1_voc_train/launch.sh; project_docs/INDEX.md; project_docs/steps/0100-pass-segformer-formal-run-preflight.md`

## Evidence

- `Clean commit d1f25d71c7f8e3fc3b0407e7cd8fa252e2d3a839; container true; GPU 0 MiB and 0 percent; 756 GB free; no trainer; output absent; weight SHA-256 8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b.`

## Decisions and assumptions

- Launch with explicit commit and clean-worktree assertion; keep the wrapper and runtime evidence gitignored.

## Next action

Start the 20000-step reportable SegFormer run in the background and verify early records.
