# Step 0052: Fail first Git-provenance-gated trainer smoke

- Time: 2026-10-25T00:08:56+08:00
- Milestone: `03-cnn-baseline`
- Status: **Failed**

## Objective and outcome

The first v4 diagnostic stopped before data loading because git rev-parse HEAD returned exit status 128 inside the container. No v4 training result or checkpoint was accepted.

## Commands

- `docker exec voc2012-gcb python tools/train_deeplabv3plus.py --diagnostic-only --output-dir artifacts/cnn_baseline/pretrained_formal_smoke_v4 --max-iterations 1 --validate-every 1 --validation-limit 2`

## Changed files

- `voc_segmentation/project_docs/milestones/03-cnn-baseline.md`

## Evidence

- `subprocess.CalledProcessError: git rev-parse HEAD returned non-zero exit status 128`

## Decisions and assumptions

- Inspect the container mount and Git safety error; preserve the requirement that unknown Git state cannot start reportable training.

## Next action

Fix container-compatible Git provenance capture, rerun tests, and rerun the diagnostic in a fresh directory.
