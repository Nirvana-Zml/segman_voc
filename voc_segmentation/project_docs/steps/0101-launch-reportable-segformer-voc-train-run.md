# Step 0101: Launch reportable SegFormer VOC-train run

- Time: 2026-10-26T00:35:33+08:00
- Milestone: `04-transformer-baseline`
- Status: **In progress**

## Objective and outcome

Launched the 20000-step reportable MiT-B1 SegFormer baseline from clean commit d1f25d7 and verified an active process plus unique finite early training records. No validation or terminal result is accepted yet.

## Commands

- `docker exec -d voc2012-gcb sh /workspace/voc_segmentation/runs/segformer_b1_voc_train/launch.sh; docker exec voc2012-gcb [process, JSONL, log and GPU observations]`

## Changed files

- `gitignored runs/segformer_b1_voc_train runtime files; STATUS.md; milestones/04-transformer-baseline.md; project step ledger`

## Evidence

- `Start 2026-10-26 00:33:53 +08:00; PID 1248; source commit d1f25d71c7f8e3fc3b0407e7cd8fa252e2d3a839; 17 unique finite train records through step 17; no validation, exit code, or result; GPU about 1067 MiB.`

## Decisions and assumptions

- Treat the run as in progress and accept no baseline metric until full official validation completes.

## Next action

Monitor the complete step-1000 official validation and verify its JSON plus selected checkpoint.
