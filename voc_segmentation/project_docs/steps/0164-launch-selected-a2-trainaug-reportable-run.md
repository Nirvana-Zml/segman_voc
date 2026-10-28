# Step 0164: Launch selected A2 trainaug reportable run

- Time: 2026-10-28T08:26:12+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **In progress**

## Objective and outcome

Launched the 40000-step selected-A2 trainaug job once from clean commit ad026db and verified unique finite CE-only history, exact runtime/provenance, active PIDs and GPU, with no terminal artifacts.

## Commands

- `docker exec -d voc2012-gcb sh /workspace/voc_segmentation/runs/gcb_selected_a2_trainaug.launch.sh; independent early history/process/GPU audit`

## Changed files

- `Ignored trainaug launch wrapper, live run logs/history/TensorBoard; STATUS.md; milestone 06; immutable step ledger`

## Evidence

- `wrapper SHA 675914fe...; start 2026-10-28 08:24:15 +08:00; PIDs 784/792; unique steps 1-43; latest loss 2.236667 grad 6.605145; GPU 1523 MiB at 38 percent`

## Decisions and assumptions

- Keep the reportable run active; accept no model-selection claim until complete official-val evidence exists.

## Next action

At step 1000, audit the full 1449-image validation, checkpoint, history, TensorBoard, and active state.
