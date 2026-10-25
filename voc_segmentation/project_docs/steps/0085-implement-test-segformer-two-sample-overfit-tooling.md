# Step 0085: Implement and test SegFormer two-sample overfit tooling

- Time: 2026-10-26T00:19:38+08:00
- Milestone: `04-transformer-baseline`
- Status: **Passed**

## Objective and outcome

Implemented a deterministic pretrained SegFormer two-sample overfit tool with provenance enforcement, disjoint encoder/decoder AdamW groups, finite-value checks, diagnostic checkpointing, JSONL history, and TensorBoard evidence.

## Commands

- `docker exec voc2012-gcb python -m py_compile tools/overfit_segformer.py; docker exec voc2012-gcb python -m unittest discover -s tests -v`

## Changed files

- `tools/overfit_segformer.py; README.md; project_docs/STATUS.md; project_docs/milestones/04-transformer-baseline.md`

## Evidence

- `Syntax check passed; complete in-container suite passed 29 of 29 tests in 5.618 seconds.`

## Decisions and assumptions

- Use the same fixed official-train samples as the CNN gate, pretrained hash-verified MiT-B1, 128 by 128 inputs, 100 AMP steps, and a maximum accepted loss ratio of 0.5.

## Next action

Run the CUDA overfit gate and accept only finite history plus the required loss decrease.
