# Step 0115: Implement matched GCB ablation trainer and contracts

- Time: 2026-10-26T09:17:03+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Pinned a fair 20,000-step A0-A4 VOC-train budget, implemented explicit linear and gated GCB fusion modes, added cumulative A1-A4 configurations, and extended the proven formal trainer to record mapped composite losses without duplicating its validation, checkpoint, resume, provenance, or AMP contracts.

## Commands

- `docker exec voc2012-gcb python -m py_compile tools/train_segformer.py tools/train_gcb_seg.py models/gcb_seg.py; docker exec voc2012-gcb python -m unittest tests.test_gcb_seg tests.test_gcb_losses tests.test_gcb_trainer tests.test_segformer_trainer tests.test_config_contract -v; docker exec voc2012-gcb python -m unittest discover -s tests -v`

## Changed files

- `GCBSeg fusion mode; shared trainer and GCB wrapper; A1-A4 configs; model/trainer/config tests; README; decision 0004; STATUS and milestone 05`

## Evidence

- `Focused tests passed 21/21 in 2.362 seconds; full target-container suite passed 48/48 in 5.962 seconds; config tests prove common A0 budget fields and cumulative factor switches.`

## Decisions and assumptions

- Use existing SegFormer 20k run as A0. A1 adds GCM, A2 gated GFF, A3 BAB/edge, A4 Lovasz; auxiliary and label smoothing stay off to prevent confounding.

## Next action

Run and independently audit a one-step effective-batch-8 A4 CUDA diagnostic with a two-image non-reportable validation limit.
