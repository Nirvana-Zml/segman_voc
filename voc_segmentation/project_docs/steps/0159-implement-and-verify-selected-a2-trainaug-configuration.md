# Step 0159: Implement and verify selected A2 trainaug configuration

- Time: 2026-10-28T08:15:29+08:00
- Milestone: `06-formal-training-and-selection`
- Status: **Passed**

## Objective and outcome

Added the provenance-locked A2 trainaug formal configuration and trainer support, passed 50 tests, and verified the real 10582-sample VOC/SBD dataset with zero official-val overlap.

## Commands

- `docker exec voc2012-gcb python -m unittest discover -s tests -v; docker exec voc2012-gcb python .test_tmp/formal_trainaug_preflight.py`

## Changed files

- `configs/gcb_selected_a2_trainaug.yaml; tools/train_segformer.py; trainer/config tests; README.md; STATUS.md; milestone 06; immutable step ledger`

## Evidence

- `config SHA 5fcbfa7c...; 50/50 tests in 5.058 seconds; trainaug 10582 unique samples, VOC 1464, SBD 9118, val overlap 0; representative masks valid`

## Decisions and assumptions

- Use A2 gated GFF without BAB, edge loss, or Lovasz for the initial 40000-step formal trainaug run; preserve official val for selection.

## Next action

Commit the verified source, then pass a one-step CUDA diagnostic and formal-run preflight before launch.
