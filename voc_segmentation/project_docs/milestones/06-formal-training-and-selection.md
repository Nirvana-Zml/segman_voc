# Milestone 06: Formal training and model selection

- State: **In progress — selected A2 trainaug configuration verified**
- Last updated: 2026-10-28 08:15 +08:00

Train validated candidates on leak-free trainaug, select only by official val
mIoU, and preserve resolved configs, provenance, logs, checkpoints, curves,
per-class IoU, timing, and peak memory. Validate TTA/ensemble choices on val.

## Selected architecture and formal configuration

- The completed A0-A4 table selects A2 at validation mIoU 0.755643. Formal
  trainaug therefore preserves A2 exactly: MiT-B1/384, GCM plus gated GFF, no
  BAB, auxiliary head, edge loss, Lovasz, or label smoothing.
- `configs/gcb_selected_a2_trainaug.yaml` uses the leak-free 10,582-sample
  `trainaug` manifest while retaining the untouched 1,449-image official val.
  Following the plan, the initial budget is 40,000 iterations with validation
  every 1,000, physical/effective batch 2/8, seed 42, and the existing
  warmup-poly AdamW schedule. Config SHA-256 is
  `5fcbfa7cfdc9250dc744509e4fe58caa18d8c11864b7b5a0595c6d3f6b7a114c`.
- The config locks the trainaug manifest, audit summary, and SBD provenance to
  SHA-256 values `7fa9f8f0...`, `de37b98a...`, and `5c2444d8...`. The trainer
  verifies all three plus audit success, count 10,582, zero val overlap, zero
  duplicates, invalid labels, dimension mismatches, and conversion errors
  before dataset construction. Terminal results will retain these hashes plus
  SBD source/license metadata.

## Verification

- Focused contract, trainer, and VOC dataset tests passed 13/13 in 0.965
  seconds. The complete in-container regression suite passed 50/50 in 5.058
  seconds after the trainer/config changes.
- A first ignored actual-data preflight script failed at import because it did
  not add `/workspace/voc_segmentation` to `sys.path`; this changed no runtime
  state and started no trainer. After correcting only the script, the preflight
  verified 10,582 unique IDs, source counts VOC=1,464 and SBD=9,118, official
  val overlap=0, and decoded representative VOC/SBD samples with only valid
  VOC class IDs or ignore 255.

## Next action

Commit this verified implementation, then run a clean-source one-step CUDA
diagnostic and formal preflight. Launch the 40,000-step reportable job only
after confirming no competing trainer, sufficient disk/GPU resources, exact
hashes, and an absent output directory.
