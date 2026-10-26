# Milestone 05: GCB-Seg diagnostics and ablations

- State: **In progress — all diagnostics verified; ablations pending**
- Last updated: 2026-10-26 09:39 +08:00

Implement GCM, GFF, BAB, auxiliary head, and loss composition. Pass focused
interface/loss tests, smoke, and overfit, then isolate each component in
controlled validation experiments.

## Verified interface evidence

- `models/modules/gcb.py` implements the planned GCM with 1×1, three
  depthwise-separable dilation 1/3/6, and global-pooling branches. Each branch
  emits 64 channels; concatenation is reduced to 256 channels. Convolutional
  blocks use Conv, GroupNorm, GELU, and Dropout2d.
- GFF projects C1/C2/C3 to 256 channels, performs gated top-down fusion from
  F4 through F1, and applies two depthwise-separable refinements at each stage.
  Disabling its gate falls back to direct additive fusion for controlled
  ablation without changing shapes.
- BAB concatenates C1 with upsampled C2, applies two 3×3 blocks, and returns a
  64-channel boundary feature plus one boundary logit. The boundary feature is
  concatenated with F1 before the main segmentation head.
- Training mode returns full-resolution main `[B,21,H,W]`, auxiliary
  `[B,21,H,W]`, and edge `[B,1,H,W]` logits. Eval mode returns only the main
  tensor. Independent switches disable global pooling, gates, BAB, or the
  auxiliary head while preserving main output shape.
- The full MiT-B1 model has 15,841,707 trainable parameters: 13,151,424 in the
  encoder and 2,690,283 in GCB-Seg modules. The accepted MiT-B1 checkpoint
  loaded strictly. New modules contain 28 GroupNorm and zero BatchNorm layers.

## Verified loss evidence

- `losses/gcb.py` implements label-smoothed main CE, present-class-only
  Lovász-Softmax, boundary BCE plus Dice, and auxiliary CE with planned weights
  1.0/0.5/0.2/0.4.
- Boundary targets compare valid horizontal and vertical neighbors, and bridge
  only class-different valid endpoints separated by one to three consecutive
  255 pixels. They mark valid sides, dilate once with a 3×3 max-pool, keep 255
  invalid, and never create an edge from an image border or padding-only void.
- Tests prove finite composite components, exact weighting, backward-safe
  gradients for every output, ignore-255 invariance in Lovász, required-head
  enforcement, and absence of outer edges for constant masks.
- The first focused run passed model and configuration checks but three loss
  tests failed at the same Lovász gradient line: an in-place subtraction used
  overlapping slices, which PyTorch 2.1 forbids. The mathematically equivalent
  non-in-place concatenated difference fixed the causal defect.
- Corrected focused tests passed 13/13 in 2.986 seconds. The complete target
  container suite passed 40/40 in 5.282 seconds.

## CUDA smoke evidence

- `tools/smoke_gcb_seg.py` pins the accepted MiT-B1 hash and provenance, uses
  official VOC train with 384×384 crops and physical batch 2, and requires
  disjoint exhaustive AdamW encoder/decoder parameter groups.
- The first run completed but was rejected despite `ok=true`: only 31 of
  234,318 valid pixels were positive boundaries. Direct batch audit found only
  four immediate valid class transitions because VOC object contours were
  separated by 255 void bands. Keeping that artifact would have reduced edge
  training to an almost all-negative task.
- Boundary generation was corrected to cross only fully void runs up to three
  pixels between different valid endpoint classes. Tests prove that a void-
  separated class boundary is recovered while padding without a second valid
  endpoint creates zero positives. Corrected focused tests passed 8/8 in 3.045
  seconds and all 40 repository tests passed in 5.253 seconds.
- The corrected run used deterministic official-train samples `2008_005679`
  and `2008_002175`; input and all semantic/auxiliary outputs were
  `[2,*,384,384]`, with edge output `[2,1,384,384]`.
- Losses were total 4.829429, CE 2.930575, Lovász 0.887470, edge 0.843542
  (BCE 0.704252, Dice 0.982833), and auxiliary CE 3.216027. Independent
  recomputation matched `CE + 0.5 Lovász + 0.2 edge + 0.4 auxiliary`.
- The corrected target has 2,474 boundary positives over 234,318 valid pixels.
  Pre-clip gradient norm was finite at 12.056706; AMP scale remained 1,024;
  the optimizer step completed with encoder/decoder learning rates
  0.00006/0.0003 and exact 13,151,424/2,690,283 parameter coverage.
- Peak allocated CUDA memory was 1,078,124,544 bytes and elapsed time was 3.119
  seconds. `artifacts/gcb_diagnostics/smoke.json` is diagnostic-only, 2,314
  bytes, and has SHA-256
  `62ba3dfeddcddffd4dfc79a88186b84a4a8a3010b11ce702ae73c3a33de3998e`.
- Independent checks passed for all finite losses, loss weighting, gradients,
  AMP, shapes, boundary signal, weight/provenance/config hashes, optimizer
  groups, and complete parameter coverage.

## Two-sample overfit tooling

- `tools/overfit_gcb_seg.py` fixes official-train samples `2008_005679` and
  `2008_002175`, nearest-neighbor-resizes their masks to 128x128, strictly
  verifies the accepted MiT-B1 checkpoint, and defaults to 100 AMP updates.
- Every history record contains total, CE, Lovasz, boundary, boundary BCE,
  boundary Dice, and auxiliary losses plus a finite pre-clip gradient norm,
  AMP scale, and both optimizer-group learning rates.
- Initial and final composite objectives use the same evaluation RNG seed,
  making dropout and drop-path masks identical while retaining training-mode
  auxiliary and boundary heads. The pass threshold is final/initial total loss
  at most 0.5.
- The tool saves a diagnostic-only checkpoint, 100-line JSONL history,
  TensorBoard event, result JSON, artifact hashes, same-sample segmentation
  metrics, boundary counts, peak CUDA memory, and provenance metadata.
- The two new optimizer and loss-recording tests passed 2/2 in 0.025 seconds;
  the full target-container suite passed 42/42 in 6.795 seconds.

## Two-sample overfit evidence

- Command: `docker exec voc2012-gcb python tools/overfit_gcb_seg.py`.
- The pretrained 128x128, batch-2 run completed 100/100 AdamW+AMP updates in
  24.154 seconds. All seven loss series and all gradient norms were finite;
  AMP scale remained 1,024 and peak allocated CUDA memory was 467,688,960
  bytes.
- With identical stochastic masks, total composite loss fell from 4.708998 to
  0.603445, a ratio of 0.128147 versus the maximum accepted 0.5. Final CE,
  Lovasz, edge, and auxiliary terms were 0.357852, 0.083703, 0.230291, and
  0.394208.
- The fixed target retained 2,051 positive boundaries over 31,699 valid
  pixels. Same-sample pixel accuracy was 0.994826 and mIoU was 0.936919; both
  are diagnostic only and are not official-validation claims.
- `history.jsonl` contains exactly steps 1-100. The 190,445,154-byte
  checkpoint is step 100 and `diagnostic_only=true`. The 56,688-byte
  TensorBoard event contains 11 scalar tags, each with exactly 100 finite
  values.
- A first read-only audit command failed because PowerShell stripped nested
  Python quotes and caused `NameError: artifacts is not defined`; it changed
  no artifact. The corrected Base64-carried audit independently recomputed all
  100 loss formulas, verified all checkpoint model tensors finite, and matched
  every metadata and file hash.
- SHA-256: result
  `e95bcfeae48aa43c5e3bc79172cada45694efddc48c0b21d8908b615c94084fc`;
  history `86535a8bf4f8e9397f74c3fe40aff8d29b5d4d83315b480e8622eb5467e55b5a`;
  checkpoint `14f6f2ad4a82c7567bb4a96ae30be85571d2ed6ebdaf24e5437278c48319fd17`;
  TensorBoard event
  `09df89f0080c86bfdb447a978406cd940efbbbf398a01e88a23583fb08fe6dcf`.

## Controlled ablation definition and trainer

- Decision 0004 accepts the existing 20,000-step SegFormer-B1 formal run as
  A0 because A1-A4 configs exactly match its official train/val split, seed,
  crop, augmentation, batch, pretrained encoder, optimizer, learning rates,
  schedule, validation interval, and evaluator.
- A1 adds GCM while retaining SegFormer-style linear multi-scale fusion; A2
  replaces it with gated top-down GFF; A3 adds BAB and edge loss 0.2; A4 adds
  Lovasz 0.5. Auxiliary loss and label smoothing are zero throughout this
  comparison, preventing unlisted simultaneous changes.
- `GCBSeg` now supports the explicit `linear` A1 and `gated` A2-A4 fusion
  modes. Linear mode rejects BAB, while both modes preserve the train/eval
  output and full-resolution contracts.
- `tools/train_gcb_seg.py` calls the shared proven formal trainer. The engine
  now accepts either tensor or mapped model outputs, records every finite
  active loss component, and retains bounded AMP recovery, history-safe
  resume, host Git provenance, checkpoint selection, TensorBoard, and native-
  resolution global-confusion validation.
- Focused model, loss, trainer, and configuration checks passed 21/21 in 2.362
  seconds. The complete target-container suite passed 48/48 in 5.962 seconds.

## Formal-trainer CUDA diagnostic

- Command: `docker exec voc2012-gcb python tools/train_gcb_seg.py --config
  configs/gcb_ablation_a4_lovasz.yaml --output-dir
  runs/diagnostics/gcb_a4_one_step --max-iterations 1 --validate-every 1
  --validation-limit 2 --diagnostic-only`.
- One effective-batch-8 update completed with total 3.558069, CE 2.924114,
  Lovasz 0.932956, edge 0.837382, auxiliary 0, and finite pre-clip gradient
  norm 9.638385. Encoder/decoder learning rates stayed at 0.00006/0.00060,
  AMP recorded zero overflow, runtime was 13.461 seconds, and peak allocated
  CUDA memory was 1,165,862,400 bytes.
- Validation was deliberately limited to two of 1,449 official-val images;
  its mIoU 0.208359 is non-reportable and only proves evaluator integration.
- Independent audit recomputed the active A4 formula, verified the two history
  records, resolved diagnostic overrides, step-1 diagnostic checkpoint, all
  finite model tensors, 15 one-point TensorBoard series, split/weight
  metadata, and file hashes.
- SHA-256: result
  `c2cc812cc163eddb353f32462fc3c7100a0c101d5c95eee1c9df7b66a14790ef`;
  history `914ebbe02f245815831d8493c95e6f8b68f038eb7a0c4e21639cb66e06fbd5f8`;
  checkpoint `618f90ff79730461cda5c84927bdda3896606e9f6106c4d287c3ba45413ba80e`;
  event `7fd08e26b579ff4488975b971d775b74b3ef0c0e34c84cee44ad1a92491c1720`.
- Commit `f18e435` captured the verified implementation, but its preceding
  staged whitespace audit identified five extra EOF blank lines and did not
  stop the semicolon-chained commit. A formatting-only correction removed all
  five; the subsequent working-tree whitespace audit passed.

## A1 formal-run preflight

- The first parallel Docker preflight was incomplete because three queries
  were denied by the Windows Docker named pipe. No process started. The
  approved serial retry passed every non-process check, but its `pgrep` pattern
  matched the checking shell itself. A later full process-table query first
  hit the same permission denial and then succeeded with approved access.
- Final process evidence contained only container PID 1, two stale completed-
  run log tails, and the `ps` command; no Python training process existed.
- Host source was clean at full commit
  `9386a12918cb01d0deee122047e655d6e823ea40`. The RTX 4060 reported 114 of
  8,188 MiB used, 0% utilization, and no compute process. `/workspace` had
  757 GB available, and `runs/gcb_ablation_a1_gcm_voc_train` was absent.
- Container hashes matched: A1 config
  `bceb04f2d0a9ae6d35c7b65f6da0c33ead063399c9452c9043b9b49baeef4e66`;
  MiT-B1 `8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b`.
  Parsed configuration asserted A1, linear fusion, no BAB, CE-only loss,
  20,000 steps, validation every 1,000, and official train/val.

## A1 reportable launch

- The preflight ledger was committed at full clean revision
  `c956520ca370a892fe4d63d9d181d2e0818315e9`. A first read-only wrapper check
  was denied by the Docker named pipe and started nothing. Approved retry
  passed `sh -n`, matched launch-wrapper SHA-256
  `3cd4fce521998a761c982f92ac0ebce809dd9ebf355a54748d45a67f3922fbd3`,
  and started the background job at 2026-10-26 09:38:08 +08:00.
- Wrapper PID 2447 and Python trainer PID 2454 were active with explicit
  `--git-commit c956520... --git-worktree-clean`. GPU use was approximately
  1,134 MiB at 14%; no exit code or terminal result existed.
- The first nested-shell early audit failed read-only when PowerShell truncated
  its Python quotes. Direct Base64-carried container Python then verified 51
  unique sequential finite records through step 51, CE 2.916139, gradient norm
  26.750391, AMP scale 1,024, no validation, and the correct resolved formal
  runtime and source commit.

## Next action

Monitor the active A1 experiment through step 1,000 and independently verify
its first complete 1,449-image validation and non-diagnostic checkpoint.
