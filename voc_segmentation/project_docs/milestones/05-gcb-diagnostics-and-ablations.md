# Milestone 05: GCB-Seg diagnostics and ablations

- State: **In progress — all diagnostics verified; ablations pending**
- Last updated: 2026-10-26 18:48 +08:00

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
- At the final handoff snapshot, the same PIDs remained active at step 182;
  latest CE was 1.692492, GPU use was about 1,138 MiB at 4%, and no exit code
  or terminal result existed.

## A1 step-1,000 validation

- The first full official validation covered all 1,449 samples and reported
  mIoU 0.604609, foreground mIoU 0.589999, pixel accuracy 0.902396, mean pixel
  accuracy 0.724140, validation CE 0.329029, and 244,886,857 valid pixels.
- Independent audit found 1,348 unique sequential finite train records at its
  snapshot, zero AMP overflows, exactly one validation record at step 1,000,
  all 21 class IoUs defined, and no terminal result or exit code.
- Recomputing intersection, union, all per-class IoUs, mIoU, foreground mIoU,
  pixel accuracy, and mean pixel accuracy from the saved 21x21 global
  confusion matrix matched every saved value.
- `best.pt` is non-diagnostic at step 1,000, contains only finite model-state
  tensors, is 166,840,812 bytes, and has SHA-256
  `b2165bb8ba17bd20af5b1df2a90b7e4ad7aa72ba2c404f4ee64999b8ec38a0db`.
  Validation JSON SHA-256 is
  `1e78e80303725915bab05d34f364d2f73584ba521f67e68a3e9689c542b5f49f`.
- TensorBoard contained all 15 expected scalar tags with its first validation
  point at step 1,000. Wrapper/trainer PIDs 2447/2454 remained active; GPU use
  was about 1,162 MiB at 20%.

## A1 terminal result

- The wrapper saved exit code 0, `result.json` reports `ok=true` and
  `diagnostic_only=false`, no Python trainer remains, and the GPU returned to
  272 MiB / 0%.
- History contains exactly 20,025 records: 20,000 unique sequential completed
  train steps, 20 complete 1,449-image validations at every 1,000 steps, and
  five valid AMP-overflow records with scale reduction followed by completion
  of the same logical step. All completed numeric values are finite.
- Independent global-confusion recomputation matched every per-class IoU,
  mIoU, foreground mIoU, pixel accuracy, and mean pixel accuracy for all 20
  validations. The validation mIoU sequence culminated at 0.736157, 0.736626,
  and 0.736464 for steps 18,000, 19,000, and 20,000.
- Best A1 is step 19,000: mIoU 0.736626, foreground mIoU 0.726605, pixel
  accuracy 0.938822, and validation CE 0.328063. Against matched A0 mIoU
  0.733889, GCM contributes +0.002738 absolute mIoU, or +0.274 percentage
  points, under this single seed and budget.
- Terminal step 20,000 has mIoU 0.736464 and CE 0.328045. Best and last are
  non-diagnostic finite checkpoints of 166,840,812 bytes with SHA-256
  `55839b61b0be34dbcc88217528ca1cd93859fcc089164c36357a2792e221f990`
  and `c8c93dd77a7b8686c5e464ced2b326481378c53b376a644fcbbcfa0897bc4a3a`.
- The first complete audit reached the TensorBoard count assertion after all
  history, confusion, selection, checkpoint, and hash checks had passed, but
  failed because the default EventAccumulator retains only 10,000 scalar
  samples. Repeating with `size_guidance={SCALARS: 0}` verified all 20,000
  training points, 20 validation points, and five extra AMP events.
- SHA-256: result
  `1813743cb1920c69cfbc3b265b220eb9e8086181c8363362190632e7371f7e41`;
  history `05b64a5b0968ef629ff16468ca5e6377ad1c02fbd630bd4cc74d997e9369d8c6`;
  best validation `840999427e09a4f423699e34c079a9838fd1e9534fea2f4e829996ca629f7f01`;
  terminal validation
  `33a139cd1897740f1aebd9d2e991ce5ec6d5627273c018a2f7339514c6c5e58c`;
  TensorBoard event
  `0778f767cdf91f291aea46b1749ef29264381310775ce3dce1309b1524bbdf06`.
- Runtime was 34,049.165 seconds (about 9 h 27 min) and peak allocated CUDA
  memory was 828,434,432 bytes. Source provenance remained clean commit
  `c956520ca370a892fe4d63d9d181d2e0818315e9`.

## Next action

Allow resumed A3 to reach step 4,000, then audit the first complete
post-resume official validation and checkpoint selection.

## A2 formal-run preflight

- The first batched Docker preflight was incomplete because the Windows named
  pipe denied the GPU and process queries. Disk, output-absence, and hash
  checks passed, no output was created, and no process started. Approved
  serial access completed the missing checks.
- The first parsed-contract assertion used nonexistent key `val_interval`;
  the corrected read-only check used `validate_every` and passed.
- Source was clean at `670709997733f99122da0ab0063ab218087e3725`. The RTX
  4060 had no compute PID, used 285/8,188 MiB at 0%, `/workspace` had 757 GB
  free, no Python trainer was present, and the A2 output directory was absent.
- A2 configuration SHA-256 is
  `c711ec5498517081b9e6270c255986daef4061b3bea15a59c881e64ddf6ffc95`;
  MiT-B1 SHA-256 remains
  `8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b`.
- Parsed A1/A2 data, training, loss, and inference sections match. The exact
  model delta is linear-to-gated fusion plus `use_gated_fusion=false` to true;
  A2 is otherwise CE-only with no BAB or auxiliary head, official train/val,
  20,000 steps, and full validation every 1,000 steps.

## A2 reportable launch

- Preflight documentation was committed as clean source
  `7ca7d0d892fec7529fc5193b078c6c9d30cb27b7`. The ignored launch wrapper
  passed shell syntax validation and has SHA-256
  `3daacd10132943540e2571bc1f54f68a47ab6218c26688f3a9323d1b1820c5e8`.
- The background job started at 2026-10-26 19:01:56 +08:00. Wrapper/trainer
  PIDs 2784/2791 were active with the exact source commit and clean-worktree
  assertion; GPU use was 1,684 MiB at 50%.
- An independent audit at step 91 verified unique sequential finite history,
  total loss exactly equal to CE, latest loss 1.848219, A2 gated fusion, no
  BAB or auxiliary head, 20,000 reportable steps, full validation every 1,000
  steps, and no validation limit.
- No exit code or terminal result existed. The first reportable acceptance
  gate is the full 1,449-image validation at step 1,000.

## A2 terminal result

- A2 exited with code 0. No Python trainer remained and the terminal GPU
  snapshot was idle. The run retained clean source provenance
  `7ca7d0d892fec7529fc5193b078c6c9d30cb27b7`.
- Independent audit found 20,000 sequential finite train steps, 20 full
  validations, and six bounded AMP recoveries in 20,026 total history records.
  All A2 total losses equal CE, and every 1,449-image validation contains the
  same 244,886,857 valid pixels.
- Recomputing all 20 global confusion matrices reproduced every reported
  metric. Best step 18,000 has mIoU 0.755643, foreground mIoU 0.746083, pixel
  accuracy 0.945714, and CE 0.274758. It improves on A1 by 0.019017 mIoU
  (+1.902 percentage points) and A0 by 0.021755 (+2.175 points).
- Terminal step 20,000 is lower at mIoU 0.752653 and CE 0.283783, so the
  selected artifact is `best.pt`. Best/last checkpoint SHA-256 values are
  `82ef8b83f1c5e65dd47bd2383b2788b374bedef10f3747ae54b9bf884e59a624`
  and `fb72e73a5ddc2c24630b99aa4073bc506a2ed10ad02c0753bd4b981ecfdc0504`;
  all recursively inspected tensors are finite and both are non-diagnostic.
- Two complete-audit attempts were rejected for assumptions about TensorBoard
  representation: tag order is not semantic, and the two AMP tags contain six
  extra overflow events. The corrected full-retention audit verified all
  training, validation, and overflow step multiplicities and passed.
- Result/history/best-validation/terminal-validation/event SHA-256 values are
  `7abb59bbe865a164df18bfa3aca4ce85ac3705fc2d2abc4f2bc603edc465674b`,
  `3e14be84e34869161a7e96df406d993cde899b2d275604bb3d66a496c96aff37`,
  `a1402f246128314c9cfb19f8e12c8571227e434b631b6ef955afca059a315ad7`,
  `4974eb0aaf3fdf4929b6d6725756449b151f42a9f8f76f38c39f8ba7d8da7ed0`,
  and `7928d46d15ac71ec167d7159c6846a8879e80a15260a708f86f3baabf7d194f9`.
- Runtime was 34,986.868 seconds and peak allocated CUDA memory was
  1,218,130,944 bytes.

## A3 formal-run preflight

- Source was clean at `38d68e2f011291dd8e0580bc4206abbd39a81315`. The RTX
  4060 had no compute PID, used 846/8,188 MiB at 0%, `/workspace` had 757 GB
  free, no Python trainer was present, and the A3 output directory was absent.
- A3 configuration SHA-256 is
  `11127f069b01f0464df539a3974aace591d2267868867f6aaf8aee4a6393354c`;
  MiT-B1 SHA-256 remains
  `8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b`.
- Parsed A2/A3 data, training, and inference sections match. The exact model
  delta enables BAB, while the exact loss delta changes edge weight from zero
  to 0.2. Gated GFF remains enabled; Lovasz, auxiliary loss, and label
  smoothing remain zero. A3 uses official train/val for 20,000 steps with full
  validation every 1,000 steps.

## A3 reportable launch

- Preflight documentation was committed as clean source
  `c3ff587f59396890816dfa6a17fdb7a1ed7bbdfa`. The ignored launch wrapper
  passed shell syntax validation and has SHA-256
  `af20a507b75c401ef458531286c9ad010349092e74af3bedb3dd2ad02b64d36b`.
- The background job started at 2026-10-27 08:05:24 +08:00. Wrapper/trainer
  PIDs 3246/3253 were active with the exact source commit and clean-worktree
  assertion; GPU use was 2,379 MiB at 54%.
- An independent audit at step 341 verified unique sequential finite history,
  nonzero boundary loss components, and the exact `total = CE + 0.2 * edge`
  formula. The latest total/CE/edge values were
  0.927822/0.804647/0.615879.
- Resolved runtime is A3 gated GFF plus BAB, no auxiliary head or Lovasz,
  20,000 reportable steps, full validation every 1,000 steps, and no limit.
  No exit code or terminal result existed; the first acceptance gate is the
  complete step-1,000 validation.

## A3 user-requested stop

- Exact trainer PID 3253 received `SIGTERM` at the user's request. Exit code
  143 was saved, no terminal result exists, no trainer remains, and GPU use
  returned to 0%. This is a saved interruption, not a completed A3 result.
- Training had reached step 3,314. The original 3,317-record history was
  archived with SHA-256
  `02f6bb11af54494f12422b79162cb15796a1a6ff7828f2ec93c4238a3495aee1`;
  the active ledger was reconciled to 3,003 records through checkpoint step
  3,000 with SHA-256
  `0d9b03305fe8d2ecb1af8763ec9e669ac3aa542fe96bef7e90bbd9f7a390217f`.
- The recoverable non-diagnostic `best.pt` is step 3,000, has mIoU 0.477465,
  finite model tensors, and SHA-256
  `89a3a2af52c6aab49c7c7670d541e0d9fcb36851d163b0e6cccdd1af6af08f26`.
  Its validation JSON SHA-256 is
  `4ed955a0efbe4e4e2909957c46b00259e41349ce062234a1691ed9c09ac600d9`.
- The hash-verified stop manifest is
  `user_stop_20261027T094129+0800.json`, SHA-256
  `93aea8ebe4c96b6ab8983ff427fd2b2657e83b443ad9730ccb59c59942796f0c`.
  A future resume must start at step 3,001 from this checkpoint and must not
  treat the discarded post-checkpoint records as completed continuation.

## A3 recovery preflight

- A read-only `rg` query first omitted the option terminator and rejected the
  `--resume` pattern; the corrected query confirmed the supported resume and
  history-reconciliation path without changing state.
- Source was clean at `d6f2218f1b22cd1f7b96eaad9112fba770df39bd`.
  The RTX 4060 had no compute process, 0/8,188 MiB reported use and 0%
  utilization; no trainer existed and `/workspace` had 757 GB free.
- Registered hashes for the step-3,000 checkpoint, active history, stop
  manifest, A3 config, and MiT-B1 weight all matched. Exit 143 remains saved,
  no terminal result exists, and all 1,068 recursively discovered checkpoint
  tensors are finite.
- Checkpoint/resolved configuration equality passed. History contains train
  steps 1-3,000 exactly once and full validations at 1,000/2,000/3,000. The
  unchanged A3 contract is eligible to continue uniquely from step 3,001.

## A3 recovery launch

- Recovery preflight documentation was committed as clean source
  `c3638045039867aaa77addd8579b32b4d532ba6d`. The original console and exit
  143 were preserved under `interruption_archive/`; the console retained SHA
  `e8e1ca010fa18366aab172b70fbd417ae2d98662839800ddc7f654aee7213619`.
- The resume wrapper passed syntax validation, has SHA-256
  `1588ffcd3ad5d57c6c4091f6c110659b3d6febcbb510489d2b92fdc89034e281`,
  and started wrapper/trainer PIDs 97/104 at 2026-10-27 12:33:55 +08:00.
- Built-in reconciliation reported 3,003 records kept and none dropped. An
  independent snapshot found unique train steps through 3,058, exact resume
  from 3,001, finite composite losses, nonzero boundary supervision, and the
  configured `CE + 0.2 * edge` formula.
- GPU use was 1,675 MiB at 11%. No new exit code or result existed; A3 remains
  active under the full 20,000-step reportable contract.
