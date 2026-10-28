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

Await user direction. Before any continuation, verify the saved step-1,000
checkpoint, manifest, histories, source/config hashes, idle GPU, and absent
trainer; then resume uniquely at step 1,001 while preserving stop evidence.

## One-step CUDA diagnostic

- A first ordinary Docker resource query was denied by the Windows named pipe;
  approved retry showed no trainer, GPU 113/8,188 MiB at 0%, no compute process,
  757 GB free, and absent diagnostic/formal output directories.
- The real-trainaug diagnostic passed one effective-batch-8 optimizer step:
  total=CE=3.193626, gradient norm 8.631807, encoder/decoder LR
  0.00006/0.0006, and peak allocated CUDA memory 1,100,638,720 bytes. It
  recorded 10,582 train samples and all locked SBD/VOC provenance fields.
- Its two-image validation is explicitly non-reportable. Independent audit
  verified exact CE-only composition, one unique train record, one validation,
  TensorBoard points, and a diagnostic-only step-1 checkpoint with 1,036 finite
  tensors. Result/history/checkpoint/event SHA-256 values are `0f75f5f2...`,
  `5ae71a43...`, `c6ede983...`, and `8d8015f1...`.

## Formal-run preflight

- An ordinary final-preflight Docker batch was denied by the named pipe; the
  approved retry passed every check. Clean source commit is
  `2420f7890b2ced7bb1d4d47d95701d8b72ee1e57`.
- No trainer or GPU compute process exists; the RTX 4060 reports 113/8,188 MiB
  at 0%, `/workspace` has 757 GB free, and
  `runs/gcb_selected_a2_trainaug` is absent.
- Parsed source and runtime evidence match: selected A2 mIoU 0.755643; config
  SHA `5fcbfa7c...`; selection summary `0011ddfd...`; manifest `7fa9f8f0...`;
  SBD archive `6a5a2918...`; MiT-B1 weight `8944b871...`; official val split
  `cdc1326d...`; trainaug count 10,582 and val overlap zero. The reportable
  contract is eligible to launch for 40,000 steps with validation every 1,000.

## Reportable trainaug launch

- The ordinary wrapper-validation and first observation calls were each denied
  once by the Windows Docker named pipe; approved retries passed, and the
  background job was launched exactly once.
- Wrapper SHA-256 is
  `675914feb7689aa2165fa519da8d03c0733e0dfee641b8a5d0b284a746f75c5b`.
  It started at 2026-10-28 08:24:15 +08:00 under clean source
  `ad026dba623eae384046aa7be5abdb6b341252e5`; wrapper/trainer PIDs are
  784/792.
- An independent step-43 audit verified unique sequential steps 1-43, no AMP
  overflow, finite loss/gradients/LRs, exact total=CE, the locked trainaug
  manifest, selected A2 model/loss, 40,000-step runtime, full val every 1,000,
  and no exit/result artifacts. Latest loss/gradient were 2.236667/6.605145.
- GPU use was 1,523/8,188 MiB at 38%. A subsequent local console snapshot had
  reached step 54 with finite loss 2.086697 and gradient norm 4.833095. The
  first reportable acceptance point is the complete step-1,000 validation.

## User-requested stop at step 1,000

- Three ordinary Docker status batches were denied by the Windows named pipe
  and passed on approved retry. Exact PID 792 was sent SIGTERM once; exit code
  143 is saved, no trainer remains, no terminal result exists, and there is no
  GPU compute PID.
- The process completed train step 1,066 before termination. All 1,067 records
  were preserved in
  `history_archive/user_stop_20261028T085659+0800.jsonl` (SHA-256
  `ced177181b46a9f58218ac958b1d66ea68dd107680445e8902c474c24cba397d`).
  Active history was atomically reconciled to 1,001 records through checkpoint
  step 1,000 (SHA-256 `abb2bc8c...`), dropping 66 post-checkpoint steps only
  from the active ledger.
- Step 1,000 completed the full 1,449-image official validation: mIoU 0.154537,
  foreground mIoU 0.118020, pixel accuracy 0.809402, CE 0.715655, and
  244,886,857 valid pixels. It is an early recovery checkpoint and not an
  accepted final selection result.
- `best.pt` is non-diagnostic step 1,000 with 1,036 finite tensors and SHA-256
  `8005ad74e9c42153d4eaa938967fb951034a5e8a6dec166f3454a4463e67f98d`.
  Validation SHA-256 is `08bf2323...`; `last.pt` is absent.
- The hash-verified stop manifest
  `user_stop_20261028T085659+0800.json` has SHA-256 `cf208b9a...`. A recovery
  must use this checkpoint and continue uniquely from step 1,001.
