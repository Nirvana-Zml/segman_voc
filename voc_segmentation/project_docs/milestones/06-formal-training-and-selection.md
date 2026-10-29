# Milestone 06: Formal training and model selection

- State: **In progress — selected A2 trainaug complete; TTA/ensemble selection pending**
- Last updated: 2026-10-29 08:00 +08:00

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

Evaluate the validation-selected step-39,000 `best.pt` first with single-scale
inference and then with the plan's 0.75/1.0/1.25-scale plus horizontal-flip
logit-averaging TTA. Retain TTA or any later ensemble only if official-val
mIoU improves over the verified 0.776017 single-scale result.

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

## Step-1,000 recovery preflight

- The first three read-only checker versions were rejected without starting a
  trainer: they respectively required source and resolved YAML to be identical
  despite the expected `runtime` section, used the nonexistent summary field
  `official_val_overlap_count` instead of `train_val_overlap`, and assumed
  flat learning-rate fields instead of the saved `learning_rates` list. Each
  correction changed only the ignored checker.
- The corrected audit passed. No trainer or competing GPU process exists; the
  RTX 4060 reports 0/8,188 MiB at 0% utilization, and `/workspace` has 756 GB
  available.
- Registered hashes for the checkpoint, active and stopped histories,
  validation, resolved and source configs, stop manifest, stopped console,
  exit code, A2 selection summary, trainaug manifest/summary/provenance,
  MiT-B1 weight, and official validation split all match. Exit 143 remains
  saved; `result.json` and `last.pt` remain absent.
- `best.pt` is non-diagnostic at step 1,000, its resolved configuration matches
  the source contract plus the expected CUDA runtime block, and all 1,036
  recursively inspected tensors are finite. Active history is exactly train
  steps 1-1,000 plus validation 1,000; the 1,067-record stop archive retains
  steps 1-1,066. Recovery is eligible to continue at exactly step 1,001.

## Step-1,000 recovery launch

- Recovery preflight evidence was committed as clean source
  `80555c803dae4baa1b3e6ff5b6e4378ff20fe1ca`. The stopped console, exit 143,
  start/source markers, and original launch wrapper were preserved under
  `interruption_archive/`; console, exit, and wrapper retained SHA-256 values
  `91ece501...`, `9d9b1872...`, and `675914fe...`.
- One ordinary final Docker inspection was denied by the Windows named pipe;
  the approved retry passed and no job existed at that time. The resume wrapper
  passed `sh -n`, has SHA-256 `0fcca8f8...`, and was launched exactly once at
  2026-10-28 12:56:06 +08:00. Wrapper/trainer PIDs are 193/201.
- The trainer reported `history_reconciliation kept=1001 dropped=0
  archive=None`. Independent audit at step 1,032 found train steps 1-1,032
  exactly once, first resumed step 1,001, the original step-1,000 validation,
  and 32 finite CE-only resumed records. Archived stop hashes still match.
- GPU use was 1,547/8,188 MiB at 5%. No new exit code or terminal result exists;
  the selected-A2 40,000-step trainaug run is actively continuing.

## User-requested pause at step 13,000

- At the user's request, the unique trainer PID 201 received `SIGTERM` once.
  It completed train step 13,478 before exit; the wrapper saved exit code 143,
  no terminal `result.json` exists, no trainer remains, and the GPU returned to
  0/8,188 MiB at 0% utilization.
- The first save script archived and reconciled history successfully, then
  stopped because it expected a `step` field in the validation JSON whose step
  is encoded in its filename. It wrote no pause manifest. The corrected,
  idempotent finalizer used the saved schema and completed without repeating or
  overwriting the archive.
- All 13,494 pre-pause records are preserved at
  `history_archive/user_pause_step_013000_20261028T181049+0800.jsonl`:
  13,478 unique train steps, 13 full validations, and three AMP-overflow
  records. Its SHA-256 is `ca185957...`.
- Active history is reconciled to 13,016 records through step 13,000: 13,000
  unique train steps, validations at every 1,000 steps through 13,000, and
  three retained overflow records. Its SHA-256 is `ecb18e96...`; 478
  uncheckpointed train records were removed only from the active history.
- The latest restart artifact is non-diagnostic `last.pt` at step 13,000,
  SHA-256 `b227903d...`, with 1,036 finite tensors. Its full official-val mIoU
  is 0.671776, foreground mIoU 0.658921, pixel accuracy 0.925984, and CE loss
  0.268512 over 244,886,857 valid pixels.
- Validation-selected `best.pt` remains step 12,000 at mIoU 0.692692, SHA-256
  `fa3b628f...`, also with 1,036 finite tensors. Selection and restart roles
  remain distinct: resume from `last.pt`, while `best.pt` remains the current
  candidate for validation-based model selection.
- The independently hash-verified pause manifest is
  `user_pause_20261028T181049+0800.json`, SHA-256 `af8155e2...`. A future
  recovery must continue uniquely at step 13,001.

## Step-13,000 recovery preflight

- Two initial Docker calls, including an approved retry, failed because the
  Docker Desktop engine was not running; no container command or trainer
  started. The first attempted executable path was stale, then the installed
  location was found from the Windows uninstall registry. Docker Desktop was
  started from that exact path and server version 29.4.3 became ready.
- The complete read-only audit then passed. The container has no trainer or GPU
  compute process, the RTX 4060 reports 0/8,188 MiB at 0% utilization, and
  `/workspace` has 756 GB available.
- The pause manifest, restart/best checkpoints, active and archived histories,
  restart validation, A2 config/selection summary, trainaug manifest/summary,
  SBD provenance, MiT-B1 weight, and official validation split all retain their
  registered SHA-256 values.
- `last.pt` is non-diagnostic step 13,000 and `best.pt` is non-diagnostic step
  12,000; both contain 1,036 finite tensors and record best mIoU 0.692692 at
  step 12,000. Active history contains exactly train steps 1-13,000, all 13
  expected validations, and three bounded overflows; the pause archive retains
  all train steps through 13,478. Recovery is eligible to start at 13,001.

## Step-13,000 recovery launch

- The passed recovery preflight was committed as clean source
  `fd1f6d3096e048c53ea1c8b4f4e67080ba79e156`. The paused console, exit 143,
  resume markers, and prior step-1,000 wrapper were preserved in the
  interruption archive. Console, exit, and wrapper retained SHA-256 values
  `02a004a5...`, `9d9b1872...`, and `0fcca8f8...`.
- The new wrapper explicitly resumes from `last.pt`, passed `sh -n`, has
  SHA-256 `8c42cd24...`, and launched exactly once at 2026-10-28 18:48:13
  +08:00. Wrapper/trainer PIDs are 71/79.
- Built-in reconciliation reported `kept=13016 dropped=0 archive=None`.
  Independent audit at step 13,020 found train steps 1-13,020 exactly once,
  first continuation at 13,001, all 13 prior validations and three prior AMP
  events retained, and 20 finite CE-only resumed records.
- GPU use was 1,547/8,188 MiB at 40%. No new exit code or terminal result
  exists; the unchanged 40,000-step selected-A2 trainaug run is active.

## Selected-A2 trainaug terminal result

- The recovered wrapper exited 0 and `result.json` reports `ok=true` and
  `diagnostic_only=false`. No trainer or GPU compute process remains; the GPU
  terminal snapshot was 0/8,188 MiB at 0% utilization.
- History contains exactly 40,054 records: train steps 1-40,000 exactly once,
  40 full official validations at every 1,000 steps, and 14 bounded AMP
  overflow events. Every completed scalar is finite and every train record
  satisfies the selected A2 CE-only loss contract.
- Independent recomputation from every saved 21x21 global confusion matrix
  reproduced all per-class IoU, mIoU, foreground mIoU, pixel accuracy, and
  mean pixel accuracy values. Every validation covers 1,449 samples and
  244,886,857 valid pixels.
- The validation-selected best is step 39,000: mIoU 0.776017, foreground mIoU
  0.767576, pixel accuracy 0.949276, and CE loss 0.200152. Terminal step 40,000
  has mIoU 0.773067, foreground mIoU 0.764470, pixel accuracy 0.948734, and CE
  loss 0.202265; therefore `best.pt`, not the terminal checkpoint, remains the
  single-scale selection candidate.
- Non-diagnostic `best.pt` is step 39,000 with SHA-256 `d2e6fb31...`;
  non-diagnostic `last.pt` is step 40,000 with SHA-256 `06ddc3f8...`. Each
  contains 1,036 recursively inspected tensors and all floating tensors are
  finite. The best validation JSON SHA-256 is `2ce5a40d...`.
- Result/history/terminal-validation SHA-256 values are `79065492...`,
  `2be3ea7a...`, and `5c89d3fe...`. The three TensorBoard event SHA-256 values
  are `329dc372...`, `e25838d2...`, and `6b5768fe...`.
- The terminal recovery segment ran 47,897.169 seconds and peaked at
  1,283,440,128 allocated CUDA bytes. Its clean injected source commit is
  `fd1f6d3096e048c53ea1c8b4f4e67080ba79e156`.
- Machine-readable independent audit
  `artifacts/selected_a2_trainaug_terminal_audit.json` has SHA-256
  `14545e48...`. Formal base training is complete, but milestone 06 remains in
  progress until validation-based TTA/ensemble selection is recorded.
