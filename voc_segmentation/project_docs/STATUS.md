# Project status

- Last updated: 2026-10-26 00:30 +08:00
- Overall state: **In progress**
- Active application root: `voc_segmentation/`
- Design authority: `VOC2012_Semantic_Segmentation_Project_Plan.md` V1.0
- Highest verified active milestone: **03 — CNN baseline**
- Next milestone: **04 — Transformer baseline**
- First hardware-safe model profile: MiT-B1, 384×384, batch 2, effective batch 8

## Milestone summary

| ID | Milestone | State | Required next evidence |
| --- | --- | --- | --- |
| 00 | Project reset | Passed | New driver valid; ledger and standalone root created |
| 01 | Environment and source | Passed | Clean Docker build and CUDA allocation report |
| 02 | VOC audit and SBD | Passed | Official audit plus provenance-verified, leak-free 10,582-image trainaug |
| 03 | CNN baseline | Passed | 20,000-step formal run complete; best val mIoU 0.562094 at step 9,000 |
| 04 | Transformer baseline | In progress | Formal trainer diagnostics verified; 20,000-step official-train run pending |
| 05 | GCB diagnostics and ablations | Not started | Interface/loss tests, smoke, overfit, controlled ablations |
| 06 | Formal training and selection | Not started | trainaug run and validation-selected checkpoint |
| 07 | test400 inference | Not started | Exactly 400 validated masks |
| 08 | Report and package | Not started | Evidence-backed report and final archives |

## Current facts and blockers

- The user deleted `SegMAN-main/`; that deletion is committed in `bdc48de` and
  the tree must not be restored implicitly.
- The former SegMAN source, docs, and tracked evidence remain recoverable from
  Git commit `111f8a0`. Gitignored runtime checkpoints/logs are not active
  evidence for the standalone model.
- VOC2012 and test400 passed the full audit. SBD was acquired from the official
  Berkeley source, checksum-verified, extracted under `data/sbd/`, and converted
  into a leak-free 10,582-sample trainaug manifest.
- The latest plan states that ImageNet classification pretraining, SBD, TTA,
  and ensembles are allowed. Artifact sources and hashes are still required.
- Student name and ID remain unknown and block only final packaging.
- Docker apt installs against `archive.ubuntu.com` / `security.ubuntu.com` and
  `mirrors.aliyun.com` failed on this network; the image builds with
  `APT_MIRROR=mirrors.tuna.tsinghua.edu.cn` (Dockerfile default).

## Next action

Launch the reportable 20,000-step SegFormer official-train run from the final
clean handoff revision. Use the complete 1,449-image official validation split
every 1,000 steps and select only by validation mIoU; do not treat any
diagnostic metric or checkpoint as reportable.

## Latest checks

- Image `voc2012-gcb:dev` built cleanly after switching apt to Tsinghua.
- In-image unit tests: 4 passed.
- GPU check passed on RTX 4060 Laptop GPU; report saved at
  `voc_segmentation/artifacts/environment.json`.
- Persistent development container `voc2012-gcb` was created. The user reports
  that the prescribed CUDA check and four-test suite both passed inside it;
  exact console text was not retained beyond the saved environment report.
- The first two host invocations of the expanded six-test suite failed because
  the managed Windows sandbox denied Python-created temporary directories.
  Running the same suite in the target container passed all six tests.
- Full official-data audit passed: train=1464, val=1449, test400=400, no split
  overlap/duplicates/missing files/dimension mismatches/invalid labels, and 20
  deterministic overlay images were generated.
- Official SBD archive verified: 1,419,539,633 bytes, MD5
  `82b4d87ceb2ed10f6038a1cba92111cb`, SHA-256
  `6a5a2918d5c73ce032fdeba876574d150d9d04113ab87540a1304cbcc715be53`.
- SBD conversion passed: trainaug=10,582, converted extras=9,118, validation
  overlap=0, duplicates=0, invalid labels=0, dimension mismatches=0, errors=0.
- The first bind-mount conversion was interrupted for poor small-file I/O;
  conversion completed in container-local storage and 9,118 masks plus both
  10,582-line manifests were verified after copying to the host.
- Final in-container regression suite: 7 tests passed in 2.099 seconds.
- Project status check reports `data_audit_and_sbd=complete`, highest verified
  stage `data_audit_and_sbd`, and next stage `cnn_baseline`.
- Standalone-project reset and completed data foundation were committed as
  `bdc48de` (`chore: replace SegMAN with standalone VOC project`). Runtime
  datasets and generated artifacts remained ignored.
- DeepLabV3+ dataset/model/metric interfaces passed all 11 in-container tests
  in 3.200 seconds; no pretrained weight was downloaded by these tests.
- The first DeepLabV3+ CUDA smoke failed before model execution because the
  loader rejected official VOC palette-mode (`P`) masks; no smoke artifact or
  checkpoint was accepted from this attempt.
- The first synthetic `P`-mode regression fixture remapped index 20 to 1 on
  PNG save because it lacked an explicit palette; real VOC indices were read
  correctly. This test failure requires a corrected fixture before rerunning.
- The next CUDA smoke reached loss/backward but produced `gradient_norm=NaN`;
  its mistakenly written `ok=true` artifact is rejected and must be
  overwritten after enforcing finite gradients and a safer AMP initial scale.
- Corrected DeepLabV3+ CUDA smoke passed one optimizer step at 384×384, batch 2:
  loss=3.037793, finite gradient norm=225.873306, peak allocated memory
  1,065,757,696 bytes, elapsed=1.149 seconds. The saved artifact is explicitly
  diagnostic-only and used no pretrained weights.
- CNN baseline interfaces, regression tests, failure corrections, and the
  diagnostic CUDA smoke pipeline were committed as `e502bcd`
  (`feat: add DeepLabV3+ baseline smoke pipeline`).
- Checkpoint round-trip, fixed-resize label preservation, and all prior
  contracts passed 13 in-container tests in 4.269 seconds; the two-sample
  overfit tool also passed an AST syntax check and is ready to run.
- Two-sample CUDA overfit passed: evaluation CE decreased from 3.002533 to
  0.017127 (ratio 0.005704 versus required maximum 0.5) over 100 steps. The
  100-point history, TensorBoard event, and step-100 diagnostic checkpoint were
  independently hash- and metadata-verified; no official val sample was used.
- Checkpoint/TensorBoard infrastructure and the verified two-sample overfit
  gate were committed as `8b68095`
  (`feat: add checkpointing and overfit diagnostics`).
- Explicit local ResNet-50 weight loading, the fixed torchvision V2 URL/path
  contract, and all existing behavior passed 15 container tests in 4.519
  seconds; the acquisition tool passed an AST syntax check before download.
- Official torchvision ResNet-50 ImageNet-1K V2 weights were downloaded and
  verified: 102,540,417 bytes, SHA-256
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`,
  320 tensors, strict ResNet-50 load and finite-tensor checks passed. The local
  file also produced finite `[1,21,64,80]` DeepLabV3+ CUDA logits.
- Provenance-checked acquisition, explicit local encoder loading, contracts,
  and verification records were committed as `ef7ca23`
  (`feat: add verified ResNet-50 pretrained weights`).
- The formal DeepLabV3+ trainer now provides gradient accumulation, AMP,
  encoder/decoder learning-rate groups, warmup-poly scheduling, finite-value
  gates, native-resolution global-confusion validation, validation-mIoU best
  selection, resolved configs, JSONL/TensorBoard logs, and checkpoint restart.
- The first expanded-suite invocation was denied access to Docker Desktop from
  the managed shell; the approved retry passed all 20 tests. A first one-step
  diagnostic was rejected because endpoint poly decay made its learning rate
  zero; the schedule now preserves the base rates for a one-step smoke.
- Final pretrained v4 diagnostic passed one effective-batch-8 optimizer step:
  train CE 2.973795, finite gradient norm 7.913279, encoder/decoder learning
  rates 0.001/0.01, and peak allocated CUDA memory 1,262,054,912 bytes.
- The v4 diagnostic evaluated two native-size official-val images only, so its
  mIoU 0.002043 is explicitly non-reportable. Its diagnostic checkpoint is
  323,230,158 bytes with SHA-256
  `ee43b5f640535fef43a89d233b7b4c2a60cb9d4c582df9dee11c85436a12a9b6`;
  checkpoint metadata, both optimizer groups, event hash, and seven one-point
  TensorBoard scalar series were independently verified.
- Because `.git` is not mounted into the container, reportable runs require an
  explicit 40-character host commit plus a host-clean assertion. The v4 result
  also records Python/CUDA/cuDNN versions and SHA-256 for official train/val
  split files; diagnostic mode correctly records no clean-worktree assertion.
- The verified formal trainer, evaluator, tests, diagnostics ledger, and usage
  documentation were committed as `b4201fc`
  (`feat: add formal DeepLabV3+ trainer`).
- The first formal-run preflight on 2026-10-25 could not connect to
  `dockerDesktopLinuxEngine` because Docker Desktop was not running. GPU,
  disk, and process checks did not execute, and no training process started.
- After Docker Desktop restarted, the full preflight passed: container
  `voc2012-gcb` was running, RTX 4060 GPU usage was initially 0 MiB/0%, the
  workspace volume had 96 GB available, and no prior trainer process existed.
- The reportable VOC-train run launched at 2026-10-25 07:50:42 +08:00 from
  clean commit `b7ed0f50860ee4c28911b4f841e8a1f7dc90c882`. At the first observation,
  PID 28 was active, 35 finite training records existed, no exit-code file was
  present, and GPU usage was about 1,671 MiB. The latest observed step 35 had
  CE 2.829548, gradient norm 8.267307, encoder LR 0.00007000, and decoder LR
  0.00070000. No validation metric or accepted checkpoint exists yet.
- The run exited with code 1 at attempted iteration 8,039. Training loss was
  finite, but the gradient norm was non-finite after AMP unscale. The scaler
  had grown from 8,192 at step 7,999 to 16,384 at step 8,000, identifying a
  dynamic-loss-scale overflow that the current hard gate mishandled as fatal.
- Eight full 1,449-image validations completed. mIoU progressed 0.3713, 0.4982,
  0.5360, 0.5401, 0.5547, 0.5431, 0.5551, then 0.5194. The verified best is
  step 7,000 at mIoU 0.555056; step 8,000 is the latest restart point.
- `best.pt`: step 7,000, 323,230,542 bytes, non-diagnostic, SHA-256
  `b271ebb2682450c69d21ee3cc5339c59e1d9fd2c826d3567811952ccc2dd9dea`.
  `last.pt`: step 8,000, 323,230,542 bytes, non-diagnostic, SHA-256
  `57fcb009202f2c9ff9c073c3e512e4e9eec356d6828447ab1a22c6c85d3c8d67`.
  No terminal `result.json` exists, so the CNN baseline gate remains open.
- AMP recovery now accepts only a non-finite gradient accompanied by verified
  GradScaler scale reduction, skips the unsafe optimizer update, records the
  event, and retries the same logical iteration. It aborts after more than four
  consecutive or 32 total overflows; non-AMP non-finite gradients remain fatal.
- All 21 in-container tests passed in 3.762 seconds. A forced CUDA overflow
  diagnostic passed: attempt 1 skipped with scale 1,024→512, attempt 2 completed
  step 1, and checkpoint/result metadata recorded exactly one overflow. The
  diagnostic checkpoint SHA-256 is
  `35ea803109d352ae1a9e1f10f0162b9dbec670353078f9531e6e640071414bfc`.
- The bounded recovery implementation, tests, diagnostic, and interruption
  records were committed as `1bd318b`
  (`fix: recover from AMP gradient overflow`).
- A first step-8,000 resume under clean commit `a4aeefe` successfully loaded the
  checkpoint and reached step 8,041, but appending to the prior failed run's
  history duplicated step numbers 8,001 onward. The process was intentionally
  terminated with SIGTERM (exit 143) before validation/checkpointing. The first
  termination one-liner failed because PowerShell expanded its shell `$()`;
  exact PID 270 was then verified and stopped. `last.pt` remains unchanged at
  SHA-256 `57fcb009202f2c9ff9c073c3e512e4e9eec356d6828447ab1a22c6c85d3c8d67`.
- Resume-history reconciliation now parses every JSONL record, rejects invalid
  or duplicate completed records at/before the checkpoint, archives the full
  pre-resume history, and atomically retains only records through the checkpoint
  step before append mode starts. All 23 container tests passed in 2.712 seconds.
- The reconciliation implementation, tests, and first-resume interruption
  evidence were committed as `6fab976`
  (`fix: reconcile history before training resume`).
- The reconciled recovery launched at 2026-10-25 13:09:39 +08:00 under clean
  commit `d789bc06c6db6883e4010ceba587d3212484ad5c`. It archived all 8,087 prior
  JSONL records (SHA-256
  `fcb176067aafa16520f8d85f6178b92e1ed811adb4dcec5830a38c157f1b2618`),
  retained 8,008 records through checkpoint step 8,000, and dropped 79
  uncheckpointed/duplicate records before appending.
- Independent observation at step 8,018 found 8,018 unique train records,
  eight validation records, zero duplicate train steps, no exit-code file, PID
  422 active, and about 2,030 MiB GPU memory at 99% utilization.
- At the user's request, exact PID 422 was terminated with SIGTERM after its
  console reached step 10,047. The wrapper saved exit code 143. Step 10,000 had
  already completed full 1,449-image validation and checkpointing; the 47 later
  uncheckpointed train records were archived and removed from active history.
- The active history now has 10,011 records through step 10,000 (10,000 train,
  10 validation, one recovered AMP overflow at step 8,380), SHA-256
  `e50f3344fdf55ca8b28da384b1996138c4ee6cb0bf1559393ba311294735aaa5`.
  The full pre-stop history archive has 10,058 records, SHA-256
  `0ff6ae540aa0312792644a4356fdf73125309f099aa6701e4d1d3a82cd47beec`.
- Saved best: step 9,000, mIoU 0.562094, SHA-256
  `07e2364f7b34490f1d42515580dd73c9c54db79143967db943e5de0f1dca311f`.
  Saved restart point: step 10,000, mIoU 0.540388, SHA-256
  `1e46f3bce6fa88eaec822ae804c88793e189cf14adf939fe0ceb62a543edde81`.
  Both checkpoints are non-diagnostic and preserve AMP scale 8,192 plus one
  recovered overflow. No `result.json` exists, so the CNN gate is not passed.
- Machine-readable stop manifest:
  `runs/deeplabv3plus_r50_voc_train/user_stop_20261025T142718+0800.json`;
  all seven recorded artifact hashes were independently matched.
- The saved interruption ledger was committed as `270f733`
  (`docs: record saved CNN training interruption`), restoring a clean source
  revision for reportable continuation.
- Formal training resumed from the verified step-10,000 `last.pt` at
  2026-10-25 16:01:32 +08:00 under clean commit
  `270f7331552b93250e5c979b6c2d345e57384c5b`. The previous exit code 143 was
  preserved as `exit_code.user_stop_step_010000.txt` before launch.
- PID 66 is active. The trainer retained all 10,011 history records through
  step 10,000, dropped none, and continued at step 10,001. At the independent
  check it had reached step 10,044 with 10,044 unique train records, ten full
  validations, one recovered AMP overflow, zero duplicate train steps, and no
  new exit-code file. GPU use was about 1,929 MiB at 67% during the first
  observation.
- Two monitoring snippets were corrected without affecting training: a shell
  `$(cat ...)` was prematurely expanded by PowerShell, and the first JSONL
  query used nonexistent key `iteration` instead of `step`. The corrected
  history query passed.
- At the user's pause request, exact PID 66 was sent SIGTERM. The trainer
  finished through attempted step 18,098, the wrapper saved exit code 143, the
  GPU returned to 0 MiB / 0%, and no `result.json` was created.
- Step 18,000 is the latest complete restart point. Its full 1,449-image
  validation has mIoU 0.499866, foreground mIoU 0.486016, pixel accuracy
  0.805109, and loss 0.690280. The non-diagnostic `last.pt` SHA-256 is
  `0af35cef20f83c5f0ac6eebde113f778edfc574befd5933adb3cf553e7127b67`.
- The validation-selected best remains step 9,000 at mIoU 0.562094 with
  SHA-256
  `07e2364f7b34490f1d42515580dd73c9c54db79143967db943e5de0f1dca311f`.
- All 18,121 pre-pause history records were preserved in
  `history_archive/user_pause_step_018000_20261025T194532+0800.jsonl`
  (SHA-256
  `468aa16e818aa7ed5709d929368442dc1929a2a1317ca930fafabb51dab9dcfa`).
  The active history was reconciled to 18,023 records through step 18,000:
  18,000 unique train records, 18 validations, five recovered AMP overflows,
  and zero duplicate train steps (SHA-256
  `baada161f3fc4832d413d276f40302da840ea797263e52c752f34b37eaae71bf`).
- Machine-readable pause manifest:
  `runs/deeplabv3plus_r50_voc_train/user_pause_20261025T194635+0800.json`;
  all seven recorded artifact hashes independently matched.
- The step-10,000 resume and step-18,000 pause ledger was committed as
  `05048a5` (`docs: record CNN resume and pause at step 18000`), restoring a
  clean provenance revision for continuation.
- Formal training resumed at 2026-10-25 21:18:13 +08:00 from the verified
  step-18,000 `last.pt` under clean commit
  `05048a5f57636cd0f17553779d5a7fb4ee33e1c3`. The previous exit code 143 was
  preserved as `exit_code.user_pause_step_018000.txt` before launch.
- PID 68 is active. The trainer retained all 18,023 active records through
  step 18,000, dropped none, and continued at step 18,001. At independent
  verification it had reached step 18,033 with 18,033 unique train records,
  18 full validations, five recovered AMP overflows, zero duplicate train
  steps, and no new exit-code file. GPU use was about 1,929 MiB at 86%.
- The formal DeepLabV3+ run completed at 2026-10-25 22:10:35 +08:00 with exit
  code 0 and `result.json` reporting `ok=true`, `diagnostic_only=false`, and a
  complete 1,449-image official validation. No trainer process remains.
- Active history contains exactly 20,000 unique completed train records for
  steps 1–20,000, zero missing or duplicate steps, 20 validations at every
  1,000 steps, and six valid isolated AMP backoffs. Every completed train and
  validation numeric value is finite.
- Validation selection was independently reproduced from history. The best is
  step 9,000 at mIoU 0.562094, foreground mIoU 0.547951, pixel accuracy
  0.860711, and validation loss 0.408478. The non-diagnostic `best.pt`
  SHA-256 is
  `07e2364f7b34490f1d42515580dd73c9c54db79143967db943e5de0f1dca311f`.
- The terminal step-20,000 validation has mIoU 0.521702, foreground mIoU
  0.507856, pixel accuracy 0.823339, and loss 0.633271. The non-diagnostic
  `last.pt` SHA-256 is
  `653dcb628aa6231631b2594b6075f6c9720fc9ac0802bfb0d3bfc5172739ec38`.
- The final result, history, configs, checkpoints, validation JSON,
  TensorBoard event, console, and exit-code hashes all matched the
  machine-readable completion manifest
  `runs/deeplabv3plus_r50_voc_train/formal_completion_20261025T235047+0800.json`.
  The terminal resume segment used clean commit `05048a5`, seed 42, effective
  batch 8, 1,588,047,360 peak CUDA bytes, and 3,175.253 seconds.
- Two initial numeric-history assertions used incorrect field assumptions
  (`encoder_lr` at top level, then non-null `gradient_norm` and unprefixed
  scale keys for overflow records). Both failed read-only without changing
  artifacts. The corrected assertion found zero non-finite completed train or
  validation records and valid scale backoff for all six overflow events.
- The project-status helper now reads CNN evidence from the configured
  `runs/deeplabv3plus_r50_voc_train/result.json`. Its summary correctly reports
  `cnn_baseline=complete`, `highest_verified_stage=cnn_baseline`, and
  `next_stage=transformer_baseline`. The updated driver skill passed
  `quick_validate.py`.
- The first SegFormer dependency probe failed before model creation: installed
  `timm==0.9.16` exposes no `mit_b1` model and raises `RuntimeError: Unknown
  model (mit_b1)`. No source or runtime artifact was modified by the probe.
- A native PyTorch MiT-B1 encoder now implements overlapping patch embedding,
  spatial-reduction self-attention, Mix-FFN depthwise convolution, stochastic
  depth, and four feature stages at reductions 4/8/16/32 with channels
  64/128/320/512. Official-style module names are retained for explicit local
  checkpoint conversion/loading.
- The SegFormer baseline adds four linear feature projections, bilinear
  alignment, 1×1 fusion, dropout, and a 21-class classifier restored to input
  resolution. It has 13,682,645 parameters: 13,151,424 encoder and 531,221
  decoder parameters.
- The resource-safe config fixes MiT-B1, 384×384 crops, physical batch 2,
  effective batch 8, and the accepted explicit weight path under
  `artifacts/weights/mit_b1_imagenet1k/`.
- Focused SegFormer/config tests passed 8/8 in 1.254 seconds. The complete
  in-container regression suite passed all 28 tests in 4.766 seconds,
  including four-scale shapes, finite 21-class original-size logits, strict
  local encoder checkpoint round-trip, and rejection of non-MiT-B1 profiles.
- The first attempt to record the Transformer probe and implementation used a
  nested quoted `python -c` command in a PowerShell argument and failed with a
  parser error before `record_step.py` ran. No file changed. The failure and
  both substantive steps were then recorded separately with safe quoting.
- NVIDIA's official Hugging Face `nvidia/mit-b1` revision
  `13ddceec4e8bdf401e7cd7acf5aebc526222518c` was acquired. The 54,732,285-byte
  ImageNet-1K source archive matches its published LFS SHA-256
  `980b86b60db37b1b1528086f6c53d253d879e9e09ebe07397b40fd275738a3bf`.
- The deterministic conversion discarded only the 1,000-class classifier and
  mapped 194 source tensors to all 176 native encoder tensors, including
  concatenated attention key/value projections. Strict loading, tensor
  finiteness, and a four-scale finite forward pass succeeded. The converted
  52,658,798-byte checkpoint SHA-256 is
  `8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b`.
- Provenance records the NVIDIA Source Code License for SegFormer, which limits
  use to non-commercial research or evaluation, plus ImageNet terms. This
  course project uses the checkpoint only for educational evaluation and
  asserts no broader rights. The provenance JSON SHA-256 is
  `2bf52c96aeb967605d4fe9975497c861cf3ee824bf62e6c892c0de3d688cf6ca`.
- A first immediate relative-path verification after download reported the
  final file absent, while the subsequent directory listing showed the atomic
  rename had completed. Absolute-path verification then matched the published
  size and SHA. An earlier read also requested nonexistent
  `voc_segmentation/.gitignore`; the root `.gitignore` was the applicable file.
  Neither read-only failure modified project state.
- The conversion/config/SegFormer focused tests passed 9/9 in 1.096 seconds;
  the acquisition tool's second run reused both files and repeated the same
  strict-load/hash results. The complete container suite passed 29/29 in 5.894
  seconds.
- The pretrained SegFormer CUDA smoke passed one 384×384, batch-2 AdamW step
  on official VOC train: loss 2.938904, finite pre-clip gradient norm 17.925528,
  encoder/decoder learning rates 0.00006/0.0006, AMP scale 1,024 unchanged,
  623,762,432 peak allocated CUDA bytes, and 1.747 seconds. The result is
  diagnostic-only; SHA-256
  `742b76f757d38b69385db2337d2a6ebf886d854a9abbe604c17cd38240df30b0`.
- The deterministic pretrained SegFormer two-sample gate passed 100 AMP steps
  at 128×128. Evaluation CE decreased from 2.910582 to 0.016329 (ratio
  0.005610, required at most 0.5); same-sample pixel accuracy was 0.996751,
  peak allocated CUDA memory was 409,270,272 bytes, and elapsed time was 8.347
  seconds. The result, 100-line history, checkpoint, and TensorBoard hashes
  were independently matched; `result.json` SHA-256 is
  `9a6e4eeae1be267ba27bc1c6fc5ada4a31e4df345db2ac1eae6950e7bfea0210`.
- A read-only reference inspection first requested nonexistent
  `engine/checkpoint.py`; the actual exported implementation is
  `engine/checkpointing.py`. A later host finiteness audit used unavailable
  `[double]::IsFinite` on the installed PowerShell/.NET runtime. The corrected
  container-Python audit confirmed all 100 losses and gradient norms finite.
  Neither failed command changed artifacts.
- `tools/train_segformer.py` now mirrors the proven CNN formal-training
  contract while using the native MiT-B1/SegFormer model and disjoint AdamW
  encoder/decoder groups. It enforces the converted checkpoint hash, effective
  batch 8, warmup-poly scheduling, finite-value gates, bounded AMP recovery,
  native-resolution global-confusion validation, validation-mIoU selection,
  atomic checkpoints, resolved configs, JSONL/TensorBoard evidence, explicit
  host Git provenance, and history-safe resume.
- Trainer syntax and two focused optimizer/provenance tests passed. The full
  target-container suite passed 31/31 in 5.918 seconds.
- The one-step effective-batch-8 diagnostic passed with train CE 3.000786,
  gradient norm 10.172207, encoder/decoder learning rates 0.00006/0.0006,
  two non-reportable validation samples, 702,257,152 peak CUDA bytes, and a
  diagnostic checkpoint SHA-256
  `9f30fb5be0af6ab2184baee36c26bbcda387e3868d71c03b547d53d907308183`.
  Its independently verified `result.json` SHA-256 is
  `bffc944d1a59f25f278756050695654646bbb72cef8b7e47bdde31def61d5ffb`.
- Forced AMP recovery reduced scale 1,024 to 512, skipped the unsafe update,
  and retried step 1. A subsequent resume loaded that checkpoint, retained all
  three pre-resume records, completed unique train and validation steps 1, 2,
  and 3, and preserved the single overflow count. Final diagnostic checkpoint
  SHA-256 is
  `b1d503bbb509ced4ff6985cfcb85d6458e2b1dc88b73e7585143e0b52430033a`;
  independently verified resumed `result.json` SHA-256 is
  `c20ad19c695c868ca3fd5fa5ce8648dce72067ea0b51fc960d44284dc18d1179`.
- A reportable-mode probe without an explicit 40-character commit and clean
  worktree assertion was rejected before any output directory was created, as
  required. All trainer diagnostic metrics and checkpoints remain ineligible
  for comparison, model selection, or submission.
- The verified trainer and evidence were committed as `7c22761`. Its staged
  whitespace check identified an extra EOF blank line but PowerShell continued
  to the commit; the defect was removed and the immutable failure/correction
  records were committed as `705dad0`. The correction is formatting-only and
  the working-tree whitespace audit then passed.
