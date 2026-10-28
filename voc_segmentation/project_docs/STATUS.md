# Project status

- Last updated: 2026-10-28 18:49 +08:00
- Overall state: **In progress**
- Active application root: `voc_segmentation/`
- Design authority: `VOC2012_Semantic_Segmentation_Project_Plan.md` V1.0
- Highest verified active milestone: **05 — GCB diagnostics and ablations**
- Next milestone: **06 — Formal training and selection**
- First hardware-safe model profile: MiT-B1, 384×384, batch 2, effective batch 8

## Milestone summary

| ID | Milestone | State | Required next evidence |
| --- | --- | --- | --- |
| 00 | Project reset | Passed | New driver valid; ledger and standalone root created |
| 01 | Environment and source | Passed | Clean Docker build and CUDA allocation report |
| 02 | VOC audit and SBD | Passed | Official audit plus provenance-verified, leak-free 10,582-image trainaug |
| 03 | CNN baseline | Passed | 20,000-step formal run complete; best val mIoU 0.562094 at step 9,000 |
| 04 | Transformer baseline | Passed | 20,000-step run complete; best val mIoU 0.733889 at step 19,000 |
| 05 | GCB diagnostics and ablations | Passed | A1 0.736626; A2 0.755643; A3 0.726746; A4 0.739100; A2 selected |
| 06 | Formal training and selection | In progress | A2 trainaug active after verified continuation at step 13,001 |
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

Monitor the active selected-A2 trainaug recovery through full official
validation and checkpoint intervals; keep best-model selection separate from
the most recent restart checkpoint.

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
- The first formal-run preflight was blocked by managed access to the Docker
  Desktop named pipe. The approved retry passed: `voc2012-gcb` was running,
  RTX 4060 GPU use was 0 MiB/0%, `/workspace` had 756 GB available, no trainer
  process existed, the formal output directory was absent, and the MiT-B1
  checkpoint matched SHA-256
  `8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b`.
- The reportable 20,000-step SegFormer run launched at 2026-10-26 00:33:53
  +08:00 from clean commit
  `d1f25d71c7f8e3fc3b0407e7cd8fa252e2d3a839`. Container PID 1248 remained
  active. At independent observation it had 17 unique finite training records
  through step 17, zero validation or overflow records, no exit-code file or
  terminal result, and about 1,067 MiB GPU memory in use.
- The formal SegFormer run exited with code 0 and no trainer process remains.
  `result.json` reports `ok=true`, `diagnostic_only=false`, clean source commit
  `d1f25d71c7f8e3fc3b0407e7cd8fa252e2d3a839`, 20,000 iterations, and 20
  complete 1,449-image official validations.
- Independent history validation found exactly 20,000 unique train records for
  steps 1 through 20,000, validation records at every 1,000 steps, and six
  isolated AMP overflow records with verified scale backoff. All recorded
  training and validation numeric values are finite.
- Validation selection was independently reproduced. Best step 19,000 has
  mIoU 0.733889, foreground mIoU 0.723720, pixel accuracy 0.938803, and loss
  0.298455. The non-diagnostic `best.pt` SHA-256 is
  `dde4ca693315e290a431cd60eb3c8446b099b457adf2421620a97c13c34fdbad`.
- Terminal step 20,000 has mIoU 0.732272, foreground mIoU 0.722018, pixel
  accuracy 0.938912, and loss 0.294703. The non-diagnostic `last.pt` SHA-256
  is `cb9fa1a73b2d60951ff41cac107db5b7f6ba10e72764d411a7422d4348fb95cc`.
- All independently audited result, history, checkpoint, config, pretrained
  weight, split, validation JSON, and TensorBoard hashes matched. Result
  SHA-256 is
  `6b9fac34b0b6fb36c5e6a63f226d3c61592eb476717b05ae3c4edf72a2305ecd`;
  history SHA-256 is
  `94dc7af8b8359113793ddeab9ceaebb18cf50a266e65c746e9a898913dd099da`.
  Runtime was 28,622.064 seconds and peak allocated CUDA memory was
  816,525,824 bytes.
- The project-status helper previously looked for obsolete
  `runs/segformer/result.json`. It now requires non-diagnostic, full-validation
  baseline results at `runs/segformer_b1_voc_train/result.json`. The helper
  reports Transformer complete and GCB diagnostics next; skill validation
  passed.
- GCB-Seg now implements the planned five-branch GCM (1×1, depthwise-separable
  dilations 1/3/6, and global pooling), 256-channel gated top-down GFF, C1/C2
  BAB with 64-channel boundary features and one edge logit, full-resolution
  main/auxiliary/edge output contract, and ablation switches. All new
  convolutional blocks use GroupNorm; the complete model has 15,841,707
  parameters, including 13,151,424 in the strictly loaded MiT-B1 encoder and
  2,690,283 in new modules.
- The composite loss implements label-smoothed CE + 0.5 Lovász-Softmax + 0.2
  boundary BCE/Dice + 0.4 auxiliary CE. Boundary truth uses valid
  four-neighbor differences, one 3×3 dilation, ignore-255 masking, and no
  artificial image-border edges. Lovász processes only present classes and
  excludes ignore pixels.
- The first 13-test focused run passed all model/config checks but three loss
  checks failed because PyTorch 2.1 rejects an in-place Lovász gradient update
  on overlapping slices. Replacing it with an equivalent non-in-place
  difference corrected the implementation. Focused tests then passed 13/13 in
  2.986 seconds and the full container suite passed 40/40 in 5.282 seconds.
  Strict pretrained loading passed; the decoder contains 28 GroupNorm and zero
  BatchNorm layers.
- The first pretrained 384×384, batch-2 CUDA smoke completed an optimizer step
  but exposed only 31 positive boundary pixels among 234,318 valid pixels.
  Direct audit showed VOC object classes were separated by 1–3-pixel 255 void
  bands, so immediate-valid-neighbor comparison suppressed genuine contours.
  The `ok=true` artifact was rejected and overwritten.
- Boundary generation now additionally bridges only different valid endpoint
  classes whose intervening pixels are entirely 255, up to a width of three.
  Ignore pixels remain invalid, and padding-only void regions have no opposite
  valid endpoint and produce no edge. Corrected focused tests passed 8/8 in
  3.045 seconds and the full suite passed 40/40 in 5.253 seconds.
- The corrected CUDA smoke passed one complete AdamW+AMP update on fixed
  official-train samples `2008_005679` and `2008_002175`. Total/CE/Lovász/
  edge/aux losses were 4.829429/2.930575/0.887470/0.843542/3.216027;
  2,474 boundary positives were recovered, pre-clip gradient norm was
  12.056706, AMP scale remained 1,024, peak CUDA allocation was 1,078,124,544
  bytes, and elapsed time was 3.119 seconds.
- Independent audit matched the configured loss formula, all three output
  shapes, exact encoder/decoder parameter coverage, configuration, provenance,
  and checkpoint hashes. The diagnostic result SHA-256 is
  `62ba3dfeddcddffd4dfc79a88186b84a4a8a3010b11ce702ae73c3a33de3998e`.
- `tools/overfit_gcb_seg.py` now implements the fixed two-sample diagnostic at
  128x128 for 100 AMP steps. It records all seven composite-loss values,
  gradient norms, both learning rates, AMP scale, TensorBoard series, a
  diagnostic-only checkpoint, pretrained provenance, boundary counts, and
  hashes. Initial/final evaluation reuses the same deterministic stochastic
  masks so the complete objective is comparable.
- The new optimizer/loss-recording contracts passed 2/2 tests in 0.025
  seconds. The complete target-container suite passed 42/42 in 6.795 seconds.
- The deterministic pretrained GCB-Seg two-sample run passed all 100 AMP
  updates. The matched initial/final composite objective fell from 4.708998 to
  0.603445, ratio 0.128147 versus the required maximum 0.5. All seven loss
  components and gradient norms were finite; AMP scale stayed at 1,024.
- The same-sample diagnostic reached pixel accuracy 0.994826 and mIoU
  0.936919; these are explicitly not validation results. The target contained
  2,051 positive boundary pixels among 31,699 valid pixels. Runtime was 24.154
  seconds and peak allocated CUDA memory was 467,688,960 bytes.
- The first independent-audit invocation failed read-only because PowerShell
  removed nested Python quotes, producing `NameError: artifacts is not
  defined`. No artifact changed. The corrected Base64-carried audit passed.
- Independent audit verified all 100 sequential history records and formulas,
  finite checkpoint tensors, 11 TensorBoard tags with 100 values each,
  parameter coverage, boundary signal, stable AMP, and every saved hash.
  Result/history/checkpoint/event SHA-256 values are respectively
  `e95bcfeae48aa43c5e3bc79172cada45694efddc48c0b21d8908b615c94084fc`,
  `86535a8bf4f8e9397f74c3fe40aff8d29b5d4d83315b480e8622eb5467e55b5a`,
  `14f6f2ad4a82c7567bb4a96ae30be85571d2ed6ebdaf24e5437278c48319fd17`,
  and `09df89f0080c86bfdb447a978406cd940efbbbf398a01e88a23583fb08fe6dcf`.
- Decision 0004 pins the controlled A0-A4 comparison to the completed A0's
  20,000-step VOC-train budget. A1 adds GCM with linear fusion, A2 adds gated
  GFF, A3 adds BAB/edge loss, and A4 adds Lovasz. Auxiliary loss and label
  smoothing remain zero so they cannot confound the four planned factors.
- The proven SegFormer formal trainer now supports GCB output mappings and
  arbitrary finite loss-component histories while preserving its existing CE
  behavior, validation evaluator, provenance, checkpoint, resume, Git, and
  bounded AMP-recovery contracts. `tools/train_gcb_seg.py` supplies GCB
  defaults without duplicating the training engine.
- Model/config/trainer focused tests passed 21/21 in 2.362 seconds. The full
  target-container regression suite passed 48/48 in 5.962 seconds.
- The one-step A4 diagnostic completed one effective-batch-8 AdamW+AMP update
  and two explicitly non-reportable validation images. Total/CE/Lovasz/edge
  losses were 3.558069/2.924114/0.932956/0.837382, the pre-clip gradient norm
  was 9.638385, AMP had zero overflows, and peak CUDA allocation was
  1,165,862,400 bytes. Runtime was 13.461 seconds.
- The diagnostic checkpoint is step 1 and `diagnostic_only=true`; the two-image
  mIoU 0.208359 is not an ablation result. Independent audit verified the
  formula, history, resolved overrides, finite checkpoint tensors, 15
  TensorBoard series, split/weight metadata, and all hashes. Result/history/
  checkpoint/event SHA-256 values are respectively
  `c2cc812cc163eddb353f32462fc3c7100a0c101d5c95eee1c9df7b66a14790ef`,
  `914ebbe02f245815831d8493c95e6f8b68f038eb7a0c4e21639cb66e06fbd5f8`,
  `618f90ff79730461cda5c84927bdda3896606e9f6106c4d287c3ba45413ba80e`,
  and `7fd08e26b579ff4488975b971d775b74b3ef0c0e34c84cee44ad1a92491c1720`.
- The verified implementation was committed as `f18e435`, but the staged
  whitespace check reported extra EOF blank lines in four ablation configs and
  the GCB wrapper. Because the shell command used `;`, commit execution still
  continued. The five formatting defects were then removed and a clean
  `git diff --check` passed; no executable content or runtime evidence changed.
- The first parallel A1 preflight was rejected because three Docker queries
  hit intermittent Windows named-pipe permission denial; no training started.
  An approved serial retry verified the GPU, disk, hashes, output absence, and
  config but its process regex matched the checking shell itself. A direct
  process-table query initially hit the same pipe denial, then succeeded under
  approved access and proved that no Python trainer was running.
- Final A1 preflight evidence: clean commit
  `9386a12918cb01d0deee122047e655d6e823ea40`; RTX 4060 at 114/8,188 MiB and
  0% utilization with no compute process; 757 GB workspace free; no GCB
  trainer; formal output absent; config SHA-256
  `bceb04f2d0a9ae6d35c7b65f6da0c33ead063399c9452c9043b9b49baeef4e66`;
  weight SHA-256
  `8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b`;
  parsed A1 contract is linear GCM, CE only, 20,000 steps, validation every
  1,000, official train/val.
- The preflight ledger was committed as clean source
  `c956520ca370a892fe4d63d9d181d2e0818315e9`. The first launch-wrapper syntax
  and hash query was denied by the Docker named pipe, so no process started;
  approved retry passed syntax, measured wrapper SHA-256
  `3cd4fce521998a761c982f92ac0ebce809dd9ebf355a54748d45a67f3922fbd3`,
  and launched the reportable A1 job at 2026-10-26 09:38:08 +08:00.
- Wrapper PID 2447 and trainer PID 2454 were active. GPU use was about 1,134
  MiB at 14%. The first Base64 audit nested inside `sh -lc` failed read-only
  because PowerShell truncated its quotes; direct container Python then passed.
- At the corrected observation, history contained 51 unique sequential finite
  train records through step 51, no validation, no exit code, and no result.
  Step 51 CE was 2.916139, pre-clip gradient norm 26.750391, and AMP scale
  1,024. The resolved config records 20,000 reportable steps, full validation,
  and no diagnostic override; the injected source commit matches `c956520`.
- Final handoff observation found wrapper/trainer PIDs 2447/2454 still active
  at step 182. The latest CE was 1.692492, GPU use was about 1,138 MiB at 4%,
  and neither `exit_code.txt` nor `result.json` existed. Source/documentation
  worktree was clean at ledger commit `fd24be2` before this snapshot record.
- A1 step 1,000 completed a full 1,449-image official validation: mIoU
  0.604609, foreground mIoU 0.589999, pixel accuracy 0.902396, mean pixel
  accuracy 0.724140, and CE 0.329029. All 21 class IoUs were defined.
- Independent recomputation from the 21x21 confusion matrix matched every
  saved metric and its 244,886,857 valid-pixel total. History had 1,348 unique
  finite train records during the audit, zero AMP overflows, and one full
  validation record at step 1,000.
- `best.pt` is a non-diagnostic step-1,000 checkpoint, 166,840,812 bytes, with
  finite model tensors and SHA-256
  `b2165bb8ba17bd20af5b1df2a90b7e4ad7aa72ba2c404f4ee64999b8ec38a0db`.
  The validation JSON SHA-256 is
  `1e78e80303725915bab05d34f364d2f73584ba521f67e68a3e9689c542b5f49f`.
  TensorBoard exposed 15 expected scalar tags. PIDs 2447/2454 remained active,
  GPU use was about 1,162 MiB at 20%, and no terminal artifacts existed.
- A1 exited successfully with code 0 and no Python trainer remained; the GPU
  returned to 272 MiB and 0%. Terminal evidence contains exactly 20,000 unique
  train steps, 20 full 1,449-image validations at steps 1,000-20,000, and five
  isolated AMP overflows with verified scale reductions. All completed
  training and validation numeric values are finite.
- Independent validation selection reproduced the A1 best at step 19,000:
  mIoU 0.736626, foreground mIoU 0.726605, pixel accuracy 0.938822, and CE
  0.328063. This is +0.002738 mIoU (+0.274 percentage points) over the matched
  A0 best 0.733889, with both best points at step 19,000.
- Terminal step 20,000 mIoU was slightly lower at 0.736464 with CE 0.328045.
  Best/last checkpoint SHA-256 values are
  `55839b61b0be34dbcc88217528ca1cd93859fcc089164c36357a2792e221f990`
  and `c8c93dd77a7b8686c5e464ced2b326481378c53b376a644fcbbcfa0897bc4a3a`;
  both are non-diagnostic, finite, and preserve selection step 19,000.
- The first terminal audit incorrectly expected 20,000 TensorBoard samples per
  tag from the default `EventAccumulator`, whose reservoir cap returned 10,000;
  all earlier artifact checks had passed. Repeating with scalar size guidance
  zero exposed all 20,000 train points plus five AMP events and passed.
- Result/history/event SHA-256 values are respectively
  `1813743cb1920c69cfbc3b265b220eb9e8086181c8363362190632e7371f7e41`,
  `05b64a5b0968ef629ff16468ca5e6377ad1c02fbd630bd4cc74d997e9369d8c6`,
  and `0778f767cdf91f291aea46b1749ef29264381310775ce3dce1309b1524bbdf06`.
  Runtime was 34,049.165 seconds and peak CUDA allocation 828,434,432 bytes.

## A2 formal-run preflight

- The first batched Docker query was partially denied by the intermittent
  Windows Docker Desktop named-pipe permission layer. Read-only disk, output-
  absence, and hash checks still passed; no output was created and no trainer
  started. Approved serial access then completed the missing GPU and process
  checks.
- A first parsed-contract audit requested nonexistent key `val_interval` and
  stopped with `KeyError`; the configuration actually uses `validate_every`.
  The corrected read-only audit passed and no artifact changed.
- Host source was clean at full commit
  `670709997733f99122da0ab0063ab218087e3725`. No GPU compute process existed;
  the RTX 4060 reported 285 of 8,188 MiB used and 0% utilization. The process
  table contained only PID 1 and stale completed-run log tails, with no Python
  trainer. `/workspace` had 757 GB available and
  `runs/gcb_ablation_a2_gff_voc_train` was absent.
- Container SHA-256 values matched the host A2 configuration and registered
  MiT-B1 weight: config
  `c711ec5498517081b9e6270c255986daef4061b3bea15a59c881e64ddf6ffc95`;
  weight
  `8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b`.
- A1 and A2 data, training, loss, and inference sections match after parsing.
  The only model differences are `feature_fusion_mode` from `linear` to
  `gated` and `use_gated_fusion` from false to true. A2 remains CE-only,
  disables BAB and auxiliary output, uses official train/val, runs 20,000
  steps, and validates every 1,000 steps.

## A2 reportable launch

- The passed preflight ledger was committed as clean source
  `7ca7d0d892fec7529fc5193b078c6c9d30cb27b7`. The ignored launch wrapper
  passed `sh -n` and has SHA-256
  `3daacd10132943540e2571bc1f54f68a47ab6218c26688f3a9323d1b1820c5e8`.
- The reportable A2 job started at 2026-10-26 19:01:56 +08:00 with wrapper
  PID 2784 and trainer PID 2791. Its command injects the exact source commit
  and clean-worktree assertion. GPU use was 1,684 MiB at 50% during the first
  process snapshot.
- Independent early audit found 91 unique sequential train records through
  step 91. Every recorded scalar was finite and each total loss exactly
  matched CE; the latest observed loss was 1.848219. The resolved configuration
  records A2 gated fusion, no BAB or auxiliary head, 20,000 reportable steps,
  full validation every 1,000 steps, and no validation limit.
- `source_commit.txt` matches the injected revision. Neither `exit_code.txt`
  nor `result.json` existed, so no terminal result is claimed.

## A2 terminal result

- The launch wrapper saved exit code 0, `result.json` reports `ok=true` and
  `diagnostic_only=false`, no Python trainer remains, and the GPU was idle at
  0% with 839 MiB allocated at the terminal observation.
- History contains exactly 20,026 records: 20,000 unique sequential train
  steps, 20 complete 1,449-image official validations at every 1,000 steps,
  and six isolated recoverable AMP overflows. All recorded numeric values are
  finite, and every completed total loss equals CE as required by A2.
- Independent recomputation from every saved 21x21 global confusion matrix
  reproduced all per-class IoUs, mIoU, foreground mIoU, pixel accuracy, and
  mean pixel accuracy. Each validation contains 244,886,857 valid pixels.
- Best A2 is step 18,000: mIoU 0.755643, foreground mIoU 0.746083, pixel
  accuracy 0.945714, and validation CE 0.274758. This is +0.019017 absolute
  mIoU (+1.902 percentage points) over A1 and +0.021755 (+2.175 points) over
  A0 under the matched single-seed 20,000-step budget.
- Terminal step 20,000 is lower at mIoU 0.752653, foreground mIoU 0.742934,
  pixel accuracy 0.945027, and CE 0.283783; model selection therefore uses
  `best.pt`, not `last.pt`.
- Both checkpoints are non-diagnostic and all 1,036 recursively discovered
  tensors in each checkpoint are finite. Best/last SHA-256 values are
  `82ef8b83f1c5e65dd47bd2383b2788b374bedef10f3747ae54b9bf884e59a624`
  and `fb72e73a5ddc2c24630b99aa4073bc506a2ed10ad02c0753bd4b981ecfdc0504`.
- The first terminal audit incorrectly required TensorBoard's tag order to
  match the sorted order stored in `result.json`; the sets were identical.
  The second incorrectly required all train-related tags to have exactly
  20,000 points, while `AMP/scale` and `System/amp_overflow` correctly contain
  20,006 points because six recovered overflows add event records. The third
  audit validated the exact step multiplicities and passed.
- SHA-256: result
  `7abb59bbe865a164df18bfa3aca4ce85ac3705fc2d2abc4f2bc603edc465674b`;
  history `3e14be84e34869161a7e96df406d993cde899b2d275604bb3d66a496c96aff37`;
  best validation
  `a1402f246128314c9cfb19f8e12c8571227e434b631b6ef955afca059a315ad7`;
  terminal validation
  `4974eb0aaf3fdf4929b6d6725756449b151f42a9f8f76f38c39f8ba7d8da7ed0`;
  TensorBoard event
  `7928d46d15ac71ec167d7159c6846a8879e80a15260a708f86f3baabf7d194f9`.
- Runtime was 34,986.868 seconds (about 9 h 43 min), peak CUDA allocation was
  1,218,130,944 bytes, and source provenance remained clean commit
  `7ca7d0d892fec7529fc5193b078c6c9d30cb27b7`.

## A3 formal-run preflight

- Source and documentation were clean at full commit
  `38d68e2f011291dd8e0580bc4206abbd39a81315`. No GPU compute process existed;
  the RTX 4060 reported 846 of 8,188 MiB used and 0% utilization. The process
  table contained only PID 1 and stale completed-run log tails, with no Python
  trainer. `/workspace` had 757 GB available and
  `runs/gcb_ablation_a3_bab_voc_train` was absent.
- Container SHA-256 values matched the host A3 configuration and registered
  MiT-B1 weight: config
  `11127f069b01f0464df539a3974aace591d2267868867f6aaf8aee4a6393354c`;
  weight
  `8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b`.
- Parsed A2/A3 data, training, and inference sections match exactly. The only
  model delta is `use_boundary_branch=false` to true, and the only loss delta
  is `edge_weight=0.0` to `0.2`. A3 retains gated GFF, CE weight 1.0, zero
  Lovasz/auxiliary/label-smoothing weights, official train/val, 20,000 steps,
  and full validation every 1,000 steps.

## A3 reportable launch

- The passed preflight ledger was committed as clean source
  `c3ff587f59396890816dfa6a17fdb7a1ed7bbdfa`. The ignored launch wrapper
  passed `sh -n` and has SHA-256
  `af20a507b75c401ef458531286c9ad010349092e74af3bedb3dd2ad02b64d36b`.
- The reportable A3 job started at 2026-10-27 08:05:24 +08:00 with wrapper
  PID 3246 and trainer PID 3253. Its command injects the exact source commit
  and clean-worktree assertion. GPU use was 2,379 MiB at 54% during the first
  process snapshot.
- Independent early audit found 341 unique sequential train records through
  step 341. Every scalar was finite, all boundary/edge-BCE/edge-Dice losses
  were nonzero, and each total matched `CE + 0.2 * edge`. At the snapshot,
  total/CE/edge were 0.927822/0.804647/0.615879.
- The resolved configuration records A3 gated GFF with BAB, no auxiliary
  head, zero Lovasz/auxiliary/label-smoothing weights, 20,000 reportable steps,
  full validation every 1,000 steps, and no validation limit. Source provenance
  matches; neither `exit_code.txt` nor `result.json` existed.

## A3 user-requested stop

- At the user's request, exact trainer PID 3253 received `SIGTERM`. The trainer
  and wrapper exited, `exit_code.txt` contains 143, no terminal `result.json`
  exists, no Python trainer remains, and the GPU returned to 0% utilization.
- The process had completed train step 3,314 when stopped. The full 3,317-record
  history (3,314 train plus validations at 1,000/2,000/3,000) was preserved at
  `history_archive/user_stop_20261027T094129+0800.jsonl`; SHA-256
  `02f6bb11af54494f12422b79162cb15796a1a6ff7828f2ec93c4238a3495aee1`.
- The active history was reconciled to the latest recoverable checkpoint at
  step 3,000 and now contains 3,003 records: train steps 1-3,000 exactly once
  plus all three validations. Its SHA-256 is
  `0d9b03305fe8d2ecb1af8763ec9e669ac3aa542fe96bef7e90bbd9f7a390217f`.
- `best.pt` is a non-diagnostic step-3,000 checkpoint with mIoU 0.477465;
  its model tensors are finite and SHA-256 is
  `89a3a2af52c6aab49c7c7670d541e0d9fcb36851d163b0e6cccdd1af6af08f26`.
  The matched step-3,000 validation SHA-256 is
  `4ed955a0efbe4e4e2909957c46b00259e41349ce062234a1691ed9c09ac600d9`.
- The machine-readable stop manifest
  `user_stop_20261027T094129+0800.json` independently matched every referenced
  size and hash; its SHA-256 is
  `93aea8ebe4c96b6ab8983ff427fd2b2657e83b443ad9730ccb59c59942796f0c`.
  A3 remains incomplete and must not be compared as a terminal ablation result.

## A3 recovery preflight

- A first read-only source search passed `--resume|resume` without `rg`'s
  option terminator, so `rg` rejected it as an option. The corrected query
  confirmed the trainer's resume path and automatic history reconciliation;
  no file or runtime artifact changed.
- Source and documentation were clean at
  `d6f2218f1b22cd1f7b96eaad9112fba770df39bd`. The RTX 4060 reported 0 of
  8,188 MiB used and 0% utilization with no compute process. Container PID 1
  and the inspection command were the only processes, `/workspace` had 757 GB
  available, and no competing trainer existed.
- The step-3,000 `best.pt`, reconciled history, and stop manifest retained
  their registered SHA-256 values `89a3a2af...`, `0d9b0330...`, and
  `93aea8eb...`. A3 config SHA-256 is `11127f069...`; MiT-B1 remains
  `8944b871...`. Exit 143 is preserved and `result.json` remains absent.
- Recursive checkpoint inspection found 1,068 tensors across model,
  optimizer, and scaler state; all floating tensors are finite. The checkpoint
  is non-diagnostic at step 3,000, selection is mIoU 0.477465 at step 3,000,
  and saved AMP overflow count is zero.
- The resolved configuration exactly matches the checkpoint config. Active
  history contains 3,003 records with train steps 1-3,000 exactly once and
  validations at 1,000/2,000/3,000. The trainer is eligible to resume at step
  3,001 under the unchanged 20,000-step A3 contract.

## A3 recovery launch

- The recovery preflight ledger was committed as clean source
  `c3638045039867aaa77addd8579b32b4d532ba6d`. Before launch, the original
  `console.log` and exit-code 143 were moved into `interruption_archive/`.
  The archived console retained SHA-256
  `e8e1ca010fa18366aab172b70fbd417ae2d98662839800ddc7f654aee7213619`.
- The ignored resume wrapper passed `sh -n` and has SHA-256
  `1588ffcd3ad5d57c6c4091f6c110659b3d6febcbb510489d2b92fdc89034e281`.
  It resumed from `best.pt` at 2026-10-27 12:33:55 +08:00 with wrapper/trainer
  PIDs 97/104 and injected the exact clean recovery source revision.
- The trainer reported `history_reconciliation kept=3003 dropped=0
  archive=None`. Independent audit found train steps 1-3,058 exactly once,
  with the first resumed record at 3,001 and all three earlier validation
  records retained. All 58 observed resumed records were finite, had nonzero
  boundary loss, and matched `total = CE + 0.2 * edge`.
- GPU use was 1,675 MiB at 11% during the first recovery snapshot. The resolved
  runtime remains the 20,000-step non-diagnostic A3 contract, and neither a new
  exit code nor terminal `result.json` existed.

## A3 terminal result

- The resumed wrapper saved exit code 0, `result.json` reports `ok=true` and
  `diagnostic_only=false`, no Python trainer remains, and the terminal GPU
  snapshot was idle at 0% with 1,107 MiB allocated.
- Reconciled history contains exactly 20,026 records: train steps 1-20,000
  exactly once, 20 complete 1,449-image official validations at every 1,000
  steps, and six isolated recovered AMP overflows after resume. Every recorded
  scalar is finite; every train record has nonzero boundary supervision and
  satisfies `total = CE + 0.2 * edge`.
- Independent recomputation from all 20 saved 21x21 global confusion matrices
  reproduced every reported metric. Each validation contains 244,886,857
  valid pixels. Best and terminal are both step 20,000: mIoU 0.726746,
  foreground mIoU 0.715886, pixel accuracy 0.940076, and CE 0.306535.
- A3 is 0.028897 mIoU lower than A2 (-2.890 percentage points), 0.009880 lower
  than A1, and 0.007143 lower than A0 under the common single-seed budget.
  Therefore BAB plus edge loss did not improve segmentation mIoU in A3, and
  A2 remains the best ablation so far.
- The selected non-diagnostic `best.pt` is step 20,000, contains 1,068 finite
  recursively discovered tensors, and has SHA-256
  `41ab47394605b7fc509c97954b1e2f5f4dbc42e1567d145066450a4311cec182`.
  `last.pt` is a finite but stale step-16,000 non-improvement artifact, not the
  terminal checkpoint; its SHA-256 is
  `69dc2e3f5bb6a45d23a55a832ae69d8689c8f15bb6730d0a6446c9a1f30cbad1`.
- TensorBoard was independently verified across both sessions. The initial
  event contains raw train steps 1-3,314 and validations through 3,000; only
  steps 1-3,000 are accepted before the saved interruption. The resumed event
  contains unique steps 3,001-20,000, validations 4,000-20,000, and six extra
  AMP events. Event SHA-256 values are
  `929e3a371265a63487c30be6b3eaa6e48ab4e7300c0839c2713623fd610016cc`
  and `de67ca5ce6749a2db8bb4471808f28e536d4bee6ddcec20da95eb0b63c544ccd`.
- Result/history/terminal-validation SHA-256 values are
  `85a691f89401c9dbad15f645b1b8a0e328698598cf612f079bed5794c2d58263`,
  `9244f4f02ce6ec70b9b292a7027627bd084b385614f4cd7467f56df64dd7edd5`,
  and `0474418b4fc0b2bcb36aed2d754c8f64b6353309bc9f51777af56e3ef3058cb1`.
  The stop manifest and original 3,317-record archive retained their verified
  hashes `93aea8eb...` and `02f6bb11...`.
- The resumed segment ran for 30,096.928 seconds (about 8 h 22 min) and peaked
  at 1,350,610,432 allocated CUDA bytes. Original source was `c3ff587...` and
  clean resume provenance was `c3638045039867aaa77addd8579b32b4d532ba6d`.

## A4 formal-run preflight

- Source and documentation were clean at full commit
  `6c8cfcb13261438c871aec992d1d0270bba3c0c5`. No GPU compute process existed;
  the RTX 4060 reported 1,107 of 8,188 MiB used and 0% utilization. The process
  table contained only PID 1, a stale A3 log tail, and the inspection command,
  with no Python trainer. `/workspace` had 757 GB available and
  `runs/gcb_ablation_a4_lovasz_voc_train` was absent.
- Container SHA-256 values matched the host A4 configuration and registered
  MiT-B1 weight: config
  `f1d5532abf381ea2d31c7a60d84cc5d7e98def6fdd7553b43207b41303c0b596`;
  weight
  `8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b`.
- Parsed A3/A4 model, data, training, and inference sections match exactly.
  The only loss delta is `lovasz_weight=0.0` to `0.5`. A4 retains gated GFF,
  BAB, CE weight 1.0, edge weight 0.2, zero auxiliary/label-smoothing weights,
  official train/val, 20,000 steps, and full validation every 1,000 steps.

## A4 reportable launch

- The passed preflight ledger was committed as clean source
  `fb2b376e43a04f5f351f83b33bfcb71c8349bbfb`. The ignored launch wrapper
  passed `sh -n` and has SHA-256
  `b6cefef816e4d97ef9138d669d0a21b4ad148cccb9e9742ab192d77ac3dd185b`.
- The reportable A4 job started at 2026-10-27 21:56:48 +08:00 with wrapper
  PID 409 and trainer PID 416. Its command injects the exact source commit and
  clean-worktree assertion. GPU use was 2,642 MiB at 11% during the first
  process snapshot.
- Independent early audit found 52 unique sequential train records through
  step 52. Every scalar was finite, Lovasz and all boundary-loss components
  were nonzero, and each total matched
  `CE + 0.5 * Lovasz + 0.2 * edge`. At the snapshot, total/CE/Lovasz/edge were
  2.211768/1.633973/0.829992/0.813998.
- The resolved configuration records A4 gated GFF with BAB, no auxiliary
  head, 20,000 reportable steps, full validation every 1,000 steps, and no
  validation limit. Source provenance matches; neither `exit_code.txt` nor
  `result.json` existed.

## A4 user-requested pause

- A first read-only checkpoint loop embedded `$f` inside a PowerShell double-
  quoted command. PowerShell expanded it before Docker, causing both iterations
  to list the run directory instead of individual checkpoint paths. Explicit
  path checks then confirmed `best.pt` exists and `last.pt` is absent; no state
  changed during the failed query.
- At the user's request, exact trainer PID 416 received `SIGTERM`. The trainer
  and wrapper exited, `exit_code.txt` contains 143, no terminal `result.json`
  exists, no Python trainer remains, and the GPU returned to 0% utilization.
- Training had completed step 1,788 when paused. The full 1,789-record history
  was preserved at `history_archive/user_pause_20261027T225706+0800.jsonl`;
  it contains train steps 1-1,788 and the step-1,000 validation. SHA-256 is
  `8d017947dccbbea86eafa66a197846b4f20fb562815da6b72e823f4234477e14`.
- Active history was reconciled to checkpoint step 1,000 and now contains
  exactly 1,001 records: train steps 1-1,000 plus the complete validation. Its
  SHA-256 is
  `384042982302d63051b296e712adcbfdda972023afe67750f64e5d2c0fc93324`.
- `best.pt` is a non-diagnostic step-1,000 checkpoint with mIoU 0.188642;
  all 1,068 recursively discovered floating tensors are finite and SHA-256 is
  `35aa3e4d122ced56851905066e26a69b5551b365776fd2528201268bf7e0ce6d`.
  The matched validation SHA-256 is
  `f31b5c64f4afe3cebc505e9e436f42733f3958cd42153898d8090f5f8c4856e8`.
- The machine-readable pause manifest
  `user_pause_20261027T225706+0800.json` matched every referenced size and hash;
  its SHA-256 is
  `22826e4013ea50faa31393007cd5fead8eb49f501a75a271dc8bb482c1950d5b`.
  A4 remains incomplete and must not be compared as a terminal result.

## A4 recovery preflight

- The first container preflight invocation was denied by the Windows Docker
  named pipe and started nothing. The approved retry reached the read-only
  audit, but its temporary checker incorrectly searched history key `event`
  instead of the actual key `type`; checkpoint loading and all hash checks had
  passed before that local assertion failed. Correcting only the checker made
  the complete audit pass.
- Source remains clean at `c8ebb4ee9304bd44bdadc0c01e8a7e740495e0fb`.
  The process table contains only container PID 1 and the inspection command,
  with no trainer. The RTX 4060 reports 120/8,188 MiB and 0% utilization, no
  compute process, and `/workspace` has 757 GB available.
- Registered SHA-256 values for the step-1,000 checkpoint, active and archived
  histories, validation, resolved config, paused console, pause manifest, A4
  config, and MiT-B1 weight all match. Exit 143 remains saved; `result.json`
  and `last.pt` remain absent.
- The checkpoint is non-diagnostic at step 1,000 with mIoU 0.188642. Its
  configuration exactly matches `resolved_config.yaml`, and all 1,068
  recursively inspected floating tensors are finite. Active history contains
  train steps 1-1,000 exactly once plus the step-1,000 full validation; every
  train record is finite and satisfies
  `total = CE + 0.5 * Lovasz + 0.2 * edge`.

## A4 recovery launch

- Three ordinary post-preflight Docker inspection batches were denied by the
  Windows named pipe; approved retries succeeded. Neither denial stopped or
  duplicated the background job. The paused console and exit 143 were first
  preserved in `interruption_archive/` with SHA-256 values `afe8c95c...` and
  `9d9b1872...`.
- The resume wrapper passed `sh -n`, has SHA-256
  `255e60000b4a0e1d6c8c684c4c3c23635a0cb273f24f745465f786a26a72fd8f`,
  and launched at 2026-10-28 00:00:55 +08:00 under clean source
  `f51af0720b3ebcad9a6cd9cda40f047cafc3d622`. Wrapper/trainer PIDs are
  125/132.
- Built-in reconciliation reported `kept=1001 dropped=0 archive=None`.
  Independent audit at step 1,070 found train steps 1-1,070 exactly once, the
  first resumed step exactly 1,001, 70 finite resumed records satisfying
  `total = CE + 0.5 * Lovasz + 0.2 * edge`, and the original step-1,000 full
  validation retained.
- GPU use was 1,794/8,188 MiB at 21%. No new exit code or terminal result
  exists. A later approved handoff snapshot found the same PIDs active at step
  1,139 with finite values and still no terminal artifacts; A4 is actively
  continuing toward 20,000 steps.

## A4 terminal result and ablation selection

- The first ordinary terminal snapshot was denied by the Docker named pipe;
  its approved retry confirmed exit code 0, no trainer, GPU 0/8,188 MiB at 0%,
  and a terminal result. The first two temporary audit versions were also
  rejected: one used the wrong AMP-overflow scale keys, and one assumed the
  TensorBoard overflow tag contained only overflow events rather than the
  regular per-step zero series plus six duplicate overflow points. Correcting
  only those audit assumptions produced a complete pass.
- A4 contains 20,000 unique sequential train records, 20 full 1,449-image
  official validations, and six bounded AMP recoveries in 20,026 history
  records. Every completed scalar is finite, every train record has nonzero
  Lovasz and boundary terms, and every total matches
  `CE + 0.5 * Lovasz + 0.2 * edge`.
- Independent recomputation of all 20 saved 21x21 confusion matrices matched
  every reported per-class IoU, mIoU, foreground mIoU, pixel accuracy, and
  mean pixel accuracy. Best and terminal are step 20,000: mIoU 0.739100,
  foreground mIoU 0.728806, pixel accuracy 0.943025, validation CE 0.319098.
- A4 improves A3 by 0.012354 mIoU (+1.235 percentage points), A1 by 0.002474,
  and A0 by 0.005212, but remains 0.016543 below A2 (-1.654 points). Under the
  common single-seed protocol, A2 remains the validation-selected ablation for
  milestone 06.
- A4 `best.pt` is non-diagnostic at step 20,000, has 1,068 finite tensors, and
  SHA-256 `feec1bc099d979f2dde054bab406c8ba62dd3834781dbf75618d0ec3b7e4bf26`.
  `last.pt` is the finite step-19,000 checkpoint with SHA-256
  `6822a3765523af5a8b82458b653dfd89cf2325a52164e58f1f2cbebc83eaf027`.
- SHA-256 values: result `fda20e6a...`, history `5a4c54e4...`, terminal
  validation `e0fd2e4e...`, initial/resumed TensorBoard events `70cc6e6e...`
  and `39d0c649...`. The recovered segment elapsed 28,090.259 seconds and
  peaked at 1,350,610,432 allocated CUDA bytes.
- `artifacts/gcb_diagnostics.json` consolidates verified A0-A4 evidence,
  selects A2 only by saved official-validation mIoU, and has SHA-256
  `0011ddfd4caeebaa6ede17a8a7e7b2da8fdee4fe379b26762a44af157ffd1874`.
  The project-status checker now reports milestone 05 complete and milestone
  06 as the next stage.

## Selected A2 trainaug configuration

- Added `configs/gcb_selected_a2_trainaug.yaml`, SHA-256
  `5fcbfa7cfdc9250dc744509e4fe58caa18d8c11864b7b5a0595c6d3f6b7a114c`.
  Its model and loss sections exactly equal A2: MiT-B1/384, gated GFF, no BAB,
  auxiliary head, edge loss, or Lovasz. It uses the plan's initial 40,000-step
  trainaug budget and validates the untouched official val every 1,000 steps.
- The shared SegFormer/GCB trainer now resolves trainaug inputs, verifies the
  configured manifest, conversion summary, and SBD provenance SHA-256 values
  before constructing the dataset, rejects a failed/leaking audit, and records
  source, license, hashes, and trainaug identity evidence in `result.json`.
- Focused config/trainer/dataset tests passed 13/13 in 0.965 seconds. The full
  required container suite passed 50/50 in 5.058 seconds.
- The first actual-data preflight failed before reading data because its
  temporary script did not add the application root to `sys.path`; no training
  process or output was created. The corrected read-only preflight loaded all
  10,582 unique samples (1,464 VOC and 9,118 SBD), found zero official-val
  overlap, and successfully decoded one VOC and one SBD image/mask with only
  labels 0-20/255.

## Selected A2 trainaug CUDA diagnostic

- The first ordinary Docker resource query was denied by the Windows named
  pipe and started nothing; approved access confirmed no trainer, GPU
  113/8,188 MiB at 0%, no compute PID, and 757 GB available.
- The diagnostic completed one effective-batch-8 optimizer step on the real
  10,582-sample trainaug dataset. Total/CE loss was 3.193626, gradient norm
  8.631807, encoder/decoder LR 0.00006/0.0006, and peak allocated CUDA memory
  1,100,638,720 bytes. Edge and auxiliary losses were zero as required by A2.
- Independent audit verified one train and one two-image validation record,
  exact CE-only weighting, complete provenance hashes, and a diagnostic-only
  step-1 checkpoint containing 1,036 finite tensors. Checkpoint/result/history
  SHA-256 values are `c6ede983...`, `0f75f5f2...`, and `5ae71a43...`.
  The two-image mIoU is diagnostic and non-reportable.

## Selected A2 trainaug formal-run preflight

- The first ordinary final-preflight Docker batch was denied by the named pipe;
  its approved retry passed. Source was clean at
  `2420f7890b2ced7bb1d4d47d95701d8b72ee1e57`; no Python trainer or GPU compute
  process existed, GPU was 113/8,188 MiB at 0%, and `/workspace` had 757 GB
  available.
- `runs/gcb_selected_a2_trainaug` is absent. Config, selection summary,
  trainaug manifest, SBD archive, official val split, and MiT-B1 weight hashes
  all match their registered values. The parsed contract selects A2, uses
  10,582 leak-free trainaug samples, preserves official val, and specifies
  40,000 steps with full validation every 1,000.

## Selected A2 trainaug reportable launch

- Ordinary Docker access was denied once while validating the launch wrapper
  and once during the first post-launch observation; approved retries passed.
  The background launch command was issued exactly once.
- The ignored wrapper passed `sh -n` with SHA-256
  `675914feb7689aa2165fa519da8d03c0733e0dfee641b8a5d0b284a746f75c5b`.
  It launched at 2026-10-28 08:24:15 +08:00 from clean source
  `ad026dba623eae384046aa7be5abdb6b341252e5`, with wrapper/trainer PIDs
  784/792 and explicit clean-worktree assertion.
- Independent early audit found 43 unique sequential train steps beginning at
  1, zero AMP overflows, finite values, and exact CE-only totals; latest loss
  was 2.236667 and gradient norm 6.605145. Resolved runtime is 40,000 steps,
  validation every 1,000, full official val, trainaug, and non-diagnostic mode.
- GPU use was 1,523/8,188 MiB at 38%. No exit code, validation, checkpoint, or
  terminal result existed. A later local log snapshot reached step 54 with
  finite loss 2.086697 and gradient norm 4.833095.

## Selected A2 trainaug user-requested stop

- Three ordinary Docker observation batches around the stop were denied by the
  Windows named pipe; approved retries succeeded. Exact trainer PID 792 then
  received SIGTERM once. Wrapper/trainer processes exited, exit code 143 was
  saved, no terminal result exists, and no GPU compute process remains.
- Training reached step 1,066. The complete 1,067-record pre-stop history was
  archived at `history_archive/user_stop_20261028T085659+0800.jsonl`, SHA-256
  `ced177181b46a9f58218ac958b1d66ea68dd107680445e8902c474c24cba397d`.
  It contains train steps 1-1,066 plus validation step 1,000.
- Active history was reconciled to exactly 1,001 records through the saved
  checkpoint: train steps 1-1,000 and the step-1,000 validation. Its SHA-256 is
  `abb2bc8cfa362db3875ed0b3ef67cec1c4deb1ac9f6a23230f302626d47892ee`;
  66 uncheckpointed train records remain only in the full archive.
- The complete 1,449-image step-1,000 validation reports mIoU 0.154537,
  foreground mIoU 0.118020, pixel accuracy 0.809402, and CE 0.715655. This is
  an early restart point, not a terminal model-selection result.
- Non-diagnostic `best.pt` is step 1,000, contains 1,036 finite tensors, and
  has SHA-256
  `8005ad74e9c42153d4eaa938967fb951034a5e8a6dec166f3454a4463e67f98d`.
  Validation SHA-256 is `08bf2323...`; no `last.pt` exists.
- The machine-readable stop manifest is
  `user_stop_20261028T085659+0800.json`, SHA-256
  `cf208b9ac441dc1228fc3ce8a67125e46047f92ab146a554193587a97f056deb`.
  A future run must resume from step 1,000 and continue uniquely at 1,001.
- Three read-only selected-A2 recovery checker versions were rejected for
  assumptions about the resolved-config runtime block, trainaug-summary key,
  and nested learning-rate fields. No trainer started and no reportable
  artifact changed during these corrections.
- The corrected step-1,000 recovery preflight passed: no trainer or GPU compute
  process exists, GPU is 0/8,188 MiB at 0%, and `/workspace` has 756 GB free.
  All registered checkpoint/history/config/data/weight/stop hashes match;
  exit 143 is preserved, `result.json` and `last.pt` are absent, all 1,036
  checkpoint tensors are finite, and active history is uniquely reconciled
  through step 1,000. The run is eligible to resume at step 1,001.
- The passed preflight ledger was committed as clean source
  `80555c803dae4baa1b3e6ff5b6e4378ff20fe1ca`. Original console, exit 143,
  start/source markers, and launch wrapper were preserved in the interruption
  archive with their registered hashes before creating new live artifacts.
- One ordinary final Docker inspection was denied by the named pipe; its
  approved retry passed. The resume wrapper passed shell syntax validation,
  has SHA-256 `0fcca8f8...`, and launched exactly once at 2026-10-28 12:56:06
  +08:00 with wrapper/trainer PIDs 193/201.
- Built-in reconciliation kept 1,001 records, dropped none, and created no
  extra archive. Independent audit at step 1,032 verified unique steps through
  1,032, exact first continuation at 1,001, 32 finite CE-only resumed records,
  and the retained step-1,000 validation. GPU was 1,547/8,188 MiB at 5%; no
  exit code or terminal result exists, so training is active.
- At the user's pause request, exact trainer PID 201 received one `SIGTERM` and
  exited with wrapper code 143 after completing train step 13,478. No trainer
  or GPU compute process remains; GPU is 0/8,188 MiB at 0%, and no terminal
  result exists.
- The first save finalizer successfully archived/reconciled history but then
  rejected a nonexistent validation `step` key; it wrote no pause manifest.
  The corrected idempotent finalizer completed using the actual validation
  schema, and an independent verification passed all artifact hashes.
- Full pre-pause history has 13,494 records through train step 13,478 and is
  archived with SHA-256 `ca185957...`. Active history has 13,016 records
  through restart step 13,000, SHA-256 `ecb18e96...`; it retains 13 complete
  validations and three bounded AMP-overflow records.
- Non-diagnostic `last.pt` at step 13,000 is the restart checkpoint, SHA-256
  `b227903d...`, with 1,036 finite tensors. Its full-val mIoU is 0.671776.
  Validation-selected `best.pt` remains step 12,000 at mIoU 0.692692,
  SHA-256 `fa3b628f...`, also with 1,036 finite tensors.
- Pause manifest `user_pause_20261028T181049+0800.json` has independently
  verified SHA-256 `af8155e2...`. A future recovery must start from `last.pt`
  at step 13,000 and continue uniquely at step 13,001.
- Two recovery audit attempts failed before container access because Docker
  Desktop was not running. The first launch path checked was stale; the actual
  per-user installation path was found from the uninstall registry, Docker
  Desktop was started there, and server version 29.4.3 became ready. No trainer
  started during these failures.
- The corrected step-13,000 recovery preflight passed with GPU 0/8,188 MiB at
  0%, no trainer or compute process, and 756 GB free. All registered pause,
  checkpoint, history, validation, config, data, and weight hashes match.
- Both non-diagnostic checkpoints contain 1,036 finite tensors. `last.pt` is
  the step-13,000 restart point; `best.pt` remains step 12,000 at mIoU
  0.692692. Active history is unique through step 13,000 and the full pause
  archive remains intact through train step 13,478. Continuation is eligible
  to begin exactly at step 13,001.
- The preflight ledger was committed as clean source
  `fd1f6d3096e048c53ea1c8b4f4e67080ba79e156`. Pause console/exit evidence and
  the prior resume wrapper were archived with unchanged SHA-256 values before
  creating new live artifacts.
- The step-13,000 resume wrapper explicitly uses `last.pt`, passed shell syntax
  validation, has SHA-256 `8c42cd24...`, and launched once at 2026-10-28
  18:48:13 +08:00 with wrapper/trainer PIDs 71/79.
- Automatic reconciliation kept 13,016 records, dropped none, and created no
  archive. Independent audit through step 13,020 verified exact first
  continuation at 13,001, unique train steps, retained 13 validations and
  three prior overflow events, and 20 finite CE-only new records. GPU use was
  1,547/8,188 MiB at 40%; no exit code or terminal result exists.
