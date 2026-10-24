# Project status

- Last updated: 2026-10-25 00:13 +08:00
- Overall state: **In progress**
- Active application root: `voc_segmentation/`
- Design authority: `VOC2012_Semantic_Segmentation_Project_Plan.md` V1.0
- Highest verified active milestone: **02 — VOC audit and SBD**
- Next milestone: **03 — CNN baseline**
- First hardware-safe model profile: MiT-B1, 384×384, batch 2, effective batch 8

## Milestone summary

| ID | Milestone | State | Required next evidence |
| --- | --- | --- | --- |
| 00 | Project reset | Passed | New driver valid; ledger and standalone root created |
| 01 | Environment and source | Passed | Clean Docker build and CUDA allocation report |
| 02 | VOC audit and SBD | Passed | Official audit plus provenance-verified, leak-free 10,582-image trainaug |
| 03 | CNN baseline | In progress | Smoke and two-sample overfit passed; formal training/validation pending |
| 04 | Transformer baseline | Not started | Comparable SegFormer validation |
| 05 | GCB diagnostics and ablations | Not started | Interface/loss tests, smoke, overfit, controlled ablations |
| 06 | Formal training and selection | Not started | trainaug run and validation-selected checkpoint |
| 07 | test400 inference | Not started | Exactly 400 validated masks |
| 08 | Report and package | Not started | Evidence-backed report and final archives |

## Current facts and blockers

- The user deleted `SegMAN-main/`; 438 tracked files are intentionally pending
  deletion and must not be restored implicitly.
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

Launch the reportable DeepLabV3+ VOC-train baseline with the verified formal
trainer and evaluate all 1,449 official validation images. Diagnostic weights
remain ineligible.

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
