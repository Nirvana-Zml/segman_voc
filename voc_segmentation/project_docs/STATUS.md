# Project status

- Last updated: 2026-10-24 20:05 +08:00
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
| 03 | CNN baseline | Not started | DeepLabV3+ smoke, overfit, formal validation |
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

Implement the shared VOC/trainaug dataset and DeepLabV3+ baseline interfaces,
then pass focused tests, a smoke run, and a tiny-set overfit diagnostic before
starting a reportable CNN baseline run.

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
