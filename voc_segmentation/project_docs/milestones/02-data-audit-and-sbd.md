# Milestone 02: VOC audit and SBD preparation

- State: **Passed**
- Last updated: 2026-10-24 20:05 +08:00

Audit official VOC train/val and test400, then acquire and convert SBD with
source/hash provenance. Preserve official labels, exclude every val ID from
`train_aug`, and save split/class/overlay evidence.

## Official-data evidence

- `artifacts/data_check/split_summary.json`: `ok=true`
- train: 1,464; val: 1,449; labeled samples checked: 2,913
- test400: 400 readable images with unique stems
- train/val overlap: 0; train duplicates: 0; val duplicates: 0
- missing images/masks, unreadable files, size mismatches, illegal labels: 0
- 20 deterministic overlay images generated with seed 42
- class pixel/image counts saved in `class_pixel_statistics.csv`
- empty `invalid_samples.txt`
- train ID SHA-256:
  `aa623bd2c8ce4443a8aaae51c524c0eb165e8e44caf6aa9e3ad33d7b75a3ef20`
- val ID SHA-256:
  `cdc1326d12f69ce5153aa5da04a4d8783e146868d42d19f82f44e74b97ac907d`
- test filename SHA-256:
  `133dcd2e68b1c1b4d9480c9d85946e3af203e86e48cd9b04eb8289aea1e3aafd`

## Verification history

Two host test runs failed because the managed Windows sandbox denied dynamic
temporary-directory writes. The same six focused tests passed inside the
target `voc2012-gcb` container in 2.384 seconds.

The first synthetic SBD MAT fixture was malformed because its boundary cells
were uninitialized. After initializing all 20 cells, the full in-container
suite passed all 7 tests in 1.567 seconds. The final post-documentation run
also passed all 7 tests in 2.099 seconds.

The first conversion against the Windows bind mount was interrupted after
694/9,118 masks because small-file I/O was too slow. The same converter then
ran entirely in container-local `/tmp` storage and its verified output was
copied back to the host artifact directory.

## SBD and trainaug evidence

- Official source:
  `https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz`
- Archive size: 1,419,539,633 bytes
- MD5: `82b4d87ceb2ed10f6038a1cba92111cb` (matches the published checksum)
- SHA-256:
  `6a5a2918d5c73ce032fdeba876574d150d9d04113ab87540a1304cbcc715be53`
- Citation: Hariharan et al., *Semantic Contours from Inverse Detectors*,
  ICCV 2011
- License note: the archive README requests citation but contains no separate
  license file; SBD use is explicitly permitted by the current project plan
- Extracted SBD: 11,355 images and 11,355 class-label MAT files
- `artifacts/sbd/provenance.json`: `ok=true`
- `artifacts/sbd/trainaug_summary.json`: `ok=true`
- trainaug: 10,582 unique samples (1,464 VOC train + 9,118 SBD extras)
- official validation overlap: 0; duplicate IDs: 0
- invalid labels: 0; image/mask dimension mismatches: 0; conversion errors: 0
- host output: 9,118 converted PNG masks and 10,582 manifest/ID lines
- trainaug ID SHA-256:
  `d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24`

## Next action

Begin milestone 03 with the DeepLabV3+ CNN baseline: implement the common VOC
dataset/training interfaces, then run unit tests, a smoke run, and a tiny-set
overfit diagnostic before formal validation.
