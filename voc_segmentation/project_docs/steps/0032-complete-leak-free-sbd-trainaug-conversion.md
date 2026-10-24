# Step 0032: Complete leak-free SBD trainaug conversion

- Time: 2026-10-24T19:53:15+08:00
- Milestone: `02-data-audit-and-sbd`
- Status: **Passed**

## Objective and outcome

Converted all 9,118 SBD-only masks in container-local storage, built the 10,582-sample trainaug manifest, validated every conversion invariant, copied the outputs to the host, and verified host counts.

## Commands

- `docker cp data/sbd/benchmark.tgz voc2012-gcb:/tmp/benchmark.tgz`
- `docker exec voc2012-gcb python tools/prepare_voc_sbd.py --sbd-root /tmp/sbd_fast/benchmark_RELEASE/dataset --archive /tmp/benchmark.tgz --output-dir /tmp/sbd_output`
- `docker cp voc2012-gcb:/tmp/sbd_output/. voc_segmentation/artifacts/sbd`

## Changed files

- `voc_segmentation/artifacts/sbd/provenance.json (gitignored runtime evidence)`
- `voc_segmentation/artifacts/sbd/trainaug_summary.json (gitignored runtime evidence)`
- `voc_segmentation/artifacts/sbd/train_aug.jsonl (gitignored runtime artifact)`
- `voc_segmentation/artifacts/sbd/train_aug.txt (gitignored runtime artifact)`
- `voc_segmentation/artifacts/sbd/SegmentationClassAug/*.png (9118 gitignored masks)`
- `voc_segmentation/project_docs/milestones/02-data-audit-and-sbd.md`
- `voc_segmentation/project_docs/STATUS.md`

## Evidence

- `trainaug=10582; VOC train=1464; SBD extras=9118; converted masks=9118; official-val overlap=0; duplicates=0; invalid labels=0; dimension mismatches=0; errors=0`
- `trainaug ID SHA256=d1326bd532648d73bc4b1bd275434eba81930982dc7401a65a8cecb26c028e24`
- `Host masks=9118; train_aug.jsonl lines=10582; train_aug.txt lines=10582`

## Decisions and assumptions

- The runtime output remains gitignored; versioned docs retain commands, provenance, hashes, counts, and artifact paths.

## Next action

Start milestone 03 by implementing the shared dataset/training interfaces and DeepLabV3+ baseline diagnostics.
