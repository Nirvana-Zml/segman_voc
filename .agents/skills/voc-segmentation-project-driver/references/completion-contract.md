# Completion contract

Use this reference when selecting the next gate, accepting evidence, writing
the report, or claiming completion.

## Stable facts

- Application root: `voc_segmentation/`
- Container root: `/workspace/voc_segmentation`
- Docker image name used by project commands: `voc2012-gcb:dev`
- Host VOC root: `data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012`
- Container VOC root: `/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012`
- Host/container test400 roots: `data/test400/test400/images` and
  `/data/test400/test400/images`
- Current GPU class: RTX 4060 Laptop, about 8 GB
- Environment target: Python 3.10, PyTorch 2.1.2, torchvision 0.16.2,
  CUDA 12.1, plus pinned `requirements.txt`
- Primary design: GCB-Seg; required baselines: DeepLabV3+ and SegFormer
- Current plan permits ImageNet classification pretraining, SBD, TTA, and
  ensembles, subject to artifact provenance.

## Gates and sufficient evidence

| Gate | Pass evidence |
| --- | --- |
| Documentation | Current status, milestones, decisions, and an immutable record for every performed step. |
| Environment | A clean Docker build, pinned-version report, focused tests, and a CUDA allocation check pass. |
| Source contract | Configs and tests enforce 21 classes, ignore 255, no label reduction, independent validation, and mIoU selection. |
| VOC data | Saved audit proves train=1464, val=1449, no overlap/missing files/invalid labels, and test400=400. |
| SBD | Source/license/hash are recorded; conversion preserves labels; `train_aug` excludes all official validation IDs and has no duplicates. |
| CNN baseline | DeepLabV3+ completes smoke, overfit, formal training, validation, and saves reproducible artifacts. |
| Transformer baseline | SegFormer uses the same evaluator and split and saves comparable validation evidence. |
| GCB-Seg diagnostics | Output/loss tests, one-step smoke, and two-sample overfit pass before formal training. |
| Ablations | GCM, GFF, BAB, and Lovasz changes are isolated and compared under the same split/evaluator. |
| Formal selection | Best checkpoint and any TTA/ensemble are selected only by saved validation mIoU. |
| Predictions | Exactly 400 original-size, single-channel `uint8` PNG masks contain only IDs 0..20 and pass a saved validation report. |
| Report/package | Every claim traces to artifacts; real student naming is used; archives contain only intended deliverables. |

File presence is structural evidence only. A gate passes only when required
contents and terminal command results are checked.

## Experiment integrity

- Keep official validation independent until all architecture, loss, training,
  TTA, and ensemble choices are fixed.
- Record exact resolved config, seed, artifact hashes, GPU, versions, crop,
  physical/effective batch, optimizer, schedule, elapsed time, and peak memory.
- Treat MiT-B2/512 as unavailable until a saved 8 GB memory profile passes.
- Label incomplete or interrupted runs accurately.
- Never infer a test mIoU from test400 predictions.
