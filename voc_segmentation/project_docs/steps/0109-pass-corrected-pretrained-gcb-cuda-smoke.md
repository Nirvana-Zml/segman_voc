# Step 0109: Pass corrected pretrained GCB CUDA smoke

- Time: 2026-10-26T08:52:25+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

The corrected deterministic 384x384, physical-batch-2 GCB-Seg smoke passed a complete pretrained AdamW and AMP optimizer step with all composite loss components, finite gradients, stable scaling, and exact parameter groups.

## Commands

- `docker exec voc2012-gcb python tools/smoke_gcb_seg.py`

## Changed files

- `voc_segmentation/tools/smoke_gcb_seg.py; ignored voc_segmentation/artifacts/gcb_diagnostics/smoke.json`

## Evidence

- `Total/CE/Lovasz/edge/aux losses 4.829429/2.930575/0.887470/0.843542/3.216027; 2474/234318 positive boundaries; gradient norm 12.056706; AMP scale 1024 to 1024; peak CUDA 1078124544 bytes; artifact SHA-256 62ba3dfeddcddffd4dfc79a88186b84a4a8a3010b11ce702ae73c3a33de3998e.`

## Decisions and assumptions

- Accept this as diagnostic-only smoke evidence, not a reportable model result or checkpoint.

## Next action

Independently audit the saved JSON, configured loss formula, hashes, output shapes, and optimizer parameter coverage.
