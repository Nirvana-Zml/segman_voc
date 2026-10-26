# Step 0106: Verify GCB-Seg interfaces and composite losses

- Time: 2026-10-26T08:39:53+08:00
- Milestone: `05-gcb-diagnostics-and-ablations`
- Status: **Passed**

## Objective and outcome

Corrected the Lovasz gradient, passed all focused and repository tests, strictly loaded the accepted MiT-B1 encoder, and verified the complete GCB-Seg parameter and normalization contracts.

## Commands

- `Run corrected focused 13-test suite; run full unittest discovery; strictly load MiT-B1 and count parameters and normalization layers; git diff --check`

## Changed files

- `models/gcb_seg.py; models/modules/gcb.py and exports; losses/gcb.py and exports; gcb_seg_b1_384.yaml; GCB tests; README; STATUS and milestone documentation`

## Evidence

- `Focused 13 of 13 passed in 2.986 seconds; full 40 of 40 passed in 5.282 seconds; 15841707 total parameters, 13151424 encoder, 2690283 decoder, 28 GroupNorm, 0 BatchNorm; strict pretrained load passed.`

## Decisions and assumptions

- Accept interface and loss gates only; require a saved CUDA optimizer-step smoke and two-sample overfit before any ablation metric is reportable.

## Next action

Implement and run the provenance-gated pretrained GCB-Seg CUDA smoke at 384 by 384 and batch 2.
