# Decision 0003: Plan permissions and artifact provenance

- Time: 2026-10-24 17:53 +08:00
- Status: Active

The user-designated V1.0 plan states that ImageNet classification pretraining,
SBD, multiscale/flip testing, and model ensembles are allowed. Before any
external data or weight enters a reportable run, record its source, scope,
relevant license, and SHA-256. TTA and ensembles enter final inference only
after improving validation mIoU.
