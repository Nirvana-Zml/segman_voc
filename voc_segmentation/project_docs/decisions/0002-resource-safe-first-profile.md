# Decision 0002: Use an 8 GB-safe first profile

- Time: 2026-10-24 17:53 +08:00
- Status: Active

The available RTX 4060 Laptop GPU is an 8 GB-class device. The first standalone
Transformer and GCB-Seg profiles therefore use MiT-B1, a 384×384 crop, physical
batch size 2, AMP, and accumulation to effective batch size 8. MiT-B2/512 is a
target configuration only after a saved memory profile passes.
