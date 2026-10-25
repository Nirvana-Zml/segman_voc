# Step 0080: Acquire and verify official MiT-B1 ImageNet weights

- Time: 2026-10-26T00:12:14+08:00
- Milestone: `04-transformer-baseline`
- Status: **Passed**

## Objective and outcome

Pinned NVIDIA Hugging Face revision 13ddceec, verified the official LFS size and full SHA-256, converted the encoder to all 176 native MiT keys, and passed strict load, finite-tensor, finite-forward, and idempotent rerun checks.

## Commands

- `Run acquire_mit_b1_weights.py and independently verify source, converted, and provenance hashes`

## Changed files

- `tools/acquire_mit_b1_weights.py; gitignored source and converted checkpoints; gitignored provenance.json`

## Evidence

- `Source 54732285 bytes SHA-256 980b86b60db37b1b1528086f6c53d253d879e9e09ebe07397b40fd275738a3bf; converted 52658798 bytes SHA-256 8944b871fc4379ddc9c72df16daeea22c53ed5eac3ce65d7219d967d77a9ee8b; 194 source tensors to 176 target tensors; strict load and finite forward passed; provenance SHA-256 2bf52c96aeb967605d4fe9975497c861cf3ee824bf62e6c892c0de3d688cf6ca`

## Decisions and assumptions

- Accept the converted checkpoint for non-commercial educational evaluation under the NVIDIA Source Code License for SegFormer and applicable ImageNet terms.

## Next action

Add conversion regression coverage and a pretrained SegFormer CUDA smoke.
