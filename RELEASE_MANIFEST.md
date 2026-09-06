# Public release manifest

Version: `v1.0.0`

## Included

- Frozen source code and configurations.
- Fifteen canonical recurrent-model checkpoints (three model families,
  five seeds).
- Five KalmanNet checkpoints.
- Final 41-input ONNX model and training-split normalization parameters.
- Compact 51-file supporting dataset and verifier.
- PC/Raspberry Pi HITL acquisition and alignment software.

## Excluded from Git

- Full acquisition JSON, training CSV and processed NumPy arrays.
- Full experiment predictions, Malolo source records and ten-session HITL
  logs (published in the unified data record).
- Intermediate checkpoints and training logs.
- Manuscript drafts, reviewer/editor correspondence and literature PDFs.

## Release gates

- [x] Public GitHub URL resolves.
- [x] Software DOI resolves.
- [x] Unified data DOI resolves.
- [x] `CHECKSUMS.sha256` passes in a clean clone.
- [x] Minimal dataset reports 302 checks and zero differences.
- [x] No absolute local paths, private hosts or submission correspondence.

Unified-data DOI: `10.5281/zenodo.22338389`.
Software DOI: `10.5281/zenodo.22463526`.
