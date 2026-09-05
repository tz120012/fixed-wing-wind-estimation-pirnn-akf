# Fixed-Wing 3D Wind Estimation with PI-GRU and AKF

This repository is the public software release for the associated *Drones*
manuscript. It contains the frozen 41-input PI-GRU/PIRNN-AKF pipeline,
parameter-matched recurrent baselines, KalmanNet comparison, PX4/JSBSim data
generation, zero-shot Rascal-to-Malolo evaluation, AKF sensitivity analysis,
ONNX deployment, and Raspberry Pi 5 + CUAV V5+ HITL tooling.

## Scope

- Five fixed seeds: 26, 42, 2024, 2025 and 2026.
- Input: 41 state channels, 100 time steps, 50 Hz nominal sampling.
- Final learned methods: Vanilla GRU, Vanilla LSTM, PI-GRU and KalmanNet.
- Post-processing: causal measurement-noise-adaptive Kalman smoother.
- Validation: PX4/JSBSim ID/OOD, conservative cross-configuration Malolo
  transfer, and ten hardware-in-the-loop sessions.

The public results do not constitute outdoor real-flight validation or prove
closed-loop control benefit.

## Quick verification

The compact verifier supports Python 3.8 or newer. Full model training and
evaluation were frozen with Python 3.13 and the versions in
`requirements-revision.txt`; the CUDA 12.8 environment is retained separately
as `requirements-revision-cu128.txt`.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-minimal.txt
unzip evidence/minimal_dataset.zip -d evidence/
.venv/bin/python evidence/minimal_dataset/verify_reported_values.py
```

Expected result: `302 checks agree, 0 differ`.

## Repository layout

- `src/`: preprocessing, models, training, inference and data generation.
- `scripts/`: frozen revision evaluation and plotting workflows.
- `config/`: frozen experiment and deployment configurations.
- `models/`: canonical final checkpoints, ONNX model and normalization.
- `HITL/`: PC, Raspberry Pi and post-processing HITL toolchain.
- `SITL/`: retained PX4/JSBSim orchestration.
- `evidence/minimal_dataset.zip`: compact figure/table source data.

## Data

The complete data are prepared for distribution under one versioned dataset
DOI. The record contains six independently downloadable archives: acquisition
JSON, training CSV, processed 41-input arrays, main/cross-configuration
results, HITL campaign logs, and the compact minimal dataset.

Reserved dataset DOI: [10.5281/zenodo.22338389](https://doi.org/10.5281/zenodo.22338389). It will become resolvable
when the Zenodo draft is published.

## Reproduction

See `REVISION_REPRODUCIBILITY.md`, `HITL/README.md`,
`CHECKPOINT_MANIFEST.csv`, and `RELEASE_MANIFEST.md`.

## Licence and citation

Original project software is MIT licensed. The Rascal model under
`src/Rascal/` retains its GPL-3.0 licence; see `THIRD_PARTY_NOTICES.md`.
Please cite the associated article and the archived software/data records.
