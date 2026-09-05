# Drones major-revision reproducibility package

This file describes the frozen, executable revision workflow. The authoritative
manuscript is:

`Paper_2/MDPI_template_APA/Local_3D_Wind_Estimation_Fixed-Wing_UAV_PIGRU_AKF.tex`

## Environment

Python 3.13 and the direct dependencies in `requirements-revision.txt` were used
for the new experiments. RTX 5090 training requires a PyTorch CUDA 12.8 build;
the older Python 3.8 / PyTorch 2.4.1+cu121 environment cannot execute `sm_120`
kernels.

## Frozen protocols

- Main experiment/statistics protocol: `config/revision_protocol.yaml`
- Repeated HITL protocol: `config/revision_hitl.yaml`
- Deployment provenance audit: `Paper_2/大修_部署证据链审计.md`

The locked seeds are `26, 42, 2024, 2025, 2026`. Test sets are never used for
model, filter or smoother selection.

## 41-dimensional dataset

The final input removes historical actuator-proxy indices 38--41 while
preserving all sample identities, splits and training-fitted normalization:

```bash
.venv/bin/python scripts/prepare_revision_41d_dataset.py
```

Expected shapes:

- Train: `(529751, 100, 41)`
- Validation: `(70430, 100, 41)`
- Test-ID: `(60369, 100, 41)`
- Test-OOD: `(42574, 100, 41)`

## Five-seed recurrent experiments

```bash
.venv/bin/python scripts/run_figure2_experiments.py \
  --stage all \
  --data-dir data/dataset_revision_41d \
  --output-dir data/revision_main_41d \
  --seeds 26,42,2024,2025,2026
```

The driver trains Vanilla GRU, a parameter-matched Vanilla LSTM (110 hidden
units versus GRU's 128; approximately 1.1% fewer parameters), and PI-GRU. It
evaluates PI-GRU plus the frozen AKF per seed. Seed-level mean and unbiased
sample standard deviation are retained separately from sample-level
uncertainty.

After evaluation, compute the preregistered uncertainty layers:

```bash
.venv/bin/python scripts/analyze_revision_main_statistics.py \
  --results-dir data/revision_main_41d \
  --data-dir data/dataset_revision_41d \
  --kalmannet-dir data/revision_kalmannet_41d \
  --output-dir data/revision_main_41d/statistics
```

This produces five-seed mean ± SD, sample-level moving-block-bootstrap 95% CIs
and paired window-level Wilcoxon tests as distinct outputs.

## KalmanNet

```bash
.venv/bin/python scripts/train_kalmannet_baseline.py \
  --data-dir data/dataset_revision_41d \
  --out-dir data/revision_kalmannet_41d \
  --seeds 26,42,2024,2025,2026
```

## AKF sensitivity and simple causal baselines

After selecting a PI-GRU checkpoint on validation data:

```bash
.venv/bin/python scripts/run_akf_revision_experiments.py \
  --checkpoint PATH/TO/SEED/CHECKPOINT
```

The script tunes fixed EMA, fixed-covariance KF and confidence-complementary
smoother parameters on validation only. It evaluates the locked test sets and
applies 0.6/0.8/1.2/1.4 multipliers to six grouped AKF constant families.

## Zero-shot cross-airframe evaluation

The Malolo acquisition changes only the simulator airframe adapter and safe
flight envelope. The Rascal normalization, PI-GRU weights, AKF constants and
decision thresholds remain frozen. Use the pinned JSBSim 1.1.2 executable;
newer JSBSim releases can trigger the documented fixed-wing motor/gear
instability during runway takeoff:

```bash
export PATH="$PWD/.venv/bin:$PATH"
python3 scripts/patch_px4_malolo_for_revision.py \
  --px4-root /path/to/PX4-Autopilot
PX4_ROOT=/path/to/PX4-Autopilot .venv/bin/python \
  src/dataset_generation/scripts/generate_dataset.py \
  --mode single_round --dataset-type test_id --airframe malolo \
  --output-dir data/cross_airframe_malolo_raw \
  --px4-dir /path/to/PX4-Autopilot --seed 260826

PX4_ROOT=/path/to/PX4-Autopilot .venv/bin/python \
  src/dataset_generation/scripts/generate_dataset.py \
  --mode single_round --dataset-type test_ood --airframe malolo \
  --output-dir data/cross_airframe_malolo_raw \
  --px4-dir /path/to/PX4-Autopilot --seed 260827

# If a simulator sortie was skipped, recover only its original logical segment:
PX4_ROOT=/path/to/PX4-Autopilot .venv/bin/python \
  src/dataset_generation/scripts/generate_dataset.py \
  --mode single_round --dataset-type test_ood --segments 3 --airframe malolo \
  --output-dir data/cross_airframe_malolo_raw \
  --px4-dir /path/to/PX4-Autopilot --seed 260827

.venv/bin/python src/dataset_generation/scripts/postprocess_to_jsbsim_csv.py \
  data/cross_airframe_malolo_raw \
  --output-dir data/cross_airframe_malolo_processed \
  --splits test_id test_ood

.venv/bin/python scripts/evaluate_cross_airframe.py \
  --csv-dir data/cross_airframe_malolo_processed \
  --checkpoints \
    data/revision_main_41d/models/seed_26/pigru \
    data/revision_main_41d/models/seed_42/pigru \
    data/revision_main_41d/models/seed_2024/pigru \
    data/revision_main_41d/models/seed_2025/pigru \
    data/revision_main_41d/models/seed_2026/pigru
```

The Malolo acquisition adapter constrains only the simulator flight envelope:
a reproducible 750 W electric replacement for the unstable stock piston model,
60--70 m acquisition altitude, 65% maximum throttle, 22 m/s maximum configured
airspeed, straight flight for the strong-wind split, and a 3.5 m/s background
wind plus gusts capped at 2.5 m/s. These settings prevent the lightweight
airframe from leaving its tabulated aerodynamic domain. The Rascal-fitted
normalizer, all five network checkpoints, AKF constants, thresholds and metric
definitions remain unchanged; no Malolo result is used for tuning.
Malolo shares much of the Rascal geometry and aerodynamic coefficient
structure, so this experiment is reported as a conservative
cross-configuration/cross-propulsion airframe transfer rather than a strong
geometry-level domain shift.

## ONNX deployment artifact

```bash
.venv/bin/python scripts/export_pigru_onnx.py \
  --checkpoint PATH/TO/FINAL/PI_GRU/CHECKPOINT \
  --output artifacts/pigru_41d_seed26.onnx
```

The export command performs an ONNX Runtime numerical-parity check and copies
the frozen normalization parameters next to the model.

## Repeated HITL

Aligned session CSV files must follow `config/revision_hitl.yaml`. Aggregate
only after five independent ID and five independent OOD sessions exist:

```bash
.venv/bin/python scripts/analyze_repeated_hitl.py
```

The script refuses incomplete session sets and treats each session—not each
frame—as an independent replicate. Transport or round-trip latency is not
reported unless flight-controller and companion clocks are synchronized.

## Current external release status

See `RELEASE_MANIFEST.md`. The public source repository is available at
<https://github.com/tz120012/fixed-wing-wind-estimation-pirnn-akf>. The unified
data record has reserved DOI <https://doi.org/10.5281/zenodo.22338389>, which
will become resolvable after the Zenodo draft is published. The versioned
software DOI remains pending creation of the GitHub `v1.0.0` release.
