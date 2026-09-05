# Revision HITL toolchain

This directory contains the PC, Raspberry Pi and offline-analysis programs for
the final 41-input, 100-step PIRNN-AKF repeated HITL experiment.

For the complete Chinese operating procedure, see
[`README_zh-CN.md`](README_zh-CN.md).

The evidence streams are deliberately independent:

- PC: PX4/JSBSim HITL process and simulator wind truth.
- Raspberry Pi 5: MAVLink states, PI-GRU/AKF estimates and processing latency.
- CUAV V5+: PX4 ULog retained on the flight controller.
- QGroundControl: manual arming, mission upload, takeoff and flight supervision.

JSBSim truth is never sent to the Pi estimator. The Pi sends `WIND_COV` and
debug named values back to PX4 only to demonstrate the return link; PX4 must
not consume these values for control during this experiment.

## Directory layout

```text
HITL/
  config/                 frozen campaign and machine-local example
  common/                 shared schema, config and provenance utilities
  pc/                     PC preflight, JSBSim launch and clock probes
  pi/                     Pi setup, preflight, clock listener and estimator
  postprocess/            staging, alignment, validation and aggregation
  revision_protocol/      frozen ONNX, scaler and configuration
  sessions/<session_id>/  manually consolidated three-stream session data
  tests/                  hardware-free regression tests
```

Generated session IDs are exactly `id_01`–`id_05` and
`ood_01`–`ood_05`.

## One-time preparation

### PC

Create the machine-local configuration:

```bash
cp HITL/config/local.example.yaml HITL/config/local.yaml
```

Edit `local.yaml`. The PC USB/HITL device and the Pi TELEM2 device are
different:

- `pc.hitl_serial_device`: normally `/dev/ttyACM0` in WSL2.
- `pi.telem_serial_device`: normally `/dev/ttyAMA10` on Raspberry Pi 5.

Export and freeze the selected seed-26 model:

```bash
.venv/bin/python scripts/export_pigru_onnx.py \
  --checkpoint data/revision_main_41d/models/seed_26/pigru/<run>/best_model.pth \
  --norm-params data/dataset_revision_41d/norm_params.pkl \
  --test-array data/dataset_revision_41d/X_test_id.npy \
  --output artifacts/pigru_41d_seed26.onnx

.venv/bin/python HITL/prepare_artifacts.py \
  --onnx artifacts/pigru_41d_seed26.onnx \
  --norm-params data/dataset_revision_41d/norm_params.pkl
```

Copy the project, including `HITL/revision_protocol/`, to the Pi. Do not copy
old `HITL/logs_in_rasbpi/` as revision evidence.

Attach the CUAV USB device to WSL2, then run:

```bash
.venv/bin/python HITL/pc/preflight.py
.venv/bin/python HITL/pc/generate_session_order.py
```

Keep `HITL/sessions/session_order.json` unchanged for the full campaign.

### Raspberry Pi 5

```bash
bash HITL/pi/setup.sh
cp HITL/config/local.example.yaml HITL/config/local.yaml
```

Edit the Pi paths/address in `local.yaml`, log out/in if `dialout` membership
was changed, then check one intended session:

```bash
.venv/bin/python HITL/pi/preflight.py --session-id id_01
```

The scaler is serialized with scikit-learn 1.3.2. `setup.sh` pins that version
and NumPy below 2 to avoid cross-version model-persistence ambiguity.

## Running one session

The example below uses `id_01`. Replace it consistently in every command.

### 1. Start PC-side JSBSim HITL

On the PC:

```bash
.venv/bin/python HITL/pc/run_jsbsim.py id_01
```

This applies the frozen wind setting and starts
`$PX4_ROOT/Tools/hitl_run.sh`. It runs until interrupted. QGC operation remains
manual. Do not start a second bridge for the same flight.

### 2. Start the Pi clock listener

In Pi terminal A:

```bash
.venv/bin/python HITL/pi/clock_marker_listener.py \
  --session id_01 \
  --output HITL/sessions/id_01/pi/clock_markers_served.jsonl \
  --duration 420
```

The UDP messages contain only session and clock data, never wind truth.

### 3. Record start clock probes

After the aircraft has reached the planned acquisition segment, on the PC:

```bash
.venv/bin/python HITL/pc/session_marker.py id_01 \
  --host <PI_IP> --phase start
```

### 4. Run the estimator on the Pi

In Pi terminal B:

```bash
.venv/bin/python HITL/pi/run_estimator.py --session-id id_01
```

The default runtime is 350 s (`protocol.total_duration_s` in
`HITL/config/campaign.yaml`). The first approximately 2 s fill the 100-step
sequence buffer; offline analysis then excludes a further 20 s warmup and
still requires at least 100 valid seconds. The longer live run leaves about
328 s after warmup, so takeoff settling and clock probes do not fail the
old 125 s minimum.

The estimator prints a countdown progress bar. When it reaches 100% the
process exits; immediately send the end clock probe, stop JSBSim, then
continue to the next session. Keep the Pi clock listener up for the whole
acquisition: `--duration 420` covers 350 s of estimation plus about 70 s
for start/end probes.

During this interval, use QGC to maintain the predefined mission. Do not
change the model, filter, serial rate or wind condition between sessions.

### 5. Record end clock probes

Immediately after the estimator exits, on the PC:

```bash
.venv/bin/python HITL/pc/session_marker.py id_01 \
  --host <PI_IP> --phase end
```

Then stop `run_jsbsim.py` with Ctrl+C. It restores the prior bridge wind files
and copies the unique new truth file to:

```text
HITL/sessions/id_01/pc/jsbsim_truth.csv
```

Download the corresponding ULog through QGC or from the CUAV SD card.

## Consolidating data

The Pi retains its raw data until collection. Copy these files manually to the
same session on the PC:

```text
HITL/sessions/id_01/pi/pi_estimator.csv
HITL/sessions/id_01/pi/clock_markers_served.jsonl
```

Place the flight-controller log at:

```text
HITL/sessions/id_01/fc/px4.ulg
```

The two PC marker files already reside in:

```text
HITL/sessions/id_01/pc/marker_start.jsonl
HITL/sessions/id_01/pc/marker_end.jsonl
```

If raw files were collected elsewhere, stage them unambiguously:

```bash
.venv/bin/python HITL/postprocess/prepare_session.py id_01 \
  --truth /path/to/one/wind_truth.csv \
  --pi /path/to/one/pi_estimator.csv \
  --ulog /path/to/one/log.ulg
```

## Time alignment

Three clock domains are retained:

1. JSBSim truth: PC wall clock (`wall_time_usec`).
2. Pi: wall clock and monotonic clock.
3. PX4: boot clock from MAVLink and ULog.

The start/end UDP probes use the four-timestamp NTP equation. Offline fitting
selects low-RTT probes and estimates both PC–Pi offset and linear drift.
Corrected Pi wall time is then matched to PC truth. PX4 ULog is independently
cross-checked through FC boot time. Estimator response peaks are never used to
choose an offset.

Run:

```bash
.venv/bin/python HITL/postprocess/align_session.py id_01
.venv/bin/python HITL/postprocess/validate_session.py id_01
```

Validation fails the entire session instead of dropping invalid frames. It
checks finite values, monotonic clocks, duration, loop rate, deadlines, clock
uncertainty, truth coverage/residual, optional ULog coverage and actual
condition wind range.

## Complete 5 + 5 campaign

Repeat the procedure for:

```text
id_01 id_02 id_03 id_04 id_05
ood_01 ood_02 ood_03 ood_04 ood_05
```

Use the frozen order in `HITL/sessions/session_order.json`. The session
identity determines the frozen speed and direction in
`HITL/config/campaign.yaml`.

After all ten `validation.json` files report `valid: true`:

```bash
.venv/bin/python HITL/postprocess/analyze_campaign.py
```

Outputs under `HITL/sessions/campaign_summary/`:

- `session_metrics.csv`
- `condition_summary.csv`
- `campaign_summary.json`

The independent session is the replicate unit. Frames are never treated as
independent experimental replicates.

## Tests

```bash
.venv/bin/python -m unittest discover -s HITL/tests -v
```

The tests cover four-timestamp parsing, clock offset/drift recovery, NaN
rejection, condition-range rejection, incomplete campaigns and metric
aggregation.

## Legacy data

`logs_in_rasbpi/`, `JSBSim_truth_wind/`, `PX4_ulog/`, `aligned/`,
`hitl_collect.sh` and `align_hitl_timestamps.py` belong to earlier campaigns.
They are retained for provenance but must not be mixed with
`HITL/sessions/` revision results.
