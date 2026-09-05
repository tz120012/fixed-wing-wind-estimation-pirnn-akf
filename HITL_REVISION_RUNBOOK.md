# Repeated Raspberry Pi 5 + CUAV V5+ HITL runbook

The authoritative runbook and all executable entry points now live in
[`HITL/README.md`](HITL/README.md).

The revised workflow separates three evidence streams:

1. PC-side JSBSim truth on the PC wall clock;
2. final 41-D ONNX + AKF estimates on the Raspberry Pi wall/monotonic clocks;
3. PX4 ULog on the flight-controller boot clock.

PC–Pi offset and drift are measured with explicit four-timestamp UDP probes.
Pi–PX4 alignment uses FC boot timestamps. JSBSim truth is never read by the Pi
estimator. Do not use the former direct `WIND`-message truth fields or the
legacy `scripts/analyze_repeated_hitl.py` path for the revision campaign.

The final entry points are:

```bash
# PC/Pi preflight and acquisition
.venv/bin/python HITL/pc/preflight.py
.venv/bin/python HITL/pi/preflight.py --session-id id_01

# Offline merge, alignment and strict acceptance
.venv/bin/python HITL/postprocess/align_session.py id_01
.venv/bin/python HITL/postprocess/validate_session.py id_01
.venv/bin/python HITL/postprocess/analyze_campaign.py
```

No result may be cited until all five ID and five OOD sessions pass the strict
per-session validator.
