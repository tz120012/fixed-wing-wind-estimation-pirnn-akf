#!/usr/bin/env python3
"""
阶段 2 集成测试：fixture JSON（含完整 stage1+stage2 字段）→ CSV(52 列) →
预处理 X (45 维) → 模型 forward+backward。

附加：检查 32-44 新维度 std > 0（不是常数死维），通过给 fixture 加入随机扰动。
"""

import json
import math
import sys
import time
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "src" / "dataset_generation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from postprocess_to_jsbsim_csv import convert_one_flight, CSV_HEADER

PRE_PATH = ROOT / "src" / "1_preprocessing_data.py"
spec = importlib.util.spec_from_file_location("preprocess_mod", str(PRE_PATH))
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)

MOD_PATH = ROOT / "src" / "2_pigru_module.py"
spec_m = importlib.util.spec_from_file_location("pigru_mod", str(MOD_PATH))
mod = importlib.util.module_from_spec(spec_m)
spec_m.loader.exec_module(mod)

FEATURE_IDX = preprocess.FEATURE_IDX


def make_fixture_json(n=400):
    rng = np.random.default_rng(0)
    entries = []
    t0 = int(time.time() * 1e6)
    for i in range(n):
        t = i / 50.0
        wn, we, wd = 1.0, 0.5, 0.0
        yaw_rad = math.radians(30.0) + 0.001 * i
        vn = 15.0 * math.cos(yaw_rad) + wn + rng.normal(0, 0.05)
        ve = 15.0 * math.sin(yaw_rad) + we + rng.normal(0, 0.05)
        vd = 0.0 + rng.normal(0, 0.02)
        airspeed = math.sqrt((vn - wn) ** 2 + (ve - we) ** 2 + (vd - wd) ** 2)
        # 加入扰动确保新维度 std > 0
        target_vn = vn - 0.2 + rng.normal(0, 0.05)
        target_ve = ve - 0.1 + rng.normal(0, 0.05)
        target_vd = 0.0 + rng.normal(0, 0.02)
        aileron_act = 0.05 + rng.normal(0, 0.005)
        elevator_act = -0.02 + rng.normal(0, 0.003)
        rudder_act = 0.005 + rng.normal(0, 0.002)
        throttle_act = 0.65 + rng.normal(0, 0.01)
        imu_ax = rng.normal(0, 0.3)
        imu_ay = rng.normal(0, 0.3)
        imu_az = -9.81 + rng.normal(0, 0.2)
        e = {
            "timestamp": round(t, 4), "wall_time_usec": t0 + int(t * 1e6),
            "airspeed_m_s": airspeed,
            "wind_north": wn, "wind_east": we, "wind_down": wd,
            "base_wind_north": wn, "base_wind_east": we, "base_wind_down": wd,
            "gust_delta_north": 0.0, "gust_delta_east": 0.0, "gust_delta_down": 0.0,
            "gust_factor": 0.0, "gust_phase": "none", "wind_regime": "steady",
            "wind_truth_aligned": True, "groundspeed_m_s": math.hypot(vn, ve),
            "velocity_north": vn, "velocity_east": ve, "velocity_down": vd,
            "roll_deg": 5.0 + rng.normal(0, 0.2), "pitch_deg": -2.0 + rng.normal(0, 0.1),
            "yaw_deg": math.degrees(yaw_rad),
            "roll_rate_rad_s": 0.01 + rng.normal(0, 0.001),
            "pitch_rate_rad_s": -0.02 + rng.normal(0, 0.001),
            "yaw_rate_rad_s": 0.03 + rng.normal(0, 0.001),
            "roll_ctrl": 0.1, "pitch_ctrl": 0.05, "yaw_ctrl": 0.0, "throttle_ctrl": 0.7,
            "target_roll_deg": 3.0, "target_pitch_deg": -1.0,
            "target_yaw_deg": math.degrees(yaw_rad) - 0.5,
            "target_attitude_valid": True, "target_attitude_age_ms": 5.0,
            "target_roll_rate_rad_s": 0.005, "target_pitch_rate_rad_s": -0.01,
            "target_yaw_rate_rad_s": 0.025,
            "target_body_rates_valid": True, "target_body_rates_age_ms": 5.0,
            # 阶段 2
            "target_velocity_north": target_vn, "target_velocity_east": target_ve,
            "target_velocity_down": target_vd,
            "target_velocity_valid": True, "target_velocity_age_ms": 5.0,
            "imu_accel_body_x": imu_ax, "imu_accel_body_y": imu_ay, "imu_accel_body_z": imu_az,
            "imu_accel_valid": True, "imu_accel_age_ms": 5.0,
            "aileron_actual_rad": aileron_act, "elevator_actual_rad": elevator_act,
            "rudder_actual_rad": rudder_act, "throttle_actual_norm": throttle_act,
            "actuator_actual_valid": True, "actuator_actual_age_ms": 5.0,

            "maneuver_regime": "straight_line", "turn_state": "non_turning", "turn_class": 0,
        }
        entries.append(e)
    return entries


def test_e2e_stage2_pipeline():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        json_path = d / "fixture.json"
        meta_path = d / "fixture_metadata.json"
        entries = make_fixture_json(n=400)
        json_path.write_text(json.dumps(entries))
        meta_path.write_text(json.dumps({"wind_north": 1.0, "wind_east": 0.5, "wind_down": 0.0}))

        csv_path = d / "fixture.csv"
        ok = convert_one_flight(str(json_path), str(meta_path), str(csv_path))
        assert ok
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
        assert len(header) == 52, f"CSV 应 52 列，实际 {len(header)}"
        for c in ["target_vn", "imu_ax", "aileron_actual"]:
            assert c in header

        X, y, w, turn = preprocess.build_features_labels_from_csv(
            str(csv_path), seq_len=50, weight_config=None, sampling_rate=50,
        )
        assert X.shape[-1] == 45, f"X.shape={X.shape}"
        assert X.shape[0] > 0

        # 检查 32-44 新维度都有非零 std（非常数死维）
        new_dim_stds = X[:, -1, 32:45].std(axis=0)
        zero_std_dims = np.where(new_dim_stds < 1e-6)[0]
        assert len(zero_std_dims) == 0, (
            f"新维度 std=0 的下标(相对32): {zero_std_dims}, "
            f"std={new_dim_stds}"
        )

        # 模型 forward + backward
        torch.manual_seed(0)
        model = mod.PIGRU(input_size=45, hidden_size=64, num_layers=1)
        model.train()
        x_t = torch.from_numpy(X[:8]).float()
        out = model(x_t)
        wind = out["wind_estimate"] if isinstance(out, dict) else out
        loss = (wind ** 2).mean()
        loss.backward()
        has_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad
        print(
            f"  e2e ok: X.shape={X.shape}, loss={loss.item():.4f}, "
            f"new_dim_std_min={new_dim_stds.min():.4f}, "
            f"new_dim_std_max={new_dim_stds.max():.4f}"
        )


if __name__ == "__main__":
    tests = [test_e2e_stage2_pipeline]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  [OK]   {fn.__name__}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [ERR]  {fn.__name__}: {type(e).__name__}: {e}")
            failures.append(fn.__name__)
    print()
    if failures:
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
