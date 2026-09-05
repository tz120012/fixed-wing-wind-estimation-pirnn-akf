#!/usr/bin/env python3
"""
阶段 1 集成测试（轻量端到端，不依赖 PX4 SITL）：
  1. 构造 fixture JSON（DataLogger 风格，含 target_*_deg 字段）
  2. postprocess_to_jsbsim_csv → CSV（应含 42 列）
  3. preprocess CSV → X/y（X.shape[-1]==32）
  4. PIGRU forward + backward 不抛 shape 异常
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


def make_fixture_json(n=400):
    """模拟 DataLogger 输出的 JSON：n 条 entry，~50Hz，含 target_*_deg。"""
    entries = []
    t0_wall = int(time.time() * 1e6)
    for i in range(n):
        t = i / 50.0
        # 真值风：steady (1.0, 0.5, 0.0)
        wn, we, wd = 1.0, 0.5, 0.0
        # 飞行：30° heading, 15 m/s
        yaw_rad = math.radians(30.0)
        # 地速 = 空速向量 + 风
        vn = 15.0 * math.cos(yaw_rad) + wn
        ve = 15.0 * math.sin(yaw_rad) + we
        vd = 0.0
        airspeed = math.sqrt((vn - wn) ** 2 + (ve - we) ** 2 + (vd - wd) ** 2)

        entry = {
            "timestamp": round(t, 4),
            "wall_time_usec": t0_wall + int(t * 1e6),
            "airspeed_m_s": airspeed,
            "wind_north": wn,
            "wind_east": we,
            "wind_down": wd,
            "base_wind_north": wn,
            "base_wind_east": we,
            "base_wind_down": wd,
            "gust_delta_north": 0.0,
            "gust_delta_east": 0.0,
            "gust_delta_down": 0.0,
            "gust_factor": 0.0,
            "gust_phase": "none",
            "wind_regime": "steady",
            "wind_truth_aligned": True,
            "groundspeed_m_s": math.hypot(vn, ve),
            "velocity_north": vn,
            "velocity_east": ve,
            "velocity_down": vd,
            "roll_deg": 5.0,
            "pitch_deg": -2.0,
            "yaw_deg": 30.0,
            "roll_rate_rad_s": 0.01,
            "pitch_rate_rad_s": -0.02,
            "yaw_rate_rad_s": 0.03,
            "roll_ctrl": 0.1,
            "pitch_ctrl": 0.05,
            "yaw_ctrl": 0.0,
            "throttle_ctrl": 0.7,
            # 阶段 1：目标量
            "target_roll_deg": 3.0,
            "target_pitch_deg": -1.0,
            "target_yaw_deg": 30.0,
            "target_attitude_valid": True,
            "target_attitude_age_ms": 5.0,
            "target_roll_rate_rad_s": 0.005,
            "target_pitch_rate_rad_s": -0.01,
            "target_yaw_rate_rad_s": 0.025,
            "target_body_rates_valid": True,
            "target_body_rates_age_ms": 5.0,
            # 阶段 2 占位（合规即可）
            "target_velocity_north": 0.0, "target_velocity_east": 0.0, "target_velocity_down": 0.0,
            "target_velocity_valid": False, "target_velocity_age_ms": -1.0,
            "imu_accel_body_x": 0.0, "imu_accel_body_y": 0.0, "imu_accel_body_z": 0.0,
            "imu_accel_valid": False, "imu_accel_age_ms": -1.0,
            "aileron_actual_rad": 0.0, "elevator_actual_rad": 0.0,
            "rudder_actual_rad": 0.0, "throttle_actual_norm": 0.0,
            "actuator_actual_valid": False, "actuator_actual_age_ms": -1.0,

            "maneuver_regime": "straight_line",
            "turn_state": "non_turning",
            "turn_class": 0,
        }
        entries.append(entry)
    return entries


def test_e2e_stage1_pipeline():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # 1. fixture JSON
        json_path = d / "fixture.json"
        meta_path = d / "fixture_metadata.json"
        entries = make_fixture_json(n=400)
        json_path.write_text(json.dumps(entries))
        meta_path.write_text(json.dumps({"wind_north": 1.0, "wind_east": 0.5, "wind_down": 0.0}))

        # 2. postprocess
        csv_path = d / "fixture.csv"
        ok = convert_one_flight(str(json_path), str(meta_path), str(csv_path))
        assert ok, "postprocess 失败"
        # 验证 CSV 至少包含阶段 1 列（阶段 2 应用后还会再多 10 列）
        with open(csv_path) as f:
            header = f.readline().strip().split(",")
        assert len(header) >= 42, f"CSV header 应≥42 列, 实际 {len(header)}"
        for c in ["target_roll_rad", "target_p_rad_s"]:
            assert c in header, f"CSV 缺新列 {c}"

        # 3. preprocess（FEATURE_DIM 由模块级常量决定：阶段1=32 / 阶段2=45）
        X, y, w, turn = preprocess.build_features_labels_from_csv(
            str(csv_path), seq_len=50, weight_config=None, sampling_rate=50,
        )
        assert X.shape[-1] == preprocess.FEATURE_DIM, f"X.shape={X.shape}"
        assert X.shape[0] > 0
        assert y.shape == (X.shape[0], 7)

        # 4. 模型 forward + backward（用当前 FEATURE_DIM）
        torch.manual_seed(0)
        model = mod.PIGRU(input_size=preprocess.FEATURE_DIM, hidden_size=64, num_layers=1)
        model.train()
        batch = X[:8]
        x_t = torch.from_numpy(batch).float()
        out = model(x_t)
        wind = out["wind_estimate"] if isinstance(out, dict) else out
        loss = (wind ** 2).mean()
        loss.backward()
        # 梯度检查
        has_grad = any(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "反向传播没有有效梯度"
        print(
            f"  e2e ok: X.shape={X.shape}, header_cols={len(header)}, "
            f"loss={loss.item():.4f}"
        )


if __name__ == "__main__":
    tests = [test_e2e_stage1_pipeline]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  [OK]   {fn.__name__}")
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failures.append(fn.__name__)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERR]  {fn.__name__}: {type(e).__name__}: {e}")
            failures.append(fn.__name__)
    print()
    if failures:
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
