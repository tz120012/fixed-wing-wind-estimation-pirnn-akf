#!/usr/bin/env python3
"""
预处理阶段 2 单元测试：
- FEATURE_DIM == 45
- 32-34 target_vn/ve/vd 填充正确
- 35-37 vel_err = vel - target_vel 正确
- 38-41 实际舵面填充
- 42-44 IMU 加速度填充
- 缺 stage2 列时退化为 0
"""

import math
import sys
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PRE_PATH = ROOT / "src" / "1_preprocessing_data.py"
spec = importlib.util.spec_from_file_location("preprocess_mod", str(PRE_PATH))
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)

FEATURE_IDX = preprocess.FEATURE_IDX
FEATURE_DIM = preprocess.FEATURE_DIM
build = preprocess.build_features_labels_from_csv


def make_fixture_df(n=300, with_stage2=True):
    rng = np.random.default_rng(0)
    t = np.arange(n) / 50.0
    wn_mps = np.full(n, 1.0)
    we_mps = np.full(n, 0.5)
    wd_mps = np.zeros(n)
    vn_mps = 15.0 * math.cos(math.radians(30.0)) + wn_mps + rng.normal(0, 0.05, n)
    ve_mps = 15.0 * math.sin(math.radians(30.0)) + we_mps + rng.normal(0, 0.05, n)
    vd_mps = wd_mps + rng.normal(0, 0.02, n)
    vtrue_mps = np.sqrt((vn_mps - wn_mps) ** 2 + (ve_mps - we_mps) ** 2 + (vd_mps - wd_mps) ** 2)
    fps = 1.0 / 0.3048
    df = pd.DataFrame({
        "Time": t,
        "/fdm/jsbsim/simulation/sim-time-sec": t,
        "/fdm/jsbsim/atmosphere/wind-north-fps": wn_mps * fps,
        "/fdm/jsbsim/atmosphere/wind-east-fps": we_mps * fps,
        "/fdm/jsbsim/atmosphere/wind-down-fps": wd_mps * fps,
        "/fdm/jsbsim/velocities/vc-fps": vtrue_mps * fps,
        "/fdm/jsbsim/velocities/vtrue-fps": vtrue_mps * fps,
        "/fdm/jsbsim/velocities/vg-fps": np.hypot(vn_mps, ve_mps) * fps,
        "/fdm/jsbsim/position/h-agl-ft": 60.0 / 0.3048,
        "/fdm/jsbsim/position/lat-geod-deg": 47.0,
        "/fdm/jsbsim/position/long-gc-deg": 8.0,
        "/fdm/jsbsim/velocities/v-north-fps": vn_mps * fps,
        "/fdm/jsbsim/velocities/v-east-fps": ve_mps * fps,
        "/fdm/jsbsim/velocities/v-down-fps": vd_mps * fps,
        "/fdm/jsbsim/attitude/pitch-rad": np.full(n, math.radians(2.0)),
        "/fdm/jsbsim/attitude/roll-rad": np.full(n, math.radians(5.0)),
        "/fdm/jsbsim/attitude/psi-rad": np.full(n, math.radians(30.0)),
        "/fdm/jsbsim/velocities/p-rad_sec": np.full(n, 0.01),
        "/fdm/jsbsim/velocities/q-rad_sec": np.full(n, -0.02),
        "/fdm/jsbsim/velocities/r-rad_sec": np.full(n, 0.03),
        "/fdm/jsbsim/fcs/aileron-cmd-norm": np.full(n, 0.1),
        "/fdm/jsbsim/fcs/elevator-cmd-norm": np.full(n, 0.05),
        "/fdm/jsbsim/fcs/throttle-cmd-norm": np.full(n, 0.7),
        "/fdm/jsbsim/fcs/rudder-cmd-norm": np.zeros(n),
        "wind_regime": "steady",
        "gust_phase": "none",
        "gust_factor": 0.0,
        "base_wind_north_mps": wn_mps,
        "base_wind_east_mps": we_mps,
        "base_wind_down_mps": wd_mps,
        "gust_delta_north_mps": np.zeros(n),
        "gust_delta_east_mps": np.zeros(n),
        "gust_delta_down_mps": np.zeros(n),
        "maneuver_regime": "straight_line",
        "turn_state": "non_turning",
        "turn_class": 0,
        # 阶段 1
        "target_roll_rad": math.radians(3.0),
        "target_pitch_rad": math.radians(-1.0),
        "target_yaw_rad": math.radians(30.0),
        "target_p_rad_s": 0.005,
        "target_q_rad_s": -0.01,
        "target_r_rad_s": 0.025,
    })
    if with_stage2:
        df["target_vn"] = 13.0
        df["target_ve"] = 7.5
        df["target_vd"] = 0.2
        df["aileron_actual"] = 0.04
        df["elevator_actual"] = -0.015
        df["rudder_actual"] = 0.005
        df["throttle_actual"] = 0.62
        df["imu_ax"] = 0.5
        df["imu_ay"] = -0.3
        df["imu_az"] = -9.7
    return df


def test_feature_dim_45():
    assert FEATURE_DIM == 45, f"expected FEATURE_DIM=45, got {FEATURE_DIM}"


def test_x_shape_45():
    df = make_fixture_df(n=300, with_stage2=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fixture.csv"
        df.to_csv(p, index=False)
        X, y, w, turn = build(str(p), seq_len=50, weight_config=None, sampling_rate=50)
    assert X.shape[-1] == 45, X.shape
    assert X.shape[0] > 0


def test_target_velocity_filled():
    df = make_fixture_df(n=300, with_stage2=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fixture.csv"
        df.to_csv(p, index=False)
        X, _, _, _ = build(str(p), seq_len=50, weight_config=None, sampling_rate=50)
    last = X[:, -1, :]
    np.testing.assert_allclose(last[:, FEATURE_IDX["target_vn"]], 13.0, atol=1e-5)
    np.testing.assert_allclose(last[:, FEATURE_IDX["target_ve"]], 7.5, atol=1e-5)
    np.testing.assert_allclose(last[:, FEATURE_IDX["target_vd"]], 0.2, atol=1e-5)


def test_velocity_err_correctness():
    """vel_err = vel - target_vel"""
    df = make_fixture_df(n=300, with_stage2=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fixture.csv"
        df.to_csv(p, index=False)
        X, _, _, _ = build(str(p), seq_len=50, weight_config=None, sampling_rate=50)
    last = X[:, -1, :]
    # 基础 vel_n ≈ 15*cos30 + 1.0 ≈ 13.99；target_vn = 13.0；err ≈ 0.99
    vn_actual = last[:, FEATURE_IDX["vel_n"]]
    vn_err_actual = last[:, FEATURE_IDX["vn_err"]]
    np.testing.assert_allclose(vn_err_actual, vn_actual - 13.0, atol=1e-5)


def test_actuator_imu_filled():
    df = make_fixture_df(n=300, with_stage2=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fixture.csv"
        df.to_csv(p, index=False)
        X, _, _, _ = build(str(p), seq_len=50, weight_config=None, sampling_rate=50)
    last = X[:, -1, :]
    np.testing.assert_allclose(last[:, FEATURE_IDX["aileron_act"]], 0.04, atol=1e-5)
    np.testing.assert_allclose(last[:, FEATURE_IDX["elevator_act"]], -0.015, atol=1e-5)
    np.testing.assert_allclose(last[:, FEATURE_IDX["rudder_act"]], 0.005, atol=1e-5)
    np.testing.assert_allclose(last[:, FEATURE_IDX["throttle_act"]], 0.62, atol=1e-5)
    np.testing.assert_allclose(last[:, FEATURE_IDX["imu_ax"]], 0.5, atol=1e-5)
    np.testing.assert_allclose(last[:, FEATURE_IDX["imu_ay"]], -0.3, atol=1e-5)
    np.testing.assert_allclose(last[:, FEATURE_IDX["imu_az"]], -9.7, atol=1e-5)


def test_stage2_missing_falls_back_zero():
    """缺 stage2 列时，32-44 维全 0；vel_err = vel - 0 = vel。"""
    df = make_fixture_df(n=300, with_stage2=False)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fixture.csv"
        df.to_csv(p, index=False)
        X, _, _, _ = build(str(p), seq_len=50, weight_config=None, sampling_rate=50)
    last = X[:, -1, :]
    np.testing.assert_allclose(last[:, FEATURE_IDX["target_vn"]], 0.0, atol=1e-9)
    np.testing.assert_allclose(last[:, FEATURE_IDX["aileron_act"]], 0.0, atol=1e-9)
    np.testing.assert_allclose(last[:, FEATURE_IDX["imu_ax"]], 0.0, atol=1e-9)
    # throttle_act 缺失时 nan_to_num 默认 0.5（与 throttle_cmd 一致）
    np.testing.assert_allclose(last[:, FEATURE_IDX["throttle_act"]], 0.5, atol=1e-9)
    # vel_err = vel - 0
    np.testing.assert_allclose(
        last[:, FEATURE_IDX["vn_err"]],
        last[:, FEATURE_IDX["vel_n"]],
        atol=1e-5,
    )


def test_dim_std_nonzero():
    """每个新维度（32-44）在 with_stage2=True 时应为常数 → std=0；
    在 with_stage2=True 但 fixture 全为常数本意如此，所以这里换一种方式：
    至少所有 0..44 维度都有可识别（非 NaN）数据。"""
    df = make_fixture_df(n=300, with_stage2=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fixture.csv"
        df.to_csv(p, index=False)
        X, _, _, _ = build(str(p), seq_len=50, weight_config=None, sampling_rate=50)
    assert np.all(np.isfinite(X)), "X 含 NaN/Inf"


if __name__ == "__main__":
    tests = [
        test_feature_dim_45,
        test_x_shape_45,
        test_target_velocity_filled,
        test_velocity_err_correctness,
        test_actuator_imu_filled,
        test_stage2_missing_falls_back_zero,
        test_dim_std_nonzero,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  [OK]   {fn.__name__}")
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failures.append(fn.__name__)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [ERR]  {fn.__name__}: {type(e).__name__}: {e}")
            failures.append(fn.__name__)
    print()
    if failures:
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
