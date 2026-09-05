#!/usr/bin/env python3
"""
预处理阶段 1 单元测试：
- 验证 X.shape[-1] == FEATURE_DIM (=32)
- 验证 feat[:, 23] = roll - target_roll
- 验证 yaw_err ∈ [-π, π]
- 验证缺失 target_* 列时退化为 0
"""

import math
import sys
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# src/1_preprocessing_data.py 文件名以数字开头，无法直接 import；用 importlib 加载
ROOT = Path(__file__).resolve().parents[4]  # repo root: wind-estimation-main
PRE_PATH = ROOT / "src" / "1_preprocessing_data.py"
spec = importlib.util.spec_from_file_location("preprocess_mod", str(PRE_PATH))
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)

FEATURE_IDX = preprocess.FEATURE_IDX
FEATURE_DIM = preprocess.FEATURE_DIM
build = preprocess.build_features_labels_from_csv


def make_fixture_df(n=300, with_target=True):
    """构造一个最小可被预处理消费的 CSV df。
    包含 jsbsim 标准列 + 阶段 1 目标量列；速度三角形大致一致。
    """
    rng = np.random.default_rng(0)
    t = np.arange(n) / 50.0
    # 风：稳定 (1.0, 0.5, 0.0) m/s
    wn_mps = np.full(n, 1.0)
    we_mps = np.full(n, 0.5)
    wd_mps = np.zeros(n)
    # 地速 (m/s)：以 15 m/s 平飞 + 风
    vn_mps = 15.0 * np.cos(np.deg2rad(30.0)) + wn_mps + rng.normal(0, 0.05, n)
    ve_mps = 15.0 * np.sin(np.deg2rad(30.0)) + we_mps + rng.normal(0, 0.05, n)
    vd_mps = wd_mps + rng.normal(0, 0.02, n)
    # 空速 = ||v_gnd - wind||
    vtrue_mps = np.sqrt(
        (vn_mps - wn_mps) ** 2 + (ve_mps - we_mps) ** 2 + (vd_mps - wd_mps) ** 2
    )
    fps = 1.0 / 0.3048  # mps → fps
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
    })
    if with_target:
        # 目标姿态：roll=3°, pitch=-1°, yaw=-179° (用于触发 yaw_err wrap)
        df["target_roll_rad"] = math.radians(3.0)
        df["target_pitch_rad"] = math.radians(-1.0)
        df["target_yaw_rad"] = math.radians(-179.0)
        df["target_p_rad_s"] = 0.005
        df["target_q_rad_s"] = -0.01
        df["target_r_rad_s"] = 0.025
    return df


def test_feature_dim_at_least_32():
    """阶段1 维度至少 32，阶段2 维度 45。"""
    assert FEATURE_DIM in (32, 45), f"FEATURE_DIM 必须是 32 或 45，实际 {FEATURE_DIM}"


def test_feature_idx_complete():
    """FEATURE_IDX 必须覆盖 0..44 全部下标。"""
    vals = sorted(FEATURE_IDX.values())
    assert vals == list(range(45)), f"FEATURE_IDX 不连续: {vals}"


def test_x_shape_with_target():
    df = make_fixture_df(n=300, with_target=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fixture.csv"
        df.to_csv(p, index=False)
        X, y, w, turn = build(str(p), seq_len=50, weight_config=None, sampling_rate=50)
    assert X.ndim == 3, X.shape
    assert X.shape[-1] == FEATURE_DIM, f"got X.shape={X.shape}"
    assert X.shape[0] > 0, "应该有非空序列"
    assert y.shape[1] == 7
    assert w.shape == (X.shape[0],)
    assert turn.shape == (X.shape[0],)


def test_attitude_error_correctness():
    """feat[:, 23] = roll - target_roll，feat[:, 25] (yaw_err) 必须 wrap 到 (-π, π]。"""
    df = make_fixture_df(n=300, with_target=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fixture.csv"
        df.to_csv(p, index=False)
        X, y, w, turn = build(str(p), seq_len=50, weight_config=None, sampling_rate=50)
    # 取每个序列窗口的最后一个时刻
    last = X[:, -1, :]
    # roll = 5°, target_roll = 3° → roll_err ≈ 2°
    expected_roll_err = math.radians(5.0 - 3.0)
    np.testing.assert_allclose(last[:, FEATURE_IDX["roll_err"]],
                               expected_roll_err, atol=1e-5)
    # pitch = 2°, target_pitch = -1° → pitch_err = 3°
    expected_pitch_err = math.radians(3.0)
    np.testing.assert_allclose(last[:, FEATURE_IDX["pitch_err"]],
                               expected_pitch_err, atol=1e-5)
    # yaw = 30°, target_yaw = -179° → naive 30 - (-179) = 209° → wrap → 209-360 = -151°
    expected_yaw_err_deg = -151.0
    yaw_err_actual = last[:, FEATURE_IDX["yaw_err"]]
    expected_yaw_err = math.radians(expected_yaw_err_deg)
    # yaw_err ∈ (-π, π]
    assert np.all(yaw_err_actual > -math.pi - 1e-6)
    assert np.all(yaw_err_actual <= math.pi + 1e-6)
    np.testing.assert_allclose(yaw_err_actual, expected_yaw_err, atol=1e-4)


def test_target_attitude_values():
    df = make_fixture_df(n=300, with_target=True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fixture.csv"
        df.to_csv(p, index=False)
        X, y, w, turn = build(str(p), seq_len=50, weight_config=None, sampling_rate=50)
    last = X[:, -1, :]
    np.testing.assert_allclose(last[:, FEATURE_IDX["target_roll"]],
                               math.radians(3.0), atol=1e-6)
    np.testing.assert_allclose(last[:, FEATURE_IDX["target_pitch"]],
                               math.radians(-1.0), atol=1e-6)
    # target_p_err = roll_rate(0.01) - target_p(0.005) = 0.005
    np.testing.assert_allclose(last[:, FEATURE_IDX["p_err"]], 0.005, atol=1e-6)


def test_missing_target_falls_back_zero():
    """当 CSV 缺 target_* 列时，feat 20-31 全为 0（误差也为 0 - 实际值 = -实际值，
    具体来说 roll_err = roll - 0 = roll）"""
    df = make_fixture_df(n=300, with_target=False)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "fixture.csv"
        df.to_csv(p, index=False)
        X, y, w, turn = build(str(p), seq_len=50, weight_config=None, sampling_rate=50)
    last = X[:, -1, :]
    # target_roll/pitch/yaw 应全为 0
    assert np.allclose(last[:, FEATURE_IDX["target_roll"]], 0.0)
    assert np.allclose(last[:, FEATURE_IDX["target_pitch"]], 0.0)
    assert np.allclose(last[:, FEATURE_IDX["target_yaw"]], 0.0)
    # roll_err = roll - 0 = math.radians(5.0)
    np.testing.assert_allclose(last[:, FEATURE_IDX["roll_err"]],
                               math.radians(5.0), atol=1e-5)


if __name__ == "__main__":
    tests = [
        test_feature_dim_at_least_32,
        test_feature_idx_complete,
        test_x_shape_with_target,
        test_attitude_error_correctness,
        test_target_attitude_values,
        test_missing_target_falls_back_zero,
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
            print(f"  [ERR]  {fn.__name__}: {type(e).__name__}: {e}")
            failures.append(fn.__name__)
    print()
    if failures:
        print(f"FAILED: {len(failures)} test(s) failed: {failures}")
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
