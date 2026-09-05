#!/usr/bin/env python3
"""
data_logger 阶段 2 单元测试：
- 目标速度 NED：fresh / stale 验证
- IMU 加速度：fresh / stale 验证
- 实际舵面（JSBSim Telnet 缓存）：fresh / stale 验证
- 三类同时缺失/部分到达
"""

import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_logger import DataLogger, _TARGET_STALE_USEC


def make_logger():
    logger = DataLogger(
        drone=None,
        wind_north=1.0,
        wind_east=0.5,
        wind_down=0.1,
        gust_params=None,
        wind_truth_csv_path=None,
        turn_state="non_turning",
        turn_class=0,
        enable_pymavlink_targets=False,
    )
    logger.is_logging = True
    logger.start_time = time.time() - 1.0
    logger._fw_metrics = SimpleNamespace(airspeed_m_s=15.0, groundspeed_m_s=14.5)
    logger._attitude = SimpleNamespace(roll_deg=2.0, pitch_deg=-3.0, yaw_deg=45.0)
    logger._gyro = (0.01, 0.02, 0.03)
    logger._velocity_ned = (10.0, 9.0, 0.5)
    logger._actuator_ctrls = (0.1, 0.2, 0.0, 0.6)
    return logger


def test_target_velocity_from_nav_controller():
    """NAV_CONTROLLER_OUTPUT fresh → target_velocity 由 nav_bearing/nav_pitch 重建。"""
    logger = make_logger()
    now = int(time.time() * 1e6)
    # nav_bearing=90° (East), nav_pitch=0° → vn=0, ve=airspeed, vd=0
    logger._nav_ctrl = (90.0, 0.0, 0.5, -2.0, 0.1)
    logger._nav_ctrl_ts = now
    logger._snapshot(t=0.5)
    e = logger.data_buffer[0]
    # airspeed=15.0 (make_logger 里设置的)
    assert abs(e["target_velocity_north"]) < 0.01              # cos(90°)≈0
    assert abs(e["target_velocity_east"] - 15.0) < 0.01       # sin(90°)=1
    assert abs(e["target_velocity_down"]) < 0.01              # sin(0°)=0
    assert e["target_velocity_valid"] is True
    assert e["target_velocity_source"] == "nav_controller"
    assert "nav_bearing_deg" in e
    assert "aspd_error_m_s" in e


def test_target_velocity_fallback_to_position_target():
    """NAV_CONTROLLER 无数据时，回退到 POSITION_TARGET_LOCAL_NED。"""
    logger = make_logger()
    now = int(time.time() * 1e6)
    logger._target_velocity_ned = (12.0, 8.0, 0.3)
    logger._target_velocity_ned_ts = now
    logger._snapshot(t=0.5)
    e = logger.data_buffer[0]
    assert abs(e["target_velocity_north"] - 12.0) < 1e-6
    assert e["target_velocity_source"] == "position_target"
    assert e["target_velocity_valid"] is True


def test_target_velocity_stale():
    """NAV_CONTROLLER 过期 + POSITION_TARGET 也无数据 → valid=False, source=none。"""
    logger = make_logger()
    stale = int(time.time() * 1e6) - (_TARGET_STALE_USEC + 100_000)
    logger._nav_ctrl = (45.0, 5.0, 0.0, 0.0, 0.0)
    logger._nav_ctrl_ts = stale
    logger._snapshot(t=0.5)
    e = logger.data_buffer[0]
    assert e["target_velocity_valid"] is False
    assert e["target_velocity_source"] == "none"


def test_imu_accel_fresh():
    logger = make_logger()
    now = int(time.time() * 1e6)
    logger._highres_imu_accel = (0.5, -0.3, -9.7)
    logger._highres_imu_ts = now
    logger._snapshot(t=1.0)
    e = logger.data_buffer[0]
    assert abs(e["imu_accel_body_x"] - 0.5) < 1e-6
    assert abs(e["imu_accel_body_y"] + 0.3) < 1e-6
    assert abs(e["imu_accel_body_z"] + 9.7) < 1e-6
    assert e["imu_accel_valid"] is True


def test_actual_actuator_fresh():
    logger = make_logger()
    now = int(time.time() * 1e6)
    logger._actual_actuator = (math.radians(2.0), math.radians(-1.0),
                               math.radians(0.5), 0.65)
    logger._actual_actuator_ts = now
    logger._snapshot(t=2.0)
    e = logger.data_buffer[0]
    assert abs(e["aileron_actual_rad"] - math.radians(2.0)) < 1e-6
    assert abs(e["elevator_actual_rad"] + math.radians(1.0)) < 1e-6
    assert abs(e["rudder_actual_rad"] - math.radians(0.5)) < 1e-6
    assert abs(e["throttle_actual_norm"] - 0.65) < 1e-6
    assert e["actuator_actual_valid"] is True


def test_all_stage2_missing():
    """阶段 2 所有量都缺失时，target_velocity / imu_accel 是 0 且 *_valid=False。

    actuator_actual 走 fallback：_actuator_ctrls=(0.1, 0.2, 0.0, 0.6) 已设置时，
    aileron/elevator/rudder = ctrl × max_deflection；throttle = ctrl 直接 clamp 到 [0,1]；
    actuator_actual_valid 仍为 False（代理值标记）。
    """
    logger = make_logger()
    logger._snapshot(t=3.0)
    e = logger.data_buffer[0]
    assert e["target_velocity_valid"] is False
    assert e["imu_accel_valid"] is False
    assert e["actuator_actual_valid"] is False  # 代理值，非真值
    assert e["target_velocity_north"] == 0.0
    assert e["imu_accel_body_x"] == 0.0
    # _actuator_ctrls=(0.1, 0.2, 0.0, 0.6) -> fallback 缩放
    assert abs(e["aileron_actual_rad"] - 0.1 * 0.35) < 1e-9
    assert abs(e["throttle_actual_norm"] - 0.6) < 1e-9
    assert e["actuator_actual_source"] == "px4_command_proxy"


def test_partial_stage2_only_imu():
    """只有 IMU 到达，其他仍缺失。"""
    logger = make_logger()
    now = int(time.time() * 1e6)
    logger._highres_imu_accel = (1.0, 0.0, -9.8)
    logger._highres_imu_ts = now
    logger._snapshot(t=4.0)
    e = logger.data_buffer[0]
    assert e["imu_accel_valid"] is True
    assert e["target_velocity_valid"] is False
    assert e["actuator_actual_valid"] is False


if __name__ == "__main__":
    tests = [
        test_target_velocity_from_nav_controller,
        test_target_velocity_fallback_to_position_target,
        test_target_velocity_stale,
        test_imu_accel_fresh,
        test_actual_actuator_fresh,
        test_all_stage2_missing,
        test_partial_stage2_only_imu,
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
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
