#!/usr/bin/env python3
"""
data_logger 阶段 1 单元测试：
- 验证新增字段 target_roll_deg / target_pitch_deg / target_yaw_deg /
  target_roll_rate_rad_s / target_pitch_rate_rad_s / target_yaw_rate_rad_s 写入
- 验证 stale > 200ms 时 *_valid = False
- 验证 enable_pymavlink_targets=False 时不启动 pymavlink 任务
"""

import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_logger import DataLogger, _quat_to_euler_rad, _TARGET_STALE_USEC


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
        enable_pymavlink_targets=False,  # 单测里不真实启动 pymavlink
    )
    logger.is_logging = True
    logger.start_time = time.time() - 1.0
    logger._fw_metrics = SimpleNamespace(airspeed_m_s=15.0, groundspeed_m_s=14.5)
    logger._attitude = SimpleNamespace(roll_deg=2.0, pitch_deg=-3.0, yaw_deg=45.0)
    logger._gyro = (0.01, 0.02, 0.03)
    logger._velocity_ned = (10.0, 9.0, 0.5)
    logger._actuator_ctrls = (0.1, 0.2, 0.0, 0.6)
    return logger


def test_quat_to_euler_identity():
    """单位四元数应映射到 (0,0,0)。"""
    r, p, y = _quat_to_euler_rad((1.0, 0.0, 0.0, 0.0))
    assert abs(r) < 1e-9 and abs(p) < 1e-9 and abs(y) < 1e-9, (r, p, y)


def test_quat_to_euler_yaw_90():
    """绕 z 轴 90° 的四元数 → yaw=π/2。"""
    half = math.cos(math.pi / 4)
    s = math.sin(math.pi / 4)
    # q = (cos(yaw/2), 0, 0, sin(yaw/2))
    r, p, y = _quat_to_euler_rad((half, 0.0, 0.0, s))
    assert abs(r) < 1e-6 and abs(p) < 1e-6, (r, p)
    assert abs(y - math.pi / 2) < 1e-6, y


def test_snapshot_no_target_falls_back_zero():
    """无 pymavlink 目标量时，target_* 字段为 0 且 *_valid=False。

    actuator_actual_* 走 fallback：当 JSBSim Telnet 不可用、_actual_actuator=None，
    且 _actuator_ctrls 已有值时，aileron/elevator/rudder = ctrl × max_deflection。
    actuator_actual_valid 仍为 False（标记这是代理值非真值）。
    """
    logger = make_logger()
    logger._snapshot(t=0.5)
    assert len(logger.data_buffer) == 1
    e = logger.data_buffer[0]
    assert e["target_roll_deg"] == 0.0
    assert e["target_pitch_deg"] == 0.0
    assert e["target_yaw_deg"] == 0.0
    assert e["target_attitude_valid"] is False
    assert e["target_attitude_age_ms"] == -1.0
    assert e["target_body_rates_valid"] is False
    # 阶段 2 字段也应在
    assert e["imu_accel_body_x"] == 0.0
    assert e["imu_accel_valid"] is False
    assert e["target_velocity_north"] == 0.0
    # _actuator_ctrls=(0.1, 0.2, 0.0, 0.6) -> fallback 用 ctrl×max_deflection
    assert abs(e["aileron_actual_rad"] - 0.1 * 0.35) < 1e-9
    assert abs(e["elevator_actual_rad"] - 0.2 * 0.30) < 1e-9
    assert e["rudder_actual_rad"] == 0.0  # yaw_ctrl=0
    assert abs(e["throttle_actual_norm"] - 0.6) < 1e-9
    assert e["actuator_actual_valid"] is False  # 代理值，非真值
    assert e["actuator_actual_source"] == "px4_command_proxy"


def test_snapshot_with_fresh_target():
    """目标量刚到达 (age≈0)，*_valid=True，数值正确转换。"""
    logger = make_logger()
    now = int(time.time() * 1e6)
    # 目标姿态：roll=10°, pitch=-5°, yaw=30° (rad)
    logger._target_attitude = (math.radians(10.0), math.radians(-5.0), math.radians(30.0))
    logger._target_attitude_ts = now
    logger._target_body_rates = (0.05, -0.04, 0.07)
    logger._target_body_rates_ts = now
    logger._snapshot(t=1.0)
    e = logger.data_buffer[0]
    assert abs(e["target_roll_deg"] - 10.0) < 1e-3
    assert abs(e["target_pitch_deg"] + 5.0) < 1e-3
    assert abs(e["target_yaw_deg"] - 30.0) < 1e-3
    assert e["target_attitude_valid"] is True
    assert 0 <= e["target_attitude_age_ms"] < 50  # ≈0 ms
    assert abs(e["target_roll_rate_rad_s"] - 0.05) < 1e-6
    assert e["target_body_rates_valid"] is True


def test_snapshot_with_stale_target():
    """目标量已过期 (age > 200ms) → *_valid=False。"""
    logger = make_logger()
    now = int(time.time() * 1e6)
    stale_ts = now - (_TARGET_STALE_USEC + 50_000)  # 250ms 之前
    logger._target_attitude = (math.radians(20.0), 0.0, 0.0)
    logger._target_attitude_ts = stale_ts
    logger._target_body_rates = (0.0, 0.0, 0.0)
    logger._target_body_rates_ts = stale_ts
    logger._snapshot(t=2.0)
    e = logger.data_buffer[0]
    # 数值仍写入（保留最后已知值），但 valid=False
    assert abs(e["target_roll_deg"] - 20.0) < 1e-3
    assert e["target_attitude_valid"] is False
    assert e["target_attitude_age_ms"] >= 200.0


def test_snapshot_partial_target():
    """只有 target_attitude 到达，target_body_rates 仍为 None。"""
    logger = make_logger()
    now = int(time.time() * 1e6)
    logger._target_attitude = (0.0, 0.0, 0.0)
    logger._target_attitude_ts = now
    logger._snapshot(t=3.0)
    e = logger.data_buffer[0]
    assert e["target_attitude_valid"] is True
    assert e["target_body_rates_valid"] is False
    assert e["target_roll_rate_rad_s"] == 0.0


if __name__ == "__main__":
    tests = [
        test_quat_to_euler_identity,
        test_quat_to_euler_yaw_90,
        test_snapshot_no_target_falls_back_zero,
        test_snapshot_with_fresh_target,
        test_snapshot_with_stale_target,
        test_snapshot_partial_target,
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
