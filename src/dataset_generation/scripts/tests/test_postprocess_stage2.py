#!/usr/bin/env python3
"""
postprocess 阶段 2 单元测试：
- CSV_HEADER 长度=52（原 36 + 阶段1 6 + 阶段2 10）
- 含完整 stage2 字段时 row 末 10 列正确
- 缺失 stage2 字段时末 10 列全 0（向后兼容）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from postprocess_to_jsbsim_csv import CSV_HEADER, row_from_entry


STAGE2_COLS = [
    "target_vn", "target_ve", "target_vd",
    "aileron_actual", "elevator_actual", "rudder_actual", "throttle_actual",
    "imu_ax", "imu_ay", "imu_az",
]


def base_entry(**override):
    e = {
        "timestamp": 0.5,
        "wind_north": 1.0, "wind_east": 0.5, "wind_down": 0.0,
        "velocity_north": 10.0, "velocity_east": 9.0, "velocity_down": 0.5,
        "airspeed_m_s": 15.0, "alt_rel": 60.0, "lat": 47.0, "lon": 8.0,
        "roll_deg": 5.0, "pitch_deg": -2.0, "yaw_deg": 30.0,
        "roll_rate_rad_s": 0.01, "pitch_rate_rad_s": -0.02, "yaw_rate_rad_s": 0.03,
        "roll_ctrl": 0.1, "pitch_ctrl": 0.05, "yaw_ctrl": 0.0, "throttle_ctrl": 0.7,
        "wind_regime": "steady", "gust_phase": "none", "gust_factor": 0.0,
        "base_wind_north": 1.0, "base_wind_east": 0.5, "base_wind_down": 0.0,
        "gust_delta_north": 0.0, "gust_delta_east": 0.0, "gust_delta_down": 0.0,
        "maneuver_regime": "straight_line", "turn_state": "non_turning", "turn_class": 0,
    }
    e.update(override)
    return e


def test_csv_header_52cols():
    assert len(CSV_HEADER) == 52, f"expected 52, got {len(CSV_HEADER)}"
    for c in STAGE2_COLS:
        assert c in CSV_HEADER, f"missing: {c}"


def test_row_full_stage2():
    e = base_entry(
        target_velocity_north=12.0, target_velocity_east=8.0, target_velocity_down=0.3,
        aileron_actual_rad=0.05, elevator_actual_rad=-0.02,
        rudder_actual_rad=0.01, throttle_actual_norm=0.65,
        imu_accel_body_x=0.5, imu_accel_body_y=-0.3, imu_accel_body_z=-9.7,
    )
    row = row_from_entry(e, t0=0.0)
    assert len(row) == 52
    last10 = row[-10:]
    assert abs(last10[0] - 12.0) < 1e-9
    assert abs(last10[1] - 8.0) < 1e-9
    assert abs(last10[2] - 0.3) < 1e-9
    assert abs(last10[3] - 0.05) < 1e-9
    assert abs(last10[6] - 0.65) < 1e-9
    assert abs(last10[7] - 0.5) < 1e-9
    assert abs(last10[9] + 9.7) < 1e-9


def test_row_missing_stage2_zero():
    e = base_entry()
    row = row_from_entry(e, t0=0.0)
    assert len(row) == 52
    for v in row[-10:]:
        assert v == 0.0


def test_row_partial_stage2():
    """只设置了 IMU，速度/舵面缺失。"""
    e = base_entry(
        imu_accel_body_x=1.5, imu_accel_body_y=0.2, imu_accel_body_z=-9.81,
    )
    row = row_from_entry(e, t0=0.0)
    last10 = row[-10:]
    # 0..6 应全 0（速度/舵面/油门）
    for i in range(7):
        assert last10[i] == 0.0
    # 7..9 是 IMU
    assert abs(last10[7] - 1.5) < 1e-9
    assert abs(last10[9] + 9.81) < 1e-9


if __name__ == "__main__":
    tests = [
        test_csv_header_52cols,
        test_row_full_stage2,
        test_row_missing_stage2_zero,
        test_row_partial_stage2,
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
