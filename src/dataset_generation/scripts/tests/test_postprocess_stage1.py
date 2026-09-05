#!/usr/bin/env python3
"""
postprocess 阶段 1 单元测试：
- 验证 CSV_HEADER 包含新增的 6 列
- 验证 row_from_entry 在含/不含 target_* 字段时都能正确工作（向后兼容）
- 验证 target_*_deg → target_*_rad 转换正确
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from postprocess_to_jsbsim_csv import CSV_HEADER, row_from_entry


NEW_STAGE1_COLS = [
    "target_roll_rad",
    "target_pitch_rad",
    "target_yaw_rad",
    "target_p_rad_s",
    "target_q_rad_s",
    "target_r_rad_s",
]


def base_entry(**override):
    e = {
        "timestamp": 0.5,
        "wind_north": 1.0,
        "wind_east": 0.5,
        "wind_down": 0.0,
        "velocity_north": 10.0,
        "velocity_east": 9.0,
        "velocity_down": 0.5,
        "airspeed_m_s": 15.0,
        "alt_rel": 60.0,
        "lat": 47.0,
        "lon": 8.0,
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
        "wind_regime": "steady",
        "gust_phase": "none",
        "gust_factor": 0.0,
        "base_wind_north": 1.0,
        "base_wind_east": 0.5,
        "base_wind_down": 0.0,
        "gust_delta_north": 0.0,
        "gust_delta_east": 0.0,
        "gust_delta_down": 0.0,
        "maneuver_regime": "straight_line",
        "turn_state": "non_turning",
        "turn_class": 0,
    }
    e.update(override)
    return e


def col_idx(name):
    return CSV_HEADER.index(name)


def test_csv_header_has_stage1_cols():
    """CSV_HEADER 必须包含 6 个阶段1 新列。"""
    for col in NEW_STAGE1_COLS:
        assert col in CSV_HEADER, f"missing column: {col}"
    # 阶段 1+2 后总列数固定 52；阶段 1 列至少存在（位置由 col_idx 查找）
    assert len(CSV_HEADER) >= 42


def test_row_backward_compatible_no_target():
    """旧 JSON 不含 target_* 时，row 长度 == header 长度，且阶段1新列全 0。"""
    e = base_entry()
    row = row_from_entry(e, t0=0.0)
    assert len(row) == len(CSV_HEADER), f"row 长度与 header 不匹配: {len(row)} vs {len(CSV_HEADER)}"
    for col in NEW_STAGE1_COLS:
        assert row[col_idx(col)] == 0.0, f"{col} 缺失时应为 0"


def test_row_with_target_correct_rad_conversion():
    """含 target_*_deg 时，CSV 列以 rad 输出，且数值正确。"""
    e = base_entry(
        target_roll_deg=10.0,
        target_pitch_deg=-5.0,
        target_yaw_deg=45.0,
        target_roll_rate_rad_s=0.07,
        target_pitch_rate_rad_s=-0.03,
        target_yaw_rate_rad_s=0.12,
    )
    row = row_from_entry(e, t0=0.0)
    assert abs(row[col_idx("target_roll_rad")] - math.radians(10.0)) < 1e-9
    assert abs(row[col_idx("target_pitch_rad")] - math.radians(-5.0)) < 1e-9
    assert abs(row[col_idx("target_yaw_rad")] - math.radians(45.0)) < 1e-9
    assert abs(row[col_idx("target_p_rad_s")] - 0.07) < 1e-9
    assert abs(row[col_idx("target_q_rad_s")] + 0.03) < 1e-9
    assert abs(row[col_idx("target_r_rad_s")] - 0.12) < 1e-9


def test_row_with_partial_target_zero_default():
    """只有部分 target_* 字段，缺失字段填 0。"""
    e = base_entry(target_roll_deg=20.0)
    row = row_from_entry(e, t0=0.0)
    assert abs(row[col_idx("target_roll_rad")] - math.radians(20.0)) < 1e-9
    assert row[col_idx("target_pitch_rad")] == 0.0
    assert row[col_idx("target_yaw_rad")] == 0.0
    assert row[col_idx("target_p_rad_s")] == 0.0


def test_row_target_with_none_value():
    """target_* 字段值为 None（合法 JSON 中可能出现）时，应填 0。"""
    e = base_entry(
        target_roll_deg=None, target_pitch_deg=None, target_yaw_deg=None,
        target_roll_rate_rad_s=None, target_pitch_rate_rad_s=None, target_yaw_rate_rad_s=None,
    )
    row = row_from_entry(e, t0=0.0)
    for col in NEW_STAGE1_COLS:
        assert row[col_idx(col)] == 0.0


if __name__ == "__main__":
    tests = [
        test_csv_header_has_stage1_cols,
        test_row_backward_compatible_no_target,
        test_row_with_target_correct_rad_conversion,
        test_row_with_partial_target_zero_default,
        test_row_target_with_none_value,
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
