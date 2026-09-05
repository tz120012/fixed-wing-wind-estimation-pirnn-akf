"""风场配置生成（按 split 决定风速 / 方向 / 湍流强度 / 阵风）。

迁移自 ``generate_dataset.py:327-379``。本模块为纯函数，仅依赖 ``random``、``numpy``。
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

from ..config.dataset import get_split_profile, load_dataset_config


def generate_wind_config(
    dataset_type: str = "train",
    seed: Optional[int] = None,
    run_index: Optional[int] = None,
) -> dict:
    """按 split 生成背景风，默认面向小泡沫固定翼的安全风包线。

    Parameters
    ----------
    dataset_type
        ``train | val | test_id | test_ood``。
    seed
        若给定，会同步设置 ``random`` 与 ``numpy.random`` 的种子（保证可重现）。
    run_index
        本轮在该 split 内的序号（0-based）。仅对 ``test_ood`` 生效：使用分层方向
        采样（8 个 45° 象限循环），避免小样本量（30 轮）下的方向偏差。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cfg = load_dataset_config()
    fp = cfg.get("flight_parameters", {})
    turb_cfg = cfg.get("turbulence", {})
    split_cfg = get_split_profile(cfg, dataset_type)

    ws_range = split_cfg.get("wind_speed_range", fp.get("wind_speed_range", [0.8, 2.5]))
    wd_range = fp.get("wind_direction_range", [0, 360])
    wdown_range = split_cfg.get("wind_down_range", fp.get("wind_down_range", [-0.3, 0.3]))
    turb_gain_range = split_cfg.get(
        "turbulence_gain_range", turb_cfg.get("gain_range", [0.25, 0.75])
    )

    speed = random.uniform(*ws_range)
    if dataset_type == "test_ood" and run_index is not None:
        octant = run_index % 8
        direction = octant * 45.0 + random.uniform(0.0, 45.0)
    else:
        direction = random.uniform(*wd_range)
    wn = speed * np.cos(np.deg2rad(direction))
    we = speed * np.sin(np.deg2rad(direction))
    w_down = random.uniform(*wdown_range)
    return {
        "wind_north": wn,
        "wind_east": we,
        "wind_down": w_down,
        "wind_speed": speed,
        "wind_direction": direction,
        "turbulence_gain": round(random.uniform(*turb_gain_range), 2),
    }


def sample_gust_start_time(segment_duration: float, gust_duration: float, preferred_range) -> float:
    """为阵风采样一个能完整落在段内的开始时刻。

    迁移自 ``generate_dataset.py:368-379``。
    """
    preferred_low, preferred_high = preferred_range
    preferred_low = max(2.0, float(preferred_low))
    preferred_high = max(preferred_low, float(preferred_high))

    latest_start = max(2.0, float(segment_duration) - float(gust_duration) - 3.0)
    low = min(preferred_low, latest_start)
    high = min(preferred_high, latest_start)
    if high < low:
        high = low
    return random.uniform(low, high)
