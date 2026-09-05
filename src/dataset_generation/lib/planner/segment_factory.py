"""段配置生成（机动类型、时长、阵风、转弯标签）。

迁移自 ``generate_dataset.py:44-78, 383-631``。本模块为纯函数。
"""

from __future__ import annotations

import copy
import random
from typing import List, Optional, Tuple

import numpy as np

from ..config.dataset import get_split_profile, load_dataset_config
from .wind_factory import sample_gust_start_time


_TURNING_MANEUVERS = {"orbit", "figure_eight"}
_NON_TURNING_MANEUVERS = {"straight_line", "climb_descent"}


def derive_turn_labels(maneuver_type: str) -> Tuple[str, int]:
    """从机动类型派生转弯状态标签 (``turn_state``, ``turn_class``)。

    迁移自 ``generate_dataset.py:66-78``。
    """
    if maneuver_type in _TURNING_MANEUVERS:
        return "turning", 1
    if maneuver_type in _NON_TURNING_MANEUVERS:
        return "non_turning", 0
    return "unknown", -1


def generate_one_segment_config(
    flight_id: int,
    dataset_type: str,
    config_type: str,
    gust_ood: bool,
    altitude: float,
    seed: Optional[int] = None,
) -> dict:
    """生成单段机动配置（不含风场，风场由会话级统一填入）。

    Parameters
    ----------
    config_type
        ``'steady'`` 或 ``'gust'``。
    gust_ood
        是否 OOD 阵风（更强 / 更长）。
    seed
        可选随机种子，确保可重现。

    迁移自 ``generate_dataset.py:383-475``。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cfg_all = load_dataset_config()
    fp = cfg_all.get("flight_parameters", {})
    gust_p = cfg_all.get("gust_parameters", {})
    maneuver_cfg = cfg_all.get("maneuver_types", {})
    split_cfg = get_split_profile(cfg_all, dataset_type)

    speed_range = split_cfg.get("speed_range", fp.get("speed_range", [10.5, 13.5]))
    cfg = {
        "id": f"{dataset_type}_{flight_id:04d}",
        "flight_id": flight_id,
        "dataset_type": dataset_type,
        "config_type": config_type,
        "gust_ood": gust_ood,
        "altitude": altitude,
        "speed": random.uniform(*speed_range),
        "heading": random.uniform(0, 360),
    }

    m_straight = maneuver_cfg.get("straight_line", {})
    m_orbit = maneuver_cfg.get("orbit", {})
    m_fig8 = maneuver_cfg.get("figure_eight", {})
    m_climb = maneuver_cfg.get("climb_descent", {})
    maneuver = random.choices(
        ["straight_line", "orbit", "figure_eight", "climb_descent"],
        weights=[
            m_straight.get("probability", 0.4),
            m_orbit.get("probability", 0.3),
            m_fig8.get("probability", 0.2),
            m_climb.get("probability", 0.1),
        ],
        k=1,
    )[0]
    cfg["maneuver_type"] = maneuver

    if maneuver == "straight_line":
        cfg["duration"] = random.uniform(*m_straight.get("duration_range", [60, 100]))
    elif maneuver == "orbit":
        cfg["radius"] = random.uniform(*m_orbit.get("radius_range", [45, 85]))
        cfg["direction"] = random.choice(["cw", "ccw"])
        cfg["duration"] = random.uniform(*m_orbit.get("duration_range", [80, 130]))
    elif maneuver == "figure_eight":
        cfg["radius"] = random.uniform(*m_fig8.get("radius_range", [35, 60]))
        cfg["duration"] = random.uniform(*m_fig8.get("duration_range", [110, 160]))
    else:
        MIN_ALT = 40.0
        alt_delta_range = m_climb.get("altitude_delta_range", [-18, 18])
        target_alt = altitude + random.uniform(*alt_delta_range)
        if target_alt < MIN_ALT:
            target_alt = MIN_ALT
        cfg["target_altitude"] = target_alt

        cr_range = m_climb.get("climb_rate_range", [0.8, 1.6])
        alt_diff = target_alt - altitude
        cr = random.uniform(*cr_range)
        cfg["climb_rate"] = cr if alt_diff >= 0 else -cr

        climb_time = abs(alt_diff) / cr if cr > 0 else 20
        level_extra = random.uniform(*m_climb.get("level_extra_range", [20, 40]))
        cfg["duration"] = climb_time + level_extra

    if config_type == "gust":
        gust_start_range = split_cfg.get(
            "gust_start_time_range", gust_p.get("start_time_range", [20, 45])
        )
        if gust_ood:
            magnitude = random.uniform(*gust_p.get("ood_magnitude_range", [3.2, 4.8]))
            duration = (
                random.uniform(*gust_p.get("ood_duration_range_long", [4.0, 6.0]))
                if random.random() < 0.5
                else random.uniform(*gust_p.get("ood_duration_range_short", [2.5, 4.0]))
            )
        else:
            magnitude = random.uniform(*gust_p.get("id_magnitude_range", [1.5, 3.0]))
            duration = random.uniform(*gust_p.get("id_duration_range", [2.5, 4.5]))

        cfg["gust"] = {
            "magnitude": magnitude,
            "duration": duration,
            "direction": random.uniform(0, 360),
            "start_time": sample_gust_start_time(cfg["duration"], duration, gust_start_range),
        }
    return cfg


def generate_segment_configs_for_run(
    dataset_type: str,
    base_flight_id: int,
    segment_types: List[str],
    altitude: Optional[float] = None,
    seed: Optional[int] = None,
) -> List[dict]:
    """根据本轮的 segment_types 生成 5 个段配置。

    迁移自 ``generate_dataset.py:478-508``。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    cfg = load_dataset_config()
    fp = cfg.get("flight_parameters", {})
    split_cfg = get_split_profile(cfg, dataset_type)

    if altitude is None:
        alt_range = split_cfg.get("altitude_range", fp.get("altitude_range", [50, 85]))
        altitude = random.uniform(*alt_range)

    configs: List[dict] = []
    for i, seg_type in enumerate(segment_types):
        is_steady = seg_type == "steady"
        gust_ood = seg_type == "gust_ood"
        seg_seed = seed * 10 + i if seed is not None else None
        configs.append(
            generate_one_segment_config(
                flight_id=base_flight_id + i,
                dataset_type=dataset_type,
                config_type="steady" if is_steady else "gust",
                gust_ood=gust_ood,
                altitude=altitude,
                seed=seg_seed,
            )
        )
    return configs


def split_segment_configs_for_sorties(segment_configs: List[dict]) -> List[List[Tuple[int, dict]]]:
    """将逻辑上的 5 段拆成若干个 sortie，每个 sortie 最多一个阵风段。

    迁移自 ``generate_dataset.py:607-631``。
    """
    groups: List[List[Tuple[int, dict]]] = []
    current_group: List[Tuple[int, dict]] = []
    has_gust = False

    for idx, seg in enumerate(segment_configs):
        seg_copy = copy.deepcopy(seg)
        seg_has_gust = bool(seg_copy.get("gust"))
        if seg_has_gust and has_gust and current_group:
            groups.append(current_group)
            current_group = []
            has_gust = False

        current_group.append((idx, seg_copy))
        has_gust = has_gust or seg_has_gust

        if seg_has_gust:
            groups.append(current_group)
            current_group = []
            has_gust = False

    if current_group:
        groups.append(current_group)
    return groups
