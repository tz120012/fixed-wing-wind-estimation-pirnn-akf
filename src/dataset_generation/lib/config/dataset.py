"""数据集配置：``configs/dataset_config.json`` 的加载与默认值合并。

设计要点：
    1. ``DEFAULT_DATASET_CONFIG`` 是兜底配置（小型泡沫固定翼）；
    2. 若 ``configs/dataset_config.json`` 存在则递归合并覆盖；
    3. 加载结果用 ``functools.lru_cache`` 做进程内缓存（替代原先挂在
       ``_load_dataset_config._cache`` 上的隐式单例）；
    4. ``deep_merge_dict`` / ``get_split_profile`` 暴露为可在测试中直接调用的纯函数。
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


_DEFAULT_DATASET_JSON = (
    Path(__file__).resolve().parents[2] / "configs" / "dataset_config.json"
)


DEFAULT_DATASET_CONFIG: dict = {
    "meta": {
        "airframe_profile": "small_foam_fixed_wing",
        "notes": "默认参数面向低空、低速的小型泡沫固定翼；若存在 dataset_config.json，会以文件中的配置覆盖这里的默认值。",
    },
    "flight_parameters": {
        "speed_range": [10.5, 13.5],
        "altitude_range": [50, 85],
        "wind_speed_range": [0.8, 2.5],
        "wind_direction_range": [0, 360],
        "wind_down_range": [-0.3, 0.3],
    },
    "gust_parameters": {
        "start_time_range": [12, 30],
        "id_magnitude_range": [1.0, 2.5],
        "id_duration_range": [18.0, 40.0],
        "ood_magnitude_range": [2.5, 4.0],
        "ood_duration_range_short": [18.0, 35.0],
        "ood_duration_range_long": [35.0, 70.0],
    },
    "turbulence": {
        "gain_range": [0.25, 0.75],
    },
    "maneuver_types": {
        "straight_line": {
            "probability": 0.60,
            "duration_range": [60, 100],
        },
        "orbit": {
            "probability": 0.15,
            "radius_range": [45, 85],
            "duration_range": [80, 130],
        },
        "figure_eight": {
            "probability": 0.10,
            "radius_range": [35, 60],
            "duration_range": [110, 160],
        },
        "climb_descent": {
            "probability": 0.15,
            "climb_rate_range": [0.8, 1.6],
            "altitude_delta_range": [-18, 18],
            "level_extra_range": [20, 40],
        },
    },
    "split_profiles": {
        "train": {
            "wind_speed_range": [0.8, 2.8],
            "wind_down_range": [-0.3, 0.3],
            "turbulence_gain_range": [0.25, 0.7],
        },
        "val": {
            "wind_speed_range": [0.8, 2.8],
            "wind_down_range": [-0.35, 0.35],
            "turbulence_gain_range": [0.25, 0.75],
        },
        "test_id": {
            "wind_speed_range": [0.8, 2.8],
            "wind_down_range": [-0.3, 0.3],
            "turbulence_gain_range": [0.25, 0.75],
        },
        "test_ood": {
            "wind_speed_range": [3.0, 4.5],
            "wind_down_range": [-0.5, 0.5],
            "turbulence_gain_range": [0.4, 1.0],
        },
    },
}


def deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """递归合并：``override`` 覆盖 ``base``，缺失项继承默认值。"""
    merged = copy.deepcopy(dict(base))
    if not isinstance(override, Mapping):
        return merged
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def get_split_profile(cfg: Mapping[str, Any], dataset_type: str) -> dict:
    """读取 split 级别配置，不存在则回退为空 dict。"""
    split_profiles = cfg.get("split_profiles", {})
    if not isinstance(split_profiles, Mapping):
        return {}
    profile = split_profiles.get(dataset_type, {})
    return dict(profile) if isinstance(profile, Mapping) else {}


@lru_cache(maxsize=4)
def _cached_load(path_str: str) -> dict:
    cfg_path = Path(path_str)
    file_cfg: dict = {}
    if cfg_path.exists():
        try:
            file_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[Config] dataset_config.json 解析失败，将使用内置默认配置: {e}")
    return deep_merge_dict(DEFAULT_DATASET_CONFIG, file_cfg)


def load_dataset_config(path: "str | Path | None" = None) -> dict:
    """加载并缓存数据集配置。

    Parameters
    ----------
    path
        若为 ``None``，使用包内默认 ``configs/dataset_config.json``。
    """
    target = Path(path) if path is not None else _DEFAULT_DATASET_JSON
    return _cached_load(str(target))


def reset_cache() -> None:
    """清空缓存（仅供测试使用）。"""
    _cached_load.cache_clear()
