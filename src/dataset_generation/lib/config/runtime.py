"""运行期配置加载（``configs/runtime.yaml``）。

提供:
    - :class:`RuntimeConfig` 数据类，强类型读取所有 sleep/timeout/retry/port/threshold；
    - :func:`load_runtime_config` 从 YAML 路径或默认位置加载并校验。

设计要点：
    1. 字段名与 YAML key 一一对应（直接 ``RuntimeConfig(**section)``）。
    2. 任何字段缺失抛 ``KeyError``，不静默使用默认值——避免重构期间忘改 YAML 时出现"看似工作其实跑错"的隐藏问题。
    3. 加载结果应当不可变；若运行期需要覆盖，请显式 ``dataclasses.replace``。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml


_DEFAULT_RUNTIME_YAML = (
    Path(__file__).resolve().parents[2] / "configs" / "runtime.yaml"
)


@dataclass(frozen=True)
class SleepsConfig:
    px4_post_start_s: float
    ekf_warmup_s: float
    between_segments_s: float
    between_sorties_s: float
    pkill_grace_s: float
    cleanup_post_stop_s: float
    pre_sortie_warmup_s: float
    segment_retry_backoff_s: float


@dataclass(frozen=True)
class RetriesConfig:
    mavsdk_connect: int
    arm: int
    takeoff: int
    segment_sample: int
    jsbsim_verify: int
    gps_verify: int
    sortie_prepare: int = 3


@dataclass(frozen=True)
class TimeoutsConfig:
    px4_terminate_s: float
    px4_kill_wait_s: float
    pkill_timeout_s: float
    mavsdk_health_s: float
    arm_check_s: float
    local_position_s: float
    takeoff_total_s: float
    round_total_s: float
    gps_subscribe_s: float
    bridge_ready_s: float
    airspeed_ready_s: float = 30.0
    airspeed_stable_s: float = 3.0
    position_stream_max_failures: int = 30
    climb_descent_event_grace_s: float = 8.0


@dataclass(frozen=True)
class QualityGatesConfig:
    min_valid_log_hz: float
    min_samples_floor: int
    min_coverage_ratio: float
    min_coverage_floor_s: float
    airspeed_filled_ratio_threshold: float
    target_stale_us: int
    wind_truth_align_window_us: int


@dataclass(frozen=True)
class PortsConfig:
    mavsdk_udp: int
    mavsdk_grpc: int
    pymavlink_udp: int
    jsbsim_bridge: int
    jsbsim_telnet: int


@dataclass(frozen=True)
class LoggingConfig:
    runtime_jsonl_path: str
    failure_jsonl_path: str
    console_level: str
    jsonl_level: str


@dataclass(frozen=True)
class RuntimeConfig:
    sleeps: SleepsConfig
    retries: RetriesConfig
    timeouts: TimeoutsConfig
    quality_gates: QualityGatesConfig
    ports: PortsConfig
    logging: LoggingConfig
    source_path: str = field(default="")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source_path: str = "") -> "RuntimeConfig":
        try:
            return cls(
                sleeps=SleepsConfig(**data["sleeps"]),
                retries=RetriesConfig(**data["retries"]),
                timeouts=TimeoutsConfig(**data["timeouts"]),
                quality_gates=QualityGatesConfig(**data["quality_gates"]),
                ports=PortsConfig(**data["ports"]),
                logging=LoggingConfig(**data["logging"]),
                source_path=source_path,
            )
        except KeyError as e:
            raise KeyError(
                f"runtime.yaml 缺少必需字段: {e}. 请同步检查 "
                f"{_DEFAULT_RUNTIME_YAML} 与 lib/config/runtime.py 的 dataclass 字段。"
            ) from e
        except TypeError as e:
            raise TypeError(
                f"runtime.yaml 字段类型/数量与 dataclass 不匹配: {e}"
            ) from e

    def with_overrides(self, **section_overrides: Mapping[str, Any]) -> "RuntimeConfig":
        """局部覆盖某些 section（用于测试或临时调试）。

        例：``rc.with_overrides(timeouts={"round_total_s": 60})``
        """
        new_kwargs: dict[str, Any] = {}
        for f in fields(self):
            if f.name in section_overrides:
                current = getattr(self, f.name)
                if hasattr(current, "__dataclass_fields__"):
                    new_kwargs[f.name] = replace(current, **section_overrides[f.name])
                else:
                    new_kwargs[f.name] = section_overrides[f.name]
        return replace(self, **new_kwargs)


def load_runtime_config(path: str | os.PathLike[str] | None = None) -> RuntimeConfig:
    """加载并解析 runtime.yaml。

    Parameters
    ----------
    path
        若为 ``None`` 使用包内默认路径（``src/dataset_generation/configs/runtime.yaml``）。
    """
    target = Path(path) if path is not None else _DEFAULT_RUNTIME_YAML
    with target.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return RuntimeConfig.from_dict(data, source_path=str(target))


@lru_cache(maxsize=4)
def _cached_load(path_str: str) -> RuntimeConfig:
    return load_runtime_config(path_str)


def get_runtime_config(path: str | os.PathLike[str] | None = None) -> RuntimeConfig:
    """带 LRU 缓存的加载入口（同一路径只读一次磁盘）。"""
    target = str(Path(path)) if path is not None else str(_DEFAULT_RUNTIME_YAML)
    return _cached_load(target)
