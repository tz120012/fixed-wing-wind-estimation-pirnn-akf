"""段级质量门：检查采样记录是否达到可入库标准。

迁移自 ``generate_dataset.py:511-603, 1214-1219``。

设计要点：
    1. 阈值（``min_valid_log_hz``、``min_coverage_ratio`` 等）从 ``runtime.yaml`` 读取，
       不再硬编码；
    2. 不再由 gate 函数 ``raise RuntimeError``，而是返回 :class:`GateResult`，由调用方
       决定是否抛错或重试；
    3. 每条违规以 :class:`FailureRecord` 形式产出，写到 ``segment_failures.jsonl`` 后
       可统计全数据集失败模式。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..config.runtime import RuntimeConfig
from ..obs import FailureCategory, FailureRecord


REQUIRED_LOG_FIELDS: Tuple[str, ...] = (
    "timestamp",
    "airspeed_m_s",
    "wind_north",
    "wind_east",
    "wind_down",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
)


def is_finite_number(value: Any) -> bool:
    """value 是否是有限数（不含 NaN / inf）。"""
    return isinstance(value, (int, float)) and math.isfinite(value)


@dataclass(frozen=True)
class GateResult:
    """段校验结果。

    Attributes
    ----------
    passed
        是否通过所有校验。
    failures
        失败列表。``passed=False`` 时至少包含一条；``passed=True`` 时为空。
    issues
        人读字符串列表（向后兼容旧 API；与 ``failures`` 一一对应）。
    """

    passed: bool
    failures: List[FailureRecord] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failures": [f.to_dict() for f in self.failures],
            "issues": list(self.issues),
        }


def validate_segment_records(
    records: Sequence[Mapping[str, Any]],
    requested_duration: float,
    runtime: RuntimeConfig,
) -> GateResult:
    """校验采集到的记录序列是否达标。

    迁移自 ``_validate_logged_records``。判定项：
        - 样本数 >= ``max(min_samples_floor, duration*min_valid_log_hz)``
        - 时间戳头尾有效 → 覆盖时长 >= ``max(min_coverage_floor_s, duration*min_coverage_ratio)``
        - 抽样前 10 条记录的字段完整性与有限性
    """
    qg = runtime.quality_gates
    failures: List[FailureRecord] = []
    issues: List[str] = []

    sample_count = len(records)
    min_samples = max(qg.min_samples_floor, int(float(requested_duration) * qg.min_valid_log_hz))
    if sample_count < min_samples:
        msg = f"样本数不足({sample_count} < {min_samples})"
        issues.append(msg)
        failures.append(FailureRecord(
            category=FailureCategory.VALIDATION,
            code="low_sample_rate",
            message=msg,
            context={
                "sample_count": sample_count,
                "min_samples": min_samples,
                "duration_s": requested_duration,
                "min_valid_log_hz": qg.min_valid_log_hz,
            },
            retriable=True,
        ))

    if not records:
        return GateResult(passed=False, failures=failures, issues=issues)

    first_ts = records[0].get("timestamp")
    last_ts = records[-1].get("timestamp")
    if not (is_finite_number(first_ts) and is_finite_number(last_ts)):
        msg = "时间戳无效"
        issues.append(msg)
        failures.append(FailureRecord(
            category=FailureCategory.VALIDATION,
            code="invalid_timestamp",
            message=msg,
            context={"first_ts": first_ts, "last_ts": last_ts},
        ))
    else:
        coverage = float(last_ts) - float(first_ts)
        min_coverage = max(qg.min_coverage_floor_s, float(requested_duration) * qg.min_coverage_ratio)
        if coverage < min_coverage:
            msg = f"有效覆盖不足({coverage:.2f}s < {min_coverage:.2f}s)"
            issues.append(msg)
            failures.append(FailureRecord(
                category=FailureCategory.VALIDATION,
                code="coverage_too_low",
                message=msg,
                context={
                    "coverage_s": round(coverage, 4),
                    "min_coverage_s": round(min_coverage, 4),
                    "duration_s": requested_duration,
                    "min_coverage_ratio": qg.min_coverage_ratio,
                },
                retriable=True,
            ))

    for idx, entry in enumerate(records[: min(10, sample_count)]):
        if not isinstance(entry, Mapping):
            msg = f"第 {idx + 1} 条记录不是对象"
            issues.append(msg)
            failures.append(FailureRecord(
                category=FailureCategory.VALIDATION,
                code="record_not_dict",
                message=msg,
                context={"index": idx},
            ))
            continue
        missing = [f for f in REQUIRED_LOG_FIELDS if f not in entry]
        if missing:
            msg = f"第 {idx + 1} 条缺字段: {', '.join(missing)}"
            issues.append(msg)
            failures.append(FailureRecord(
                category=FailureCategory.VALIDATION,
                code="missing_fields",
                message=msg,
                context={"index": idx, "missing": missing},
            ))
            continue
        invalid = [f for f in REQUIRED_LOG_FIELDS if not is_finite_number(entry[f])]
        if invalid:
            msg = f"第 {idx + 1} 条字段非有限数: {', '.join(invalid)}"
            issues.append(msg)
            failures.append(FailureRecord(
                category=FailureCategory.VALIDATION,
                code="non_finite_field",
                message=msg,
                context={"index": idx, "fields": invalid},
            ))

    return GateResult(passed=len(failures) == 0, failures=failures, issues=issues)


def validate_airspeed_filled_ratio(
    filled_ratio: float, runtime: RuntimeConfig
) -> Optional[FailureRecord]:
    """额外质量门：airspeed 前向填充比例不得超过 ``airspeed_filled_ratio_threshold``。

    返回 ``None`` 表示通过；否则返回 ``FailureRecord``。
    """
    threshold = runtime.quality_gates.airspeed_filled_ratio_threshold
    if filled_ratio <= threshold:
        return None
    return FailureRecord(
        category=FailureCategory.VALIDATION,
        code="airspeed_degraded",
        message=(
            f"airspeed 前向填充比例过高 ({filled_ratio:.1%} > {threshold:.1%})，"
            f"PX4 airspeed_selector 异常，段不可用"
        ),
        context={"filled_ratio": filled_ratio, "threshold": threshold},
        retriable=True,
    )


def segment_metadata_path(data_path: Path) -> Path:
    """返回段数据文件对应的 ``*_metadata.json`` 路径。"""
    data_path = Path(data_path)
    return data_path.with_name(f"{data_path.stem}_metadata.json")


def segment_output_is_valid(data_path: Path, runtime: RuntimeConfig) -> bool:
    """检查段数据 + metadata 同时存在且通过校验。

    与原 ``_segment_output_is_valid`` 同语义；用于扫描已有数据时跳过完好段。
    """
    data_path = Path(data_path)
    metadata_path = segment_metadata_path(data_path)
    if not data_path.exists() or not metadata_path.exists():
        return False
    try:
        records = json.loads(data_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(records, list) or not isinstance(metadata, dict):
        return False
    requested_duration = metadata.get("duration")
    if not is_finite_number(requested_duration):
        return False
    return validate_segment_records(records, requested_duration, runtime).passed


def cleanup_segment_outputs(output_file: Path) -> None:
    """删除段数据 + metadata（用于校验失败后的清理）。"""
    data_path = Path(output_file)
    metadata_path = segment_metadata_path(data_path)
    data_path.unlink(missing_ok=True)
    metadata_path.unlink(missing_ok=True)
