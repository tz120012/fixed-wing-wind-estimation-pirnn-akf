"""段级失败的结构化记录。

``FailureRecord`` 用于：
    - 替换 ``try/except: pass`` 中静默吞下的异常（变成 logger.warning(record)）；
    - 让段级校验失败可分类、可统计（不再是字符串拼接的 RuntimeError）；
    - 把"失败原因"持久化到 ``logs/segment_failures.jsonl``，事后可分析全数据集失败模式。

一个典型流程：
    >>> rec = FailureRecord(category=FailureCategory.PROCESS,
    ...                     code="px4_terminate_timeout",
    ...                     message="PX4 SIGTERM 5s 内未退出",
    ...                     context={"pid": 1234})
    >>> logger.failure(rec)
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .. import errors as _errors


class FailureCategory(str, enum.Enum):
    """高层故障域。与 ``errors.py`` 异常基类一一对应（便于按类聚合）。"""

    PROCESS = "process"             # PX4 / mavsdk_server / JSBSim 进程故障
    FLIGHT = "flight"               # 飞控（GPS / arm / takeoff / offboard）
    TELEMETRY = "telemetry"         # 数据采集订阅故障
    VALIDATION = "validation"       # 段级质量门拒绝
    CLEANUP = "cleanup"             # 兜底清理（不影响主路径，但应记录）
    CONFIG = "config"               # 配置/环境问题
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureRecord:
    """单次失败事件的结构化描述。

    Attributes
    ----------
    category
        高层故障域，用于聚合统计。
    code
        机器可读的故障代码（蛇形小写），同一 category 下唯一标识子原因，
        例：``"coverage_too_low"``、``"sitl_terminate_timeout"``。
    message
        人读说明，可包含具体数值。
    context
        额外上下文（pid、超时秒数、覆盖率等）。会原样写入 JSONL。
    retriable
        软提示：调用方是否应当重试。校验失败常为 True，崩溃失败常为 False。
    timestamp
        失败发生时的 wall-clock 秒。默认填当前时间。
    """

    category: FailureCategory
    code: str
    message: str
    context: Mapping[str, Any] = field(default_factory=dict)
    retriable: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.timestamp,
            "category": self.category.value,
            "code": self.code,
            "message": self.message,
            "retriable": self.retriable,
            "context": dict(self.context),
        }


_EXCEPTION_TO_CATEGORY: dict[type[BaseException], FailureCategory] = {
    _errors.ProcessError: FailureCategory.PROCESS,
    _errors.FlightControllerError: FailureCategory.FLIGHT,
    _errors.TelemetryError: FailureCategory.TELEMETRY,
    _errors.ValidationError: FailureCategory.VALIDATION,
}


def classify_exception(exc: BaseException) -> FailureCategory:
    """根据异常类型反查 ``FailureCategory``。"""
    for cls, cat in _EXCEPTION_TO_CATEGORY.items():
        if isinstance(exc, cls):
            return cat
    return FailureCategory.UNKNOWN


def from_exception(
    exc: BaseException,
    *,
    code: Optional[str] = None,
    extra_context: Optional[Mapping[str, Any]] = None,
) -> FailureRecord:
    """把任意异常折叠为 ``FailureRecord``。

    ``code`` 默认取异常类名的 snake_case，例如 ``GpsTimeout`` -> ``gps_timeout``。
    """
    category = classify_exception(exc)
    if code is None:
        code = _camel_to_snake(type(exc).__name__)
    context: dict[str, Any] = {}
    if isinstance(exc, _errors.CollectionError):
        context.update(exc.context)
    if extra_context:
        context.update(extra_context)
    retriable = getattr(exc, "retriable", category != FailureCategory.UNKNOWN)
    return FailureRecord(
        category=category,
        code=code,
        message=str(exc) or type(exc).__name__,
        context=context,
        retriable=retriable,
    )


def _camel_to_snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)
