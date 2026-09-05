"""数据采集模块的领域异常层级。

设计原则：
  - 所有可预期异常都继承 ``CollectionError``，便于上层统一捕获。
  - 异常按"故障域"分组（进程/飞控/采集/校验），每类对应一种重试策略。
  - 异常携带可选 ``context`` 字典，写入结构化日志后便于事后分析。
  - 不要直接 raise ``RuntimeError`` 或 bare ``Exception``，那会让上层无法分类。

使用样例：
    raise SitlStartFailed("PX4 未在 30s 内监听 4560",
                          context={"timeout_s": 30, "pid": 12345})
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


class CollectionError(Exception):
    """数据采集相关异常的根。

    所有自定义异常都应继承本类。``context`` 字段会在结构化日志中以 JSON 形式保留，
    供事后定位用。``retriable`` 用作软提示，最终是否重试仍由调用方决定。
    """

    retriable: bool = False

    def __init__(
        self,
        message: str = "",
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context) if context else {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": str(self),
            "retriable": self.retriable,
            "context": self.context,
        }


class ProcessError(CollectionError):
    """进程 / 资源类故障的基类（PX4、mavsdk_server、JSBSim）。"""

    retriable = True


class SitlStartFailed(ProcessError):
    """PX4 SITL 启动后未通过就绪检查（bridge 未监听 / mavsdk 未就绪等）。"""


class SitlCrashed(ProcessError):
    """PX4 SITL 进程在采集过程中异常退出。

    调用方应当假设 mavsdk_server / JSBSim 状态已损坏，需要做完整重启。
    """


class MavsdkServerDead(ProcessError):
    """mavsdk_server 子进程崩溃或 gRPC 端口无响应。"""


class JsbsimBridgeMissing(ProcessError):
    """JSBSim bridge (端口 4560) 未启动或不可达。"""


class FlightControllerError(CollectionError):
    """飞控（MAVSDK 端）相关故障的基类。"""

    retriable = True


class GpsTimeout(FlightControllerError):
    """GPS / EKF 在限定时间内未达到可起飞状态。"""


class ArmTimeout(FlightControllerError):
    """达到 arm 重试上限仍无法解锁。"""


class TakeoffFailed(FlightControllerError):
    """起飞过程中高度未达预期或检测到下俯。"""


class OffboardLost(FlightControllerError):
    """Offboard 模式被外力取消（manual override / failsafe）。"""


class TelemetryError(CollectionError):
    """数据采集 / 遥测层故障。"""

    retriable = True


class TelemetryStarvation(TelemetryError):
    """遥测速率显著低于预期，疑似采集回路饥饿。"""


class WindTruthMissing(TelemetryError):
    """jsbsim_bridge 风真值 CSV 不存在或无法读取。"""


class PymavlinkSilent(TelemetryError):
    """pymavlink 监听端未在限定窗口收到任何 MAVLink 消息。"""


class ValidationError(CollectionError):
    """段级质量门拒绝采样结果。"""

    retriable = True


class CoverageTooLow(ValidationError):
    """有效采样覆盖时长不足请求时长 ``min_coverage_ratio``。"""


class AirspeedDegraded(ValidationError):
    """airspeed 前向填充比例超过阈值，PX4 airspeed_selector 已退化。"""


class MissingFields(ValidationError):
    """必填字段缺失（REQUIRED_LOG_FIELDS）。"""


class LowSampleRate(ValidationError):
    """采集频率低于 ``min_valid_log_hz``。"""


__all__ = [
    "CollectionError",
    "ProcessError",
    "SitlStartFailed",
    "SitlCrashed",
    "MavsdkServerDead",
    "JsbsimBridgeMissing",
    "FlightControllerError",
    "GpsTimeout",
    "ArmTimeout",
    "TakeoffFailed",
    "OffboardLost",
    "TelemetryError",
    "TelemetryStarvation",
    "WindTruthMissing",
    "PymavlinkSilent",
    "ValidationError",
    "CoverageTooLow",
    "AirspeedDegraded",
    "MissingFields",
    "LowSampleRate",
]
