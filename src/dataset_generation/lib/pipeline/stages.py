"""Pipeline / Stage 抽象。

每个 Stage 是一个原子步骤（写风场、起飞、采样、校验等），返回 :class:`StageResult`：
    - ``success=True`` 时把 ``artifacts`` 合并进上下文供后续 Stage 使用；
    - ``success=False`` 时携带 :class:`FailureRecord`，由 Pipeline 决定是否中断。

Pipeline 不直接负责"重试"——重试逻辑在更上层的 ``recovery.runner`` 中实现，
本层只关心一次执行的成败与可观测性。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..obs import FailureRecord, get_logger


_LOGGER = get_logger("pipeline")


@dataclass
class PipelineCtx:
    """Pipeline 在各 Stage 间传递的可变上下文。

    Stage 可读 ``data``，并通过返回 ``StageResult.artifacts`` 把新键写入。
    ``failures`` 累积所有 Stage 上报的失败（即使 Pipeline 决定继续）。
    """

    data: Dict[str, Any] = field(default_factory=dict)
    failures: List[FailureRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def update(self, **kwargs: Any) -> None:
        self.data.update(kwargs)


@dataclass
class StageResult:
    """单个 Stage 的执行结果。

    Attributes
    ----------
    success
        Stage 是否成功。
    failure
        Stage 失败时的结构化记录；成功时为 ``None``。
    artifacts
        Stage 产出的命名工件，会被合并到 ``PipelineCtx.data``。
    elapsed_ms
        Stage 耗时（毫秒）。
    """

    success: bool
    failure: Optional[FailureRecord] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @classmethod
    def ok(cls, **artifacts: Any) -> "StageResult":
        return cls(success=True, artifacts=artifacts)

    @classmethod
    def fail(cls, failure: FailureRecord, **artifacts: Any) -> "StageResult":
        return cls(success=False, failure=failure, artifacts=artifacts)


class Stage:
    """Pipeline 阶段基类。

    子类至少要实现 ``async def execute(self, ctx) -> StageResult``。
    可选覆盖 ``critical = True`` 表示失败时必须中止 pipeline。
    """

    name: str = "stage"
    critical: bool = False

    async def execute(self, ctx: PipelineCtx) -> StageResult:
        raise NotImplementedError

    async def run(self, ctx: PipelineCtx) -> StageResult:
        """带计时与结构化日志的标准入口（不要在子类中覆盖此方法）。"""
        start = time.monotonic()
        _LOGGER.info(f"stage start: {self.name}", stage=self.name)
        try:
            result = await self.execute(ctx)
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            from ..obs.failure import from_exception
            failure = from_exception(exc)
            _LOGGER.error(
                f"stage exception: {self.name}",
                stage=self.name, error=str(exc), elapsed_ms=round(elapsed_ms, 1),
            )
            return StageResult(success=False, failure=failure, elapsed_ms=elapsed_ms)

        elapsed_ms = (time.monotonic() - start) * 1000
        if not isinstance(result, StageResult):
            raise TypeError(
                f"Stage {self.name}.execute 必须返回 StageResult, 实际 {type(result)}"
            )
        result.elapsed_ms = elapsed_ms

        if result.success:
            _LOGGER.info(
                f"stage ok: {self.name}",
                stage=self.name, elapsed_ms=round(elapsed_ms, 1),
                artifacts=list(result.artifacts.keys()),
            )
        else:
            _LOGGER.warning(
                f"stage failed: {self.name}",
                stage=self.name, elapsed_ms=round(elapsed_ms, 1),
                code=result.failure.code if result.failure else None,
                detail=result.failure.message if result.failure else None,
            )
        return result


class Pipeline:
    """按顺序执行一组 Stage 的容器。

    遇到 ``critical=True`` 的 Stage 失败时立即中止；非关键失败仍累积到 ctx。
    """

    def __init__(self, stages: Sequence[Stage], name: str = "pipeline") -> None:
        self.stages = list(stages)
        self.name = name

    async def execute(self, ctx: PipelineCtx) -> PipelineCtx:
        for stage in self.stages:
            result = await stage.run(ctx)
            if result.failure is not None:
                ctx.failures.append(result.failure)
                if stage.critical and not result.success:
                    _LOGGER.error(
                        f"pipeline aborted at critical stage: {stage.name}",
                        pipeline=self.name, stage=stage.name,
                    )
                    return ctx
            if result.success:
                ctx.data.update(result.artifacts)
            elif not stage.critical:
                # 非关键 Stage 失败时，artifacts 仍可能含有部分产出
                ctx.data.update(result.artifacts)
        return ctx
