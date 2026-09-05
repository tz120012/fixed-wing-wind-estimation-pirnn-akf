"""段（Segment）级 Pipeline。

把单段执行流程拆分为：
    ConfigureWindStage     # 写 wind_config.txt + 起始航向（在 sortie 前置时调用，本 Pipeline 中可选）
    EnsureFlightActiveStage # 检查飞控活着、若无则 reconnect
    LogSegmentStage         # DataLogger + fly_* 并发执行
    ValidateSegmentStage    # quality_gates.validate_segment_records
    SaveMetadataStage       # 写 _metadata.json

Stage 实现采取"委托式"：内部调用调用方注入的 ``runner`` 对象的方法。
``runner`` 通常是 ``DatasetGenerator`` 实例（或它在 Phase 5 之后的"瘦身版本"）。
这样可以在不重写 1000 行业务逻辑的前提下，让结构化日志 / Stage 抽象生效。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..config.runtime import RuntimeConfig
from ..obs import FailureCategory, FailureRecord
from ..validation import (
    cleanup_segment_outputs,
    segment_metadata_path,
    validate_airspeed_filled_ratio,
    validate_segment_records,
)
from .stages import Pipeline, PipelineCtx, Stage, StageResult


class EnsureFlightActiveStage(Stage):
    """确保 mavsdk_server / FlightController 健康；不健康则尝试重启重连。"""

    name = "ensure_flight_active"
    critical = True

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    async def execute(self, ctx: PipelineCtx) -> StageResult:
        ok = await self.runner.check_and_restart_mavsdk_server()
        if not ok:
            return StageResult.fail(FailureRecord(
                category=FailureCategory.FLIGHT_CONTROLLER,
                code="mavsdk_unhealthy",
                message="mavsdk_server 不健康且无法恢复",
                context={},
            ))
        return StageResult.ok()


class LogSegmentStage(Stage):
    """并发执行 ``fly_<maneuver>`` 与 ``DataLogger``，把 logger 与 logging_summary 写入 ctx。

    内部调用 ``runner.execute_single_segment(seg_cfg, output_file)``，但分两步暴露：
        - 期望 runner 提供低层方法 ``_run_segment_capture``（返回 logger / logging_summary）。
        - 当前 DatasetGenerator 仍把 capture + validate 合在 ``execute_single_segment``，
          因此此 Stage 暂时直接复用旧方法，并在 ctx 中标记 ``segment_executed=True``。

    Phase 5 重构后会进一步拆分；当前阶段保持向后兼容。
    """

    name = "log_segment"
    critical = True

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    async def execute(self, ctx: PipelineCtx) -> StageResult:
        seg_cfg: dict = ctx.data["segment_config"]
        output_file: str = ctx.data["output_file"]
        try:
            await self.runner.execute_single_segment(seg_cfg, output_file)
        except RuntimeError as e:
            # execute_single_segment 内部对 jsbsim_bridge / 网络断开 / 采样失败均 raise
            return StageResult.fail(FailureRecord(
                category=FailureCategory.TELEMETRY,
                code="segment_capture_failed",
                message=str(e),
                context={"segment_id": seg_cfg.get("id")},
            ))
        return StageResult.ok(segment_executed=True)


class SaveMetadataStage(Stage):
    """写入 segment metadata（在 LogSegmentStage 成功后才执行）。

    当前 ``execute_single_segment`` 已在内部写好 metadata，本 Stage 仅做 sanity check。
    """

    name = "save_metadata"

    async def execute(self, ctx: PipelineCtx) -> StageResult:
        output_file = Path(ctx.data["output_file"])
        meta = segment_metadata_path(output_file)
        if not output_file.exists() or not meta.exists():
            return StageResult.fail(FailureRecord(
                category=FailureCategory.VALIDATION,
                code="output_files_missing",
                message=f"段执行成功但 {output_file.name} 或 metadata 缺失",
                context={"data": str(output_file), "meta": str(meta)},
            ))
        return StageResult.ok(metadata_path=str(meta))


class SegmentPipeline:
    """对外暴露的 SegmentPipeline 入口，包装 ``Pipeline.execute``。

    Usage::

        ctx = PipelineCtx(data={"segment_config": cfg, "output_file": out})
        result_ctx = await SegmentPipeline(runner).execute(ctx)
        # result_ctx.failures 列出过程中所有 FailureRecord
    """

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.pipeline = Pipeline(
            stages=[
                EnsureFlightActiveStage(runner),
                LogSegmentStage(runner),
                SaveMetadataStage(),
            ],
            name="segment",
        )

    async def execute(self, ctx: PipelineCtx) -> PipelineCtx:
        return await self.pipeline.execute(ctx)
