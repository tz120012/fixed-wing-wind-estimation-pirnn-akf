"""Sortie（一次起飞-降落）级 Pipeline。

每个 sortie 对应一组逻辑段。Stage 划分：
    PrepareEnvStage    # stop_px4_sitl + 写风场 + start_px4_sitl
    TakeoffStage       # initialize_controllers + arm_and_takeoff
    RunSegmentsStage   # 串行跑 SegmentPipeline 跑完 sortie 内全部段
    LandStage          # 降落
    CleanupStage       # stop_px4_sitl + disconnect

与 SegmentPipeline 同样采用"委托式"实现：所有 Stage 调用 ``runner`` 现有方法。
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Tuple

from ..obs import FailureCategory, FailureRecord
from .stages import Pipeline, PipelineCtx, Stage, StageResult


class PrepareEnvStage(Stage):
    """停旧 SITL → 写风场 → 启 SITL → 验证 bridge。"""

    name = "prepare_env"
    critical = True

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    async def execute(self, ctx: PipelineCtx) -> StageResult:
        wind_cfg = ctx.data["wind_config"]
        seg_cfgs = ctx.data["sortie_segment_configs"]
        log_fn = ctx.data.get("log_fn")
        # 委托给现有方法（已在 Phase 2 内部委托给 Px4SitlProcess）
        await self.runner._prepare_sortie_environment(wind_cfg, seg_cfgs, log_fn=log_fn)
        return StageResult.ok()


class RunSegmentsStage(Stage):
    """串行跑 sortie 内每个段。"""

    name = "run_segments"
    critical = False

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    async def execute(self, ctx: PipelineCtx) -> StageResult:
        seg_pairs: List[Tuple[int, dict]] = ctx.data["segment_pairs"]
        run_index: int = ctx.data["run_index"]
        log_fn = ctx.data.get("log_fn", print)
        outputs: List[str] = []
        for original_idx, seg_cfg in seg_pairs:
            output_file = ctx.data["resolve_output"](run_index, original_idx)
            try:
                await self.runner.execute_single_segment(seg_cfg, output_file)
                outputs.append(output_file)
                log_fn(f"[Pipeline] 逻辑段 {original_idx + 1}/5 已保存 {output_file}")
                await asyncio.sleep(self.runner.runtime.sleeps.between_segments_s)
            except Exception as e:
                # 把段级失败累计但不立即中止（让 sortie 继续做 cleanup）
                return StageResult.fail(FailureRecord(
                    category=FailureCategory.TELEMETRY,
                    code="sortie_segment_failed",
                    message=str(e),
                    context={"run_index": run_index, "segment_idx": original_idx},
                ), outputs=outputs)
        return StageResult.ok(outputs=outputs)


class CleanupStage(Stage):
    """sortie 收尾：land + stop_px4_sitl + disconnect。"""

    name = "cleanup"

    def __init__(self, runner: Any, land_first: bool = True) -> None:
        self.runner = runner
        self.land_first = land_first

    async def execute(self, ctx: PipelineCtx) -> StageResult:
        log_fn = ctx.data.get("log_fn")
        await self.runner._cleanup_after_sortie(log_fn=log_fn, land_first=self.land_first)
        return StageResult.ok()


class SortiePipeline:
    """单次 sortie 编排。

    Pipeline 在 ``execute`` 末尾**始终**运行 ``CleanupStage``，即使中间 Stage 失败。
    这是与朴素 ``Pipeline.execute`` 行为的关键差别。
    """

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self._main = Pipeline(
            stages=[PrepareEnvStage(runner), RunSegmentsStage(runner)],
            name="sortie.main",
        )
        self._cleanup = CleanupStage(runner, land_first=True)

    async def execute(self, ctx: PipelineCtx) -> PipelineCtx:
        try:
            await self._main.execute(ctx)
        finally:
            cleanup_result = await self._cleanup.run(ctx)
            if not cleanup_result.success and cleanup_result.failure is not None:
                ctx.failures.append(cleanup_result.failure)
        return ctx
