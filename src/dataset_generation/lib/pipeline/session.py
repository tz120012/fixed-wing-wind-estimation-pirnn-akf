"""Session（多 sortie 单轮）级 Pipeline。

对应一轮（论文版每轮 5 段，可能拆成 1-3 个 sortie）的串行执行：
    1. 用 ``planner.split_segment_configs_for_sorties`` 拆段为 sortie 组；
    2. 每个 sortie 单独跑 :class:`SortiePipeline`；
    3. sortie 间按 ``runtime.sleeps.between_sorties_s`` 间隔。

注意：本 Pipeline 不负责 ``run_multi_segment_session`` 中的"逐段重试 + 重启 sortie"逻辑，
那部分留在 ``DatasetGenerator`` 中（Phase 5 会迁入 ``recovery.runner``）。当前 Phase 4
仅提供供 Phase 5 / Phase 6 重构使用的骨架。
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, List, Tuple

from ..planner.segment_factory import split_segment_configs_for_sorties
from .stages import PipelineCtx
from .sortie import SortiePipeline


class SessionPipeline:
    """跑一轮（一组段配置）的 Session 编排。"""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self.sortie = SortiePipeline(runner)

    async def execute(
        self,
        run_index: int,
        wind_config: dict,
        segment_configs: List[dict],
        resolve_output,
    ) -> List[PipelineCtx]:
        """串行执行该轮的所有 sortie，返回每个 sortie 的 ctx。"""
        groups: List[List[Tuple[int, dict]]] = split_segment_configs_for_sorties(segment_configs)
        results: List[PipelineCtx] = []
        for sortie_idx, seg_pairs in enumerate(groups, start=1):
            sortie_seg_cfgs = [seg for _, seg in seg_pairs]
            ctx = PipelineCtx(data={
                "run_index": run_index,
                "wind_config": wind_config,
                "sortie_segment_configs": sortie_seg_cfgs,
                "segment_pairs": seg_pairs,
                "resolve_output": resolve_output,
                "log_fn": getattr(self.runner, "_log", None) or print,
            })
            await self.sortie.execute(ctx)
            results.append(ctx)
            # sortie 间隔
            interval = (
                self.runner.runtime.sleeps.between_segments_s
                if sortie_idx < len(groups)
                else self.runner.runtime.sleeps.between_sorties_s
            )
            await asyncio.sleep(interval)
        return results
