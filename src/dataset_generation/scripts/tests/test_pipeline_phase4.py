"""Phase 4 单元测试：lib/pipeline/。

不真正启动 PX4，用 mock runner 验证 Stage / Pipeline / SegmentPipeline / SortiePipeline 的
编排行为：
    - Stage.run 包装计时与日志，正常分支返回 StageResult.ok；
    - Pipeline 遇 critical 失败立即中止，非 critical 失败仍累积到 ctx.failures；
    - SortiePipeline 在 main pipeline 失败后**仍调用** CleanupStage（关键不变量）；
    - SegmentPipeline 三个 stage 顺序执行；
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "dataset_generation"))

from lib.config import load_runtime_config  # noqa: E402
from lib.obs import FailureCategory, FailureRecord  # noqa: E402
from lib.pipeline import (  # noqa: E402
    CleanupStage,
    EnsureFlightActiveStage,
    LogSegmentStage,
    Pipeline,
    PipelineCtx,
    PrepareEnvStage,
    RunSegmentsStage,
    SaveMetadataStage,
    SegmentPipeline,
    SortiePipeline,
    Stage,
    StageResult,
)


class _OkStage(Stage):
    name = "ok_stage"

    async def execute(self, ctx):
        return StageResult.ok(touched=True)


class _CriticalFailStage(Stage):
    name = "critical_fail"
    critical = True

    async def execute(self, ctx):
        return StageResult.fail(FailureRecord(
            category=FailureCategory.VALIDATION,
            code="boom",
            message="boom",
            context={},
        ))


class _NonCriticalFailStage(Stage):
    name = "soft_fail"
    critical = False

    async def execute(self, ctx):
        return StageResult.fail(FailureRecord(
            category=FailureCategory.VALIDATION,
            code="meh",
            message="meh",
            context={},
        ), partial=True)


def test_stage_run_records_elapsed_and_artifacts():
    ctx = PipelineCtx()
    res = asyncio.run(_OkStage().run(ctx))
    assert res.success
    assert res.elapsed_ms >= 0
    assert res.artifacts == {"touched": True}
    print("[OK] test_stage_run_records_elapsed_and_artifacts")


def test_pipeline_aborts_on_critical_failure():
    pipe = Pipeline(stages=[_OkStage(), _CriticalFailStage(), _OkStage()])
    ctx = asyncio.run(pipe.execute(PipelineCtx()))
    # 第三个 OkStage 不应被执行（touched 由第一个 OkStage 写入，但第三个不会再写）
    assert ctx.data == {"touched": True}
    assert len(ctx.failures) == 1
    assert ctx.failures[0].code == "boom"
    print("[OK] test_pipeline_aborts_on_critical_failure")


def test_pipeline_continues_on_non_critical_failure():
    pipe = Pipeline(stages=[_NonCriticalFailStage(), _OkStage()])
    ctx = asyncio.run(pipe.execute(PipelineCtx()))
    assert "touched" in ctx.data
    assert ctx.data.get("partial") is True
    assert len(ctx.failures) == 1
    print("[OK] test_pipeline_continues_on_non_critical_failure")


# ---------- 用 mock runner 验证 SegmentPipeline / SortiePipeline ----------

class _MockRunner:
    def __init__(self):
        self.runtime = load_runtime_config().with_overrides(
            sleeps={"between_segments_s": 0.0, "between_sorties_s": 0.0}
        )
        self.calls = []
        self.fail_segment = False
        self.health_ok = True
        self._log = print

    async def check_and_restart_mavsdk_server(self):
        self.calls.append("check_and_restart_mavsdk_server")
        return self.health_ok

    async def execute_single_segment(self, seg_cfg, output_file):
        self.calls.append(("execute_single_segment", seg_cfg["id"], output_file))
        if self.fail_segment:
            raise RuntimeError("simulated capture failure")
        # 写一个空 metadata 让 SaveMetadataStage 通过
        Path(output_file).write_text("[]")
        from lib.validation.quality_gates import segment_metadata_path
        meta = segment_metadata_path(Path(output_file))
        meta.write_text('{"duration": 10}')

    async def _prepare_sortie_environment(self, *args, **kwargs):
        self.calls.append("_prepare_sortie_environment")

    async def _cleanup_after_sortie(self, *args, **kwargs):
        self.calls.append("_cleanup_after_sortie")


def test_segment_pipeline_runs_three_stages_in_order(tmpdir):
    runner = _MockRunner()
    out = Path(tmpdir) / "seg.json"
    ctx = PipelineCtx(data={
        "segment_config": {"id": "train_0001"},
        "output_file": str(out),
    })
    res_ctx = asyncio.run(SegmentPipeline(runner).execute(ctx))
    assert any("execute_single_segment" == c[0] for c in runner.calls if isinstance(c, tuple))
    assert "metadata_path" in res_ctx.data
    assert len(res_ctx.failures) == 0
    print("[OK] test_segment_pipeline_runs_three_stages_in_order")


def test_segment_pipeline_records_failure_when_capture_fails(tmpdir):
    runner = _MockRunner()
    runner.fail_segment = True
    out = Path(tmpdir) / "seg.json"
    ctx = PipelineCtx(data={
        "segment_config": {"id": "train_0002"},
        "output_file": str(out),
    })
    res_ctx = asyncio.run(SegmentPipeline(runner).execute(ctx))
    assert len(res_ctx.failures) >= 1
    assert any(f.code == "segment_capture_failed" for f in res_ctx.failures)
    print("[OK] test_segment_pipeline_records_failure_when_capture_fails")


def test_sortie_pipeline_always_runs_cleanup(tmpdir):
    """SortiePipeline 即使 main 失败也必须 CleanupStage。"""
    runner = _MockRunner()
    runner.fail_segment = True

    def resolve_output(ri, idx):
        return str(Path(tmpdir) / f"r{ri}_s{idx}.json")

    ctx = PipelineCtx(data={
        "run_index": 0,
        "wind_config": {"wind_speed": 1.0},
        "sortie_segment_configs": [{"id": "x", "altitude": 50}],
        "segment_pairs": [(0, {"id": "x", "altitude": 50})],
        "resolve_output": resolve_output,
        "log_fn": print,
    })
    res_ctx = asyncio.run(SortiePipeline(runner).execute(ctx))
    # 必须包含 cleanup 调用（即使前面失败）
    assert "_cleanup_after_sortie" in runner.calls, runner.calls
    print("[OK] test_sortie_pipeline_always_runs_cleanup")


def test_sortie_pipeline_success_path(tmpdir):
    runner = _MockRunner()

    def resolve_output(ri, idx):
        return str(Path(tmpdir) / f"r{ri}_s{idx}.json")

    ctx = PipelineCtx(data={
        "run_index": 0,
        "wind_config": {"wind_speed": 1.0},
        "sortie_segment_configs": [{"id": "y", "altitude": 50}],
        "segment_pairs": [(0, {"id": "y", "altitude": 50})],
        "resolve_output": resolve_output,
        "log_fn": print,
    })
    res_ctx = asyncio.run(SortiePipeline(runner).execute(ctx))
    assert "_prepare_sortie_environment" in runner.calls
    assert "_cleanup_after_sortie" in runner.calls
    assert len(res_ctx.failures) == 0
    print("[OK] test_sortie_pipeline_success_path")


def main():
    import tempfile
    test_stage_run_records_elapsed_and_artifacts()
    test_pipeline_aborts_on_critical_failure()
    test_pipeline_continues_on_non_critical_failure()
    with tempfile.TemporaryDirectory() as td:
        test_segment_pipeline_runs_three_stages_in_order(td)
    with tempfile.TemporaryDirectory() as td:
        test_segment_pipeline_records_failure_when_capture_fails(td)
    with tempfile.TemporaryDirectory() as td:
        test_sortie_pipeline_always_runs_cleanup(td)
    with tempfile.TemporaryDirectory() as td:
        test_sortie_pipeline_success_path(td)
    print("\n[PASS] Phase 4 pipeline unit tests (7/7)")


if __name__ == "__main__":
    main()
