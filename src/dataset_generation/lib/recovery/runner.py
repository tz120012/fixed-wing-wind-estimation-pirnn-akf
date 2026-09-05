"""恢复 / 重试编排（recovery）。

把原 ``DatasetGenerator.{generate_missing_report, run_single_run_recovery,
run_single_segment_recovery, run_full_paper_automated}`` 从 ``generate_dataset.py``
迁出，作为独立的 *协议化* 入口，对 ``runner`` 仅依赖以下方法：
    - ``runner.run_multi_segment_session(wind_cfg, seg_cfgs, out_dir, base_id, run_number, log_fn)``
    - ``runner.execute_single_segment(seg_cfg, output_file)``
    - ``runner._prepare_sortie_environment(...)`` / ``runner._cleanup_after_sortie(...)``
    - ``runner.stop_px4_sitl()``
    - 字段：``runner.output_dir``、``runner.base_dir``、``runner.runtime``、``runner._px4_log_path``

这种"协议依赖"让 Phase 6 的进一步重构（彻底重写 DatasetGenerator）可以替换实现，
而无需触碰本文件。

公开函数：
    - :func:`generate_missing_report`
    - :func:`run_single_run_recovery`
    - :func:`run_single_segment_recovery`
    - :func:`run_full_paper_automated`
"""

from __future__ import annotations

import asyncio
import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from ..planner.schedule import SEGMENTS_PER_RUN, build_paper_run_schedule
from ..planner.segment_factory import generate_segment_configs_for_run
from ..planner.wind_factory import generate_wind_config
from ..validation.quality_gates import segment_output_is_valid


def _segment_filename(run_number: int, segment_number: int) -> str:
    return f"datasets-{run_number}-{segment_number}.json"


async def generate_missing_report(runner: Any, report_path: Path) -> None:
    """扫描全数据集，写一份"已采 / 缺失"报告到 ``report_path``。"""
    schedule = build_paper_run_schedule()
    total_runs = len(schedule)
    total_segments = total_runs * SEGMENTS_PER_RUN
    missing = []

    for run_index, (dataset_type, base_flight_id, _segment_types) in enumerate(schedule):
        output_subdir = runner.output_dir / dataset_type
        for i in range(SEGMENTS_PER_RUN):
            filename = _segment_filename(run_index + 1, i + 1)
            filepath = output_subdir / filename
            if not segment_output_is_valid(filepath, runner.runtime):
                missing.append({
                    "run": run_index + 1,
                    "segment": i + 1,
                    "dataset_type": dataset_type,
                    "filename": filename,
                    "flight_id": base_flight_id + i,
                })

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("缺失数据报告 (Missing Data Report)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"总计: {len(missing)}/{total_segments} 段缺失\n\n")
        if not missing:
            f.write(f"所有 {total_segments} 段数据已完整采集！\n")
        else:
            f.write("缺失明细:\n")
            f.write("-" * 60 + "\n")
            for item in missing:
                f.write(
                    f"Run {item['run']:2d} | Seg {item['segment']} | "
                    f"{item['dataset_type']:8s} | {item['filename']}\n"
                )
            f.write("\n" + "=" * 60 + "\n")
            f.write("补采命令示例 (補采单轮):\n")
            f.write("-" * 60 + "\n")
            runs_missing = sorted({item["run"] for item in missing})
            for run_num in runs_missing:
                f.write(f"python generate_dataset.py --mode recover --run {run_num}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("补采单段命令示例:\n")
            f.write("-" * 60 + "\n")
            for item in missing[:10]:
                f.write(
                    f"python generate_dataset.py --mode recover "
                    f"--run {item['run']} --segment {item['segment']}\n"
                )
            if len(missing) > 10:
                f.write(f"... (还有 {len(missing) - 10} 条)\n")

    print(f"[Recovery] 缺失数据报告已生成: {report_path}")
    print(f"[Recovery] 缺失: {len(missing)}/{total_segments} 段")


async def run_single_run_recovery(
    runner: Any, run_number: int, base_seed: int = 26, log_path: Optional[Path] = None
) -> None:
    """补采指定轮次的所有段。"""
    schedule = build_paper_run_schedule()
    total_runs = len(schedule)
    if not 1 <= run_number <= total_runs:
        raise ValueError(f"run_number 必须在1-{total_runs}之间，当前: {run_number}")

    run_index = run_number - 1
    dataset_type, base_flight_id, segment_types = schedule[run_index]
    print(f"[Recovery] 补采 Run {run_number}/{total_runs}: {dataset_type}")

    run_seed = base_seed + run_index
    wind_config = generate_wind_config(
        dataset_type=dataset_type, seed=run_seed, run_index=run_index
    )
    segment_configs = generate_segment_configs_for_run(
        dataset_type, base_flight_id, segment_types, seed=run_seed
    )

    output_subdir = runner.output_dir / dataset_type
    output_subdir.mkdir(parents=True, exist_ok=True)
    await runner.run_multi_segment_session(
        wind_config, segment_configs, output_subdir, base_flight_id, run_number=run_number
    )
    print(f"[Recovery] Run {run_number} 补采完成")


async def run_single_segment_recovery(
    runner: Any, run_number: int, segment_number: int, base_seed: int = 26
) -> None:
    """补采指定轮次的指定段（不重跑同轮其它段）。"""
    schedule = build_paper_run_schedule()
    total_runs = len(schedule)
    if not 1 <= run_number <= total_runs:
        raise ValueError(f"run_number 必须在1-{total_runs}之间")
    if not 1 <= segment_number <= SEGMENTS_PER_RUN:
        raise ValueError(f"segment_number 必须在1-{SEGMENTS_PER_RUN}之间")

    run_index = run_number - 1
    dataset_type, base_flight_id, segment_types = schedule[run_index]
    print(f"[Recovery] 补采 Run {run_number} Segment {segment_number}: {dataset_type}")

    run_seed = base_seed + run_index
    wind_config = generate_wind_config(
        dataset_type=dataset_type, seed=run_seed, run_index=run_index
    )
    segment_configs = generate_segment_configs_for_run(
        dataset_type, base_flight_id, segment_types, seed=run_seed
    )

    segment_index = segment_number - 1
    segment_config = segment_configs[segment_index]
    output_subdir = runner.output_dir / dataset_type
    output_subdir.mkdir(parents=True, exist_ok=True)

    segment_config["wind_north"] = wind_config["wind_north"]
    segment_config["wind_east"] = wind_config["wind_east"]
    segment_config["wind_down"] = wind_config.get("wind_down", 0.0)
    segment_config["sortie_index"] = 1
    segment_config["sortie_count"] = 1
    segment_config["segment_index_in_run"] = segment_number

    output_path = output_subdir / _segment_filename(run_number, segment_number)

    await runner._prepare_sortie_environment(wind_config, [segment_config])
    try:
        await runner.execute_single_segment(segment_config, str(output_path))
        print(f"[Recovery] 已保存: {output_path}")
    finally:
        await runner._cleanup_after_sortie(land_first=True)
    print(f"[Recovery] Run {run_number} Segment {segment_number} 补采完成")


async def run_full_paper_automated(
    runner: Any,
    log_path: Optional[Path] = None,
    round_timeout: Optional[int] = None,
    max_retries: int = 1,
    base_seed: int = 26,
    skip_existing: bool = True,
) -> None:
    """论文版全自动采集：每轮重启 SITL，多段采集，无人监管。

    ``round_timeout`` 默认从 ``runner.runtime.timeouts.round_total_s`` 读取。
    """
    schedule = build_paper_run_schedule()
    total_runs = len(schedule)
    total_segments = total_runs * SEGMENTS_PER_RUN
    if round_timeout is None:
        round_timeout = runner.runtime.timeouts.round_total_s

    log_path = Path(
        log_path or runner.base_dir.parent / "logs" / f"multi_segment_{total_runs}runs.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_msg(msg: str) -> None:
        line = f"[{datetime.datetime.now().isoformat()}] {msg}\n"
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

    log_msg(
        f"========== 无人值守 {total_runs} 轮/{total_segments} 段采集开始 "
        f"(seed={base_seed}, skip_existing={skip_existing}) =========="
    )

    skipped_count = 0
    for run_index, (dataset_type, base_flight_id, segment_types) in enumerate(schedule):
        output_subdir = runner.output_dir / dataset_type
        output_subdir.mkdir(parents=True, exist_ok=True)

        if skip_existing:
            all_exist = all(
                segment_output_is_valid(
                    output_subdir / _segment_filename(run_index + 1, i + 1), runner.runtime
                )
                for i in range(SEGMENTS_PER_RUN)
            )
            if all_exist:
                log_msg(f"Run {run_index + 1}/{total_runs} 数据与 metadata 已完整存在，跳过")
                skipped_count += 1
                continue

        run_seed = base_seed + run_index
        wind_config = generate_wind_config(
            dataset_type=dataset_type, seed=run_seed, run_index=run_index
        )
        segment_configs = generate_segment_configs_for_run(
            dataset_type, base_flight_id, segment_types, seed=run_seed
        )
        runner._px4_log_path = log_path.parent / "px4_last_run.log"

        log_msg(
            f"Run {run_index + 1}/{total_runs} {dataset_type} base_id={base_flight_id} "
            f"wind=({wind_config['wind_speed']:.1f}m/s, {wind_config['wind_direction']:.0f}°)"
        )
        try:
            for attempt in range(max_retries + 1):
                try:
                    await asyncio.wait_for(
                        runner.run_multi_segment_session(
                            wind_config,
                            segment_configs,
                            output_subdir,
                            base_flight_id,
                            run_number=run_index + 1,
                            log_fn=log_msg,
                        ),
                        timeout=round_timeout,
                    )
                    log_msg(f"  -> OK (attempt {attempt + 1})")
                    break
                except asyncio.TimeoutError:
                    log_msg(f"  -> TIMEOUT (attempt {attempt + 1})")
                except Exception as e:
                    log_msg(f"  -> FAILED: {e} (attempt {attempt + 1})")
                if attempt < max_retries:
                    await runner.stop_px4_sitl()
                    await asyncio.sleep(runner.runtime.sleeps.between_sorties_s)
            else:
                log_msg("  -> 放弃本轮，继续下一轮")
        finally:
            await runner.stop_px4_sitl()
            runner._px4_log_path = None
            await asyncio.sleep(runner.runtime.sleeps.between_segments_s)

    log_msg(f"========== {total_runs} 轮采集结束 (跳过 {skipped_count} 轮) ===========")
    await generate_missing_report(runner, log_path.parent / "missing_data_report.txt")
    log_msg("缺失数据报告已生成: logs/missing_data_report.txt")
