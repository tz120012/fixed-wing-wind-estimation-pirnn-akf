"""命令行入口。

把原 ``generate_dataset.main()`` 的 70 行 dispatcher 抽到这里，统一管理参数与
mode 分发。``scripts/generate_dataset.py`` 仅保留极薄入口。
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Callable, Optional


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "PX4-SITL + JSBSim 数据集采集（论文版 160 轮 + 断点续传 + 补采）"
        )
    )
    p.add_argument(
        "--mode",
        default="multi_segment_160",
        choices=[
            "multi_segment_160",
            "multi_segment_80",
            "single_round",
            "recover",
            "report",
        ],
        help=(
            "multi_segment_160: 全自动论文版160轮; "
            "multi_segment_80: 兼容旧命令但同样执行160轮; "
            "single_round: 测试单轮; recover: 补采; report: 生成缺失报告"
        ),
    )
    p.add_argument("--output-dir", type=str, default=None, help="数据输出根目录")
    p.add_argument("--px4-dir", type=str, default=None, help="PX4-Autopilot 根目录")
    p.add_argument(
        "--airframe",
        choices=["rascal", "malolo"],
        default="rascal",
        help=(
            "JSBSim fixed-wing model. Use malolo for the frozen zero-shot "
            "cross-airframe revision dataset."
        ),
    )
    p.add_argument("--log", type=str, default=None, help="运行日志路径")
    p.add_argument("--round-timeout", type=int, default=None, help="单轮超时(秒)；默认从 runtime.yaml 读取")
    p.add_argument("--seed", type=int, default=26, help="随机种子（默认26）")
    p.add_argument(
        "--dataset-type",
        choices=["train", "val", "test_id", "test_ood"],
        default="train",
        help=(
            "single_round 条件；跨机型修订实验分别使用 test_id 和 test_ood"
        ),
    )
    p.add_argument(
        "--segments",
        type=int,
        nargs="+",
        choices=range(1, 6),
        help="single_round: collect only selected logical segment numbers (1--5)",
    )
    p.add_argument("--no-skip", action="store_true", help="不跳过已存在的数据")
    p.add_argument("--run", type=int, help="补采模式：指定轮次")
    p.add_argument("--segment", type=int, help="补采模式：指定段，需要同时指定 --run")
    return p


def run(
    generator_factory: Callable[..., "object"],
    require_mavsdk_runtime: Callable[[], None],
    default_output_dir: Path,
    argv: Optional[list] = None,
) -> int:
    """执行 CLI。

    Parameters
    ----------
    generator_factory
        以 ``output_dir, px4_dir`` 构造 ``DatasetGenerator`` 的工厂。
    require_mavsdk_runtime
        在需要 mavsdk 的模式下做依赖检查；缺依赖时由它抛 ``RuntimeError``。
    default_output_dir
        ``--output-dir`` 未给定时的默认值（通常是 ``data_generation/data``）。
    """
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir
    gen = generator_factory(
        output_dir=output_dir,
        px4_dir=args.px4_dir,
        airframe=args.airframe,
    )

    if args.mode in {"multi_segment_160", "multi_segment_80", "single_round", "recover"}:
        require_mavsdk_runtime()

    if args.mode in {"multi_segment_160", "multi_segment_80"}:
        asyncio.run(
            gen.run_full_paper_automated(
                log_path=args.log,
                round_timeout=args.round_timeout,
                base_seed=args.seed,
                skip_existing=not args.no_skip,
            )
        )
    elif args.mode == "single_round":
        from .planner.segment_factory import generate_segment_configs_for_run
        from .planner.wind_factory import generate_wind_config

        async def one_round():
            dataset_type = args.dataset_type
            wind_config = generate_wind_config(
                dataset_type=dataset_type,
                seed=args.seed,
                run_index=0,
            )
            gust_label = "gust_ood" if dataset_type == "test_ood" else "gust_id"
            segment_configs = generate_segment_configs_for_run(
                dataset_type,
                1,
                ["steady", gust_label, gust_label, "steady", gust_label],
                seed=args.seed,
            )
            if args.segments:
                selected = set(args.segments)
                for index, segment in enumerate(segment_configs, start=1):
                    segment["_original_segment_index"] = index
                segment_configs = [
                    segment
                    for segment in segment_configs
                    if segment["_original_segment_index"] in selected
                ]
            await gen.run_multi_segment_session(
                wind_config,
                segment_configs,
                output_dir / dataset_type,
                1,
                run_number=1,
            )

        asyncio.run(one_round())
        print(
            f"单轮测试完成，见 {output_dir / args.dataset_type}/"
            "datasets-1-[1-5].json"
        )
    elif args.mode == "recover":
        if args.run is None:
            print("错误: recover 模式必须指定 --run")
            return 2
        if args.segment is not None:
            asyncio.run(
                gen.run_single_segment_recovery(args.run, args.segment, base_seed=args.seed)
            )
        else:
            asyncio.run(
                gen.run_single_run_recovery(args.run, base_seed=args.seed, log_path=args.log)
            )
    elif args.mode == "report":
        async def gen_report():
            report_path = output_dir.parent / "logs" / "missing_data_report.txt"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            await gen.generate_missing_report(report_path)

        asyncio.run(gen_report())

    return 0
