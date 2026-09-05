"""
运行 PI-GRU 物理损失对照实验。

默认用途：
1. 保持现有数据与训练配置不变
2. 用小 lambda_physics 先比较 `attitude` vs `6dof`
3. 避免手工反复修改 config.yaml

示例：
    python src/3c_run_physics_ablation.py
    python src/3c_run_physics_ablation.py --lambda_list 0.01,0.02,0.05 --physics_modes attitude,6dof
    python src/3c_run_physics_ablation.py --dry_run
"""

import argparse
import os
import subprocess
import sys
from typing import List


DEFAULT_LAMBDA_LIST = "0.02,0.05,0.1"
DEFAULT_PHYSICS_MODES = "attitude,6dof"


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(',') if item.strip()]


def build_run_tag(base_tag, physics_mode: str) -> str:
    cleaned_mode = physics_mode.strip().lower()
    if not base_tag:
        return f"pilot_{cleaned_mode}"
    return f"{base_tag}_{cleaned_mode}"


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行 attitude/6dof 小权重对照实验")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="可选配置文件路径，默认使用 config/config.yaml",
    )
    parser.add_argument(
        "--lambda_list",
        type=str,
        default=DEFAULT_LAMBDA_LIST,
        help="逗号分隔的小权重列表，默认 0.02,0.05,0.1",
    )
    parser.add_argument(
        "--physics_modes",
        type=str,
        default=DEFAULT_PHYSICS_MODES,
        help="逗号分隔的物理模式列表，默认 attitude,6dof",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="physics_ablation",
        help="实验标签前缀，会自动追加 physics_mode",
    )
    parser.add_argument(
        "--processed_dir_override",
        type=str,
        default=None,
        help="覆盖训练所用预处理数据目录",
    )
    parser.add_argument(
        "--model_save_path_override",
        type=str,
        default=None,
        help="覆盖模型保存目录",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="只打印将执行的命令，不真正运行",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    train_script = os.path.join(script_dir, '3_train_pigru.py')
    config_path = args.config_path or os.path.join(project_root, 'config', 'config.yaml')

    lambda_values = parse_csv_list(args.lambda_list)
    physics_modes = parse_csv_list(args.physics_modes)
    if not lambda_values:
        raise ValueError("lambda_list 不能为空")
    if not physics_modes:
        raise ValueError("physics_modes 不能为空")

    print("=" * 70)
    print(" 运行 physics ablation")
    print("=" * 70)
    print(f"配置文件: {config_path}")
    print(f"lambda_list: {', '.join(lambda_values)}")
    print(f"physics_modes: {', '.join(physics_modes)}")
    print(f"processed_dir_override: {args.processed_dir_override or '-'}")
    print(f"model_save_path_override: {args.model_save_path_override or '-'}")
    print(f"dry_run: {args.dry_run}")

    for idx, physics_mode in enumerate(physics_modes, start=1):
        run_tag = build_run_tag(args.run_tag, physics_mode)
        command = [
            sys.executable,
            train_script,
            "--mode", "sweep",
            "--config_path", config_path,
            "--lambda_list", ",".join(lambda_values),
            "--physics_mode", physics_mode,
            "--run_tag", run_tag,
        ]
        if args.processed_dir_override:
            command.extend(["--processed_dir_override", args.processed_dir_override])
        if args.model_save_path_override:
            command.extend(["--model_save_path_override", args.model_save_path_override])

        print("\n" + "#" * 70)
        print(f"[{idx}/{len(physics_modes)}] physics_mode={physics_mode}")
        print("命令:")
        print(" ".join(command))

        if args.dry_run:
            continue

        subprocess.run(command, check=True, cwd=project_root)

    print("\n" + "=" * 70)
    print("✅ physics ablation 任务已完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
