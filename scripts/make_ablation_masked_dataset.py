"""make_ablation_masked_dataset.py

为「输入特征组消融」生成一份数据副本：把 src/2_pigru_module.py 里 FEATURE_GROUPS
中指定组的特征列，在已归一化的 X_*.npy 上置 0（= 该组特征恒为训练集均值，
不携带任何信息），其余文件（y_*.npy / w_*.npy / turn_class_*.npy / norm_params.pkl）
原样复制。

不修改任何训练/模型代码：3_train_pigru.py 本身支持
`--processed_dir_override` 指向任意预处理目录，因此消融只需生成一份新的
数据目录并把该参数指过去即可，训练/评估逻辑完全不变。

用法示例：
  # 消融“实际舵面”组（38-41），只处理 train/val/test_id/test_ood 四个 X 文件
  .venv/bin/python scripts/make_ablation_masked_dataset.py \
      --groups actuator \
      --src_dir data/dataset_new_processed \
      --out_dir data/dataset_ablation_actuator

  # 消融“舵面指令”组（15-18）
  .venv/bin/python scripts/make_ablation_masked_dataset.py \
      --groups control_cmd \
      --out_dir data/dataset_ablation_control_cmd

  # 可一次消融多组，逗号分隔： --groups actuator,imu_accel
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from importlib import import_module

_pigru_module = import_module("2_pigru_module")
FEATURE_GROUPS = _pigru_module.FEATURE_GROUPS
resolve_feature_mask_indices = _pigru_module.resolve_feature_mask_indices

X_FILES = ["X_train.npy", "X_val.npy", "X_test_id.npy", "X_test_ood.npy"]
COPY_ONLY_GLOBS = [
    "y_*.npy", "w_*.npy", "turn_class_*.npy", "norm_params.pkl",
]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a feature-group-masked copy of the processed dataset.")
    p.add_argument("--groups", required=True, help="comma-separated FEATURE_GROUPS keys to zero out")
    p.add_argument("--src_dir", default="data/dataset_new_processed", help="source processed dir (relative to project root)")
    p.add_argument("--out_dir", required=True, help="output dir for masked copy (relative to project root)")
    p.add_argument("--dry_run", action="store_true", help="only print the plan, do not write files")
    args = p.parse_args()

    group_names = [g.strip() for g in args.groups.split(",") if g.strip()]
    mask_idx = resolve_feature_mask_indices(group_names)
    print(f"[mask] groups={group_names} -> zeroed feature indices={mask_idx}")
    for g in group_names:
        print(f"        {g}: {FEATURE_GROUPS[g]}")

    src_dir = PROJECT_ROOT / args.src_dir
    out_dir = PROJECT_ROOT / args.out_dir
    if not src_dir.is_dir():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for xfile in X_FILES:
        src_path = src_dir / xfile
        if not src_path.exists():
            print(f"[skip] {xfile} not found in src_dir")
            continue
        dst_path = out_dir / xfile
        print(f"[mask] {xfile}: loading ...")
        if args.dry_run:
            print(f"        [dry-run] would zero cols {mask_idx} and save to {dst_path}")
            continue
        X = np.load(src_path)
        before = float(np.abs(X[:, :, mask_idx]).mean())
        X[:, :, mask_idx] = 0.0
        after = float(np.abs(X[:, :, mask_idx]).mean())
        np.save(dst_path, X)
        print(f"        shape={X.shape} mean|masked_cols| {before:.6f} -> {after:.6f}, saved -> {dst_path}")
        del X

    for pattern in COPY_ONLY_GLOBS:
        for src_path in sorted(src_dir.glob(pattern)):
            dst_path = out_dir / src_path.name
            if args.dry_run:
                print(f"[copy] [dry-run] {src_path.name}")
                continue
            shutil.copy2(src_path, dst_path)
            print(f"[copy] {src_path.name} -> {dst_path}")

    tag = "ablation_" + "_".join(group_names)
    print(f"\n[done] masked dataset written to: {out_dir}")
    print("Next: train with e.g. (real 3_train_pigru.py CLI flags, no code changes):")
    print(f"  .venv/bin/python src/3_train_pigru.py \\\n"
          f"      --mode single --lambda_list 0.1 \\\n"
          f"      --processed_dir_override {args.out_dir} \\\n"
          f"      --model_save_path_override data/p0_feat_ablation \\\n"
          f"      --run_tag {tag}")


if __name__ == "__main__":
    main()
