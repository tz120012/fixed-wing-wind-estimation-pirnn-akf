"""eval_feature_ablation.py

对「输入特征组消融」训练出的 PI-GRU checkpoint，在其对应的（同样置零）
Test-ID / Test-OOD 数据上跑一遍前向推理，输出与主表口径一致的
RMSE / MAE / 分轴 RMSE / 风向 MAE+P95，方便与 main_table_extended_metrics.csv
里未消融的 PI-GRU 基线直接对比。

只读取现有 checkpoint 做推理，不做任何训练、不改任何训练/模型代码：
复用 src/experiments/paper_evidence_chain_eval.py 里已经在用的
load_pigru_model / predict_pigru / denorm_wind / denorm_y 等函数。

用法：
  .venv/bin/python scripts/eval_feature_ablation.py \
      --data-dir data/dataset_ablation_actuator \
      --model data/p0_feat_ablation/train_lambda0.1_20260719_151148/train_lambda0.1_20260719_151155/best_model.pth \
      --tag ablation_actuator

  .venv/bin/python scripts/eval_feature_ablation.py \
      --data-dir data/dataset_ablation_control_cmd \
      --model data/p0_feat_ablation/train_lambda0.1_20260719_161908/train_lambda0.1_20260719_161919/best_model.pth \
      --tag ablation_control_cmd
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_module(module_path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EVID = load_module(PROJECT_ROOT / "src" / "experiments" / "paper_evidence_chain_eval.py", "evidence_chain_eval")


def direction_mae_p95(wt: np.ndarray, wp: np.ndarray, min_wh: float = 0.5):
    wh = np.linalg.norm(wt[:, :2], axis=1)
    m = wh >= min_wh
    td = np.degrees(np.arctan2(wt[m, 1], wt[m, 0]))
    pd_ = np.degrees(np.arctan2(wp[m, 1], wp[m, 0]))
    diff = np.abs((pd_ - td + 180.0) % 360.0 - 180.0)
    return float(np.mean(diff)), float(np.percentile(diff, 95))


def metrics(wt: np.ndarray, wp: np.ndarray) -> dict:
    err = wp - wt
    dmae, dp95 = direction_mae_p95(wt, wp)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "north_rmse": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "east_rmse": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "down_rmse": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "dir_mae": dmae,
        "dir_p95": dp95,
        "n": int(len(wt)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="masked processed dir (relative to project root)")
    ap.add_argument("--model", required=True, help="path to best_model.pth (relative to project root)")
    ap.add_argument("--splits", default="test_id,test_ood")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = PROJECT_ROOT / args.data_dir
    model_path = PROJECT_ROOT / args.model

    scaler_X, scaler_y = EVID.load_norm_params(data_dir)
    model = EVID.load_pigru_model(model_path, scaler_X, scaler_y, device)

    print(f"\n===== tag={args.tag or args.model} =====")
    print(f"data_dir={data_dir}")
    print(f"model={model_path}")

    for split in args.splits.split(","):
        X = np.load(data_dir / f"X_{split}.npy")
        y = np.load(data_dir / f"y_{split}.npy")
        pred = EVID.predict_pigru(model, X, args.batch_size, device)
        wind_pred = EVID.denorm_wind(pred["wind"], scaler_y)
        y_denorm = EVID.denorm_y(y, scaler_y)
        wind_true = y_denorm[:, :3]
        m = metrics(wind_true.astype(np.float64), wind_pred.astype(np.float64))
        print(f"[{split}] n={m['n']} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} "
              f"N={m['north_rmse']:.4f} E={m['east_rmse']:.4f} D={m['down_rmse']:.4f} "
              f"dirMAE={m['dir_mae']:.2f} dirP95={m['dir_p95']:.2f}")


if __name__ == "__main__":
    main()
