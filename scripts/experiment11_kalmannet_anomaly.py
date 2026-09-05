"""Run the KalmanNet baseline under the SAME anomaly-injection windows used by the
AKF/EMA robustness sweep, so Table 4 can include a fair neural-augmented-KF row
(addresses reviewer M3).

Reuses the exact window configs from figure5_akf_vs_ema_anomaly_sweep_detail.csv and
computes the identical four metrics on the same anomaly slice, mirroring
experiment8_ekf_anomaly.py.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from akf_experiment_utils import (
    closure_rmse, horizontal_rmse, jitter_mean, load_context, max_step_jump,
)
from src.experiments import paper_evidence_chain_eval as evidence
from train_kalmannet_baseline import KIN_IDX, KalmanNet, kinematic_measurement, predict

SWEEP_DETAIL = PROJECT_ROOT / "data/figure5/figure5_akf_vs_ema_anomaly_sweep_detail.csv"
CKPT = PROJECT_ROOT / "data/baseline_kalmannet/kalmannet_seed26.pth"
OUT = PROJECT_ROOT / "data/figure5/table4_kalmannet_anomaly_rows.csv"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ctx = load_context("train_data1/train_lambda0.0_0.1_0.3_0.5_0.8_1.0_20260518_175113/train_lambda0.1_20260518_185047")

    meta = pickle.load(open(PROJECT_ROOT / "data/dataset_new_processed/norm_params.pkl", "rb"))
    sx = meta["scaler_X"]
    mean = torch.tensor(sx.mean_[KIN_IDX], device=device, dtype=torch.float32)
    scale = torch.tensor(sx.scale_[KIN_IDX], device=device, dtype=torch.float32)

    model = KalmanNet(hidden_size=64).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device)["model_state_dict"])
    model.eval()

    detail = pd.read_csv(SWEEP_DETAIL)
    configs = detail[["start_idx", "window_size", "anomaly_type", "anomaly_strength",
                      "anomaly_start_step", "anomaly_end_step"]].drop_duplicates().reset_index(drop=True)
    print(f"running KalmanNet on {len(configs)} window configs")

    rows = []
    for k, c in configs.iterrows():
        s = int(c.start_idx); ws = int(c.window_size)
        X_window = ctx.X[s:s + ws].copy()
        wind_true = ctx.wind_true[s:s + ws]
        X_anom, a0, a1 = evidence.inject_anomaly(
            X_window, ctx.scaler_X, c.anomaly_type, float(c.anomaly_strength), seed=42
        )
        last_phys = evidence.denorm_last_step(X_anom, ctx.scaler_X)
        kn_wind = predict(model, X_anom.astype(np.float32), mean, scale, device).astype(np.float64)
        sl = slice(a0, a1)
        bnd = slice(max(0, a0 - 20), min(ws, a1 + 20))
        rows.append({
            "method": "KalmanNet",
            "anomaly_window_h_rmse_mps": horizontal_rmse(wind_true[sl], kn_wind[sl]),
            "anomaly_window_jitter_mean": jitter_mean(kn_wind[sl]),
            "max_step_jump_mps": max_step_jump(kn_wind[bnd]),
            "airspeed_closure_rmse_mps": closure_rmse(kn_wind[sl], last_phys[sl]),
            "start_idx": s, "anomaly_type": c.anomaly_type, "anomaly_strength": c.anomaly_strength,
        })
        print(f"  [{k+1}/{len(configs)}] {c.anomaly_type}@{s}: "
              f"rmse={rows[-1]['anomaly_window_h_rmse_mps']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    metrics = ["anomaly_window_h_rmse_mps", "anomaly_window_jitter_mean",
               "max_step_jump_mps", "airspeed_closure_rmse_mps"]
    print("\n=== KalmanNet aggregate over anomaly windows (mean +/- std) ===")
    for m in metrics:
        print(f"  {m:32s} {df[m].mean():.4f} +/- {df[m].std():.4f}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
