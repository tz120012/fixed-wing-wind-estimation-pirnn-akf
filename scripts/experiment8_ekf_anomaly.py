"""Run PX4-EKF2 under the SAME anomaly-injection windows used by the AKF/EMA
robustness sweep, so Table 4 can include a fair EKF row (addresses reviewer M6).

Reuses the exact 32 window configs (8 starts x 4 anomaly types, strength 3.0,
window 1000) from figure5_akf_vs_ema_anomaly_sweep_detail.csv and computes the
identical four metrics on the same anomaly slice.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from akf_experiment_utils import (
    closure_rmse, horizontal_rmse, jitter_mean, load_context, max_step_jump,
)
from src.experiments import paper_evidence_chain_eval as evidence

PX4_EKF_DIR = PROJECT_ROOT / "src/px4_ekf2"
SWEEP_DETAIL = PROJECT_ROOT / "data/figure5/figure5_akf_vs_ema_anomaly_sweep_detail.csv"
OUT = PROJECT_ROOT / "data/figure5/table4_ekf_anomaly_rows.csv"
AIRSPEED_COL = 19


def make_ekf():
    sys.path.insert(0, str(PX4_EKF_DIR))
    spec_path = PX4_EKF_DIR / "eval_px4_ekf2.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp8_px4_ekf2_eval", spec_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_ekf_window(module, X_anom, scaler_X, dt):
    estimator = module.PX4EKF2WindEstimator(dt=dt)
    pred = np.zeros((len(X_anom), 3), dtype=np.float64)
    for i in range(len(X_anom)):
        estimator.reset()
        Xd = scaler_X.inverse_transform(X_anom[i])
        last = np.zeros(3, dtype=np.float64)
        for t in range(Xd.shape[0]):
            last = estimator.step(Xd[t, 0:3], float(Xd[t, AIRSPEED_COL]))
        pred[i] = last
    return pred


def main():
    ctx = load_context("train_data1/train_lambda0.0_0.1_0.3_0.5_0.8_1.0_20260518_175113/train_lambda0.1_20260518_185047")
    dt = 1.0 / float(ctx.config["data"]["sampling_rate"])
    module = make_ekf()

    detail = pd.read_csv(SWEEP_DETAIL)
    configs = detail[["start_idx", "window_size", "anomaly_type", "anomaly_strength",
                      "anomaly_start_step", "anomaly_end_step"]].drop_duplicates().reset_index(drop=True)
    print(f"running EKF on {len(configs)} window configs")

    rows = []
    for k, c in configs.iterrows():
        s = int(c.start_idx); ws = int(c.window_size)
        X_window = ctx.X[s:s + ws].copy()
        wind_true = ctx.wind_true[s:s + ws]
        X_anom, a0, a1 = evidence.inject_anomaly(
            X_window, ctx.scaler_X, c.anomaly_type, float(c.anomaly_strength), seed=42
        )
        last_phys = evidence.denorm_last_step(X_anom, ctx.scaler_X)
        ekf_wind = run_ekf_window(module, X_anom, ctx.scaler_X, dt)
        sl = slice(a0, a1)
        bnd = slice(max(0, a0 - 20), min(ws, a1 + 20))
        rows.append({
            "method": "PX4-EKF2",
            "anomaly_window_h_rmse_mps": horizontal_rmse(wind_true[sl], ekf_wind[sl]),
            "anomaly_window_jitter_mean": jitter_mean(ekf_wind[sl]),
            "max_step_jump_mps": max_step_jump(ekf_wind[bnd]),
            "airspeed_closure_rmse_mps": closure_rmse(ekf_wind[sl], last_phys[sl]),
            "start_idx": s, "anomaly_type": c.anomaly_type, "anomaly_strength": c.anomaly_strength,
        })
        print(f"  [{k+1}/{len(configs)}] {c.anomaly_type}@{s}: "
              f"rmse={rows[-1]['anomaly_window_h_rmse_mps']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    metrics = ["anomaly_window_h_rmse_mps", "anomaly_window_jitter_mean",
               "max_step_jump_mps", "airspeed_closure_rmse_mps"]
    print("\n=== PX4-EKF2 aggregate over anomaly windows (mean +/- std) ===")
    for m in metrics:
        print(f"  {m:32s} {df[m].mean():.4f} +/- {df[m].std():.4f}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
