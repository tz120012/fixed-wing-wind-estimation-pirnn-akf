"""Decisive check: is true-wind turbulence intensity predictable from the
observable INPUT features at all?

High temporal autocorrelation of the target only means it is *slowly varying*,
not that it is a learnable function of X. Here we correlate the windowed
turbulence intensity of the TRUE wind against windowed variability of the
observable inputs (ground velocity, airspeed, gyro, accel). If these are
uncorrelated, no amount of retraining can make q_scale track turbulence.
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from akf_experiment_utils import load_context
from src.experiments import paper_evidence_chain_eval as evidence


def rolling_std(x, win):
    x = np.asarray(x, dtype=np.float64)
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    c2 = np.cumsum(np.insert(x * x, 0, 0.0, axis=0), axis=0)
    i = np.arange(len(x))
    lo = np.maximum(0, i - win + 1)
    k = (i + 1 - lo).reshape(-1, 1) if x.ndim > 1 else (i + 1 - lo)
    mean = (c[i + 1] - c[lo]) / k
    var = (c2[i + 1] - c2[lo]) / k - mean ** 2
    return np.sqrt(np.maximum(var, 0.0))


def main():
    model = "train_data1/train_lambda0.01_0.03_0.05_0.07_0.09_0.11_0.13_0.15_0.17_0.19_20260601_111456/train_lambda0.1_20260601_111507"
    ctx = load_context(model)
    n = min(len(ctx.X), 40000)
    win = 50

    wind_true = ctx.wind_true[:n, :2]
    turb = rolling_std(wind_true, win).mean(axis=1)

    last_phys = evidence.denorm_last_step(ctx.X[:n], ctx.scaler_X)
    # observable windowed variability proxies
    vg = last_phys[:, 0:3]
    tas = last_phys[:, evidence.FEATURE_IDX["airspeed"]]
    feats = {
        "vg_horiz_std": rolling_std(vg[:, :2], win).mean(axis=1),
        "vg_down_std": rolling_std(vg[:, 2:3], win).ravel(),
        "airspeed_std": rolling_std(tas, win),
    }
    print(f"true turbulence intensity: mean={turb.mean():.3f} cv={turb.std()/(turb.mean()+1e-9):.3f}")
    print("=== correlation of OBSERVABLE windowed variability with TRUE turbulence intensity ===")
    for name, f in feats.items():
        print(f"  corr({name}, turb) = {np.corrcoef(f, turb)[0,1]:.3f}")

    # also: how well does PI-GRU wind's own windowed std (a network-internal proxy) track true turbulence?
    out = evidence.predict_pigru(ctx.model, ctx.X[:n], 2000, ctx.device)
    pigru_wind = evidence.denorm_wind(out["wind"], ctx.scaler_y)[:, :2]
    pigru_turb = rolling_std(pigru_wind, win).mean(axis=1)
    print(f"  corr(PI-GRU wind windowed std, true turb) = {np.corrcoef(pigru_turb, turb)[0,1]:.3f}")


if __name__ == "__main__":
    main()
