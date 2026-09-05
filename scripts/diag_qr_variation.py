"""Diagnostic: does PI-GRU produce meaningful q_scale / r_scale variation?

Checks whether the network-predicted covariance scales actually vary across
the Test-OOD set, and whether they correlate with true-wind dynamics (which
should drive Q) and kinematic measurement residuals (which should drive R).
"""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from akf_experiment_utils import load_context
from src.experiments import paper_evidence_chain_eval as evidence


def summarize(name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float64)
    print(f"{name:14s} mean={a.mean():.4f} std={a.std():.4f} "
          f"min={a.min():.4f} p50={np.percentile(a,50):.4f} "
          f"max={a.max():.4f} cv={a.std()/ (abs(a.mean())+1e-9):.3f}")


def main() -> None:
    import sys as _sys
    model = _sys.argv[1] if len(_sys.argv) > 1 else "train_data1/train_lambda0.0_0.1_0.3_0.5_0.8_1.0_20260518_175113/train_lambda0.1_20260518_185047"
    ctx = load_context(model)
    n = min(len(ctx.X), 30000)
    out = evidence.predict_pigru(ctx.model, ctx.X[:n], 2000, ctx.device)
    q = out["q_scale"]
    r = out["r_scale"]
    print("=== q_scale per-axis ===")
    for ax, lbl in enumerate(["N", "E", "D"]):
        summarize(f"q_scale[{lbl}]", q[:, ax])
    print("=== r_scale per-axis ===")
    for ax, lbl in enumerate(["N", "E", "D"]):
        summarize(f"r_scale[{lbl}]", r[:, ax])

    wind_true = ctx.wind_true[:n]
    true_speed_change = np.linalg.norm(np.diff(wind_true[:, :2], axis=0), axis=1)
    true_speed_change = np.concatenate([[0.0], true_speed_change])

    last_phys = evidence.denorm_last_step(ctx.X[:n], ctx.scaler_X)
    pigru_wind = evidence.denorm_wind(out["wind"], ctx.scaler_y)
    vg = last_phys[:, 0:3]
    tas = last_phys[:, evidence.FEATURE_IDX["airspeed"]]
    closure_resid = np.abs(np.linalg.norm(vg - pigru_wind, axis=1) - tas)

    # windowed turbulence intensity (rolling std, W=50) — the trained target
    win = 50
    wtrue = wind_true[:, :2].astype(np.float64)
    ccs = np.cumsum(np.insert(wtrue, 0, 0.0, axis=0), axis=0)
    cc2 = np.cumsum(np.insert(wtrue * wtrue, 0, 0.0, axis=0), axis=0)
    ii = np.arange(len(wtrue))
    lo_i = np.maximum(0, ii - win + 1)
    kk = (ii + 1 - lo_i).reshape(-1, 1)
    mn = (ccs[ii + 1] - ccs[lo_i]) / kk
    vv = np.sqrt(np.maximum((cc2[ii + 1] - cc2[lo_i]) / kk - mn ** 2, 0.0))
    turb_intensity = vv.mean(axis=1)

    qm = q.mean(axis=1)
    rm = r.mean(axis=1)
    print("=== correlations ===")
    print(f"corr(q_mean, windowed_turb_intensity) = {np.corrcoef(qm, turb_intensity)[0,1]:.3f}")
    print(f"corr(q_mean, |d(true_wind)|) = {np.corrcoef(qm, true_speed_change)[0,1]:.3f}")
    print(f"corr(r_mean, closure_resid)  = {np.corrcoef(rm, closure_resid)[0,1]:.3f}")
    print(f"corr(q_mean, r_mean)         = {np.corrcoef(qm, rm)[0,1]:.3f}")
    print(f"q_mean dynamic range (p95/p5)= {np.percentile(qm,95)/ (np.percentile(qm,5)+1e-9):.3f}")
    print(f"r_mean dynamic range (p95/p5)= {np.percentile(rm,95)/ (np.percentile(rm,5)+1e-9):.3f}")


if __name__ == "__main__":
    main()
