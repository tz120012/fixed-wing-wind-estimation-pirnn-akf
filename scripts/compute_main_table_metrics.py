"""Compute extended main-table metrics (RMSE, MAE, per-axis RMSE, direction MAE/P95)
for EKF / Vanilla GRU / PI-GRU / PIRNN-AKF on Test-ID and Test-OOD from cached
predictions. No model re-run, no fabrication: reads the figure-2 prediction npz
files (per seed) and the EKF prediction npz.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "data/figure2/fig2_20260526_122624/predictions"
EKF_NPZ = ROOT / "data/figure2/fig2_20260526_122624/ekf_predictions.npz"
KALMANNET_DIR = ROOT / "data/baseline_kalmannet"
SEEDS = [26, 42, 2026]
SPLITS = ["test_id", "test_ood"]
DIR_MIN_WH = 0.5  # m/s; exclude near-zero horizontal wind to avoid arctan2 singularity


def direction_mae_p95(wt, wp, min_wh=DIR_MIN_WH):
    wh = np.linalg.norm(wt[:, :2], axis=1)
    m = wh >= min_wh
    td = np.degrees(np.arctan2(wt[m, 1], wt[m, 0]))
    pd_ = np.degrees(np.arctan2(wp[m, 1], wp[m, 0]))
    diff = np.abs((pd_ - td + 180.0) % 360.0 - 180.0)
    return float(np.mean(diff)), float(np.percentile(diff, 95))


def metrics(wt, wp):
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
    }


def main():
    rows = []
    learn_methods = ["vanilla_gru", "pigru", "kalmannet", "pirnn_akf"]
    for split in SPLITS:
        per = {m: [] for m in learn_methods}
        for seed in SEEDS:
            d = np.load(PRED_DIR / f"seed{seed}_{split}.npz")
            wt = d["wind_true"].astype(np.float64)
            for m in learn_methods:
                if m == "kalmannet":
                    pred = np.load(KALMANNET_DIR / f"seed{seed}_{split}_kalmannet.npy").astype(np.float64)
                    assert len(pred) == len(wt), f"KalmanNet len mismatch {len(pred)} vs {len(wt)}"
                else:
                    pred = d[m].astype(np.float64)
                per[m].append(metrics(wt, pred))
        for m in learn_methods:
            df = pd.DataFrame(per[m])
            agg = {f"{c}_mean": df[c].mean() for c in df.columns}
            agg.update({f"{c}_std": df[c].std(ddof=0) for c in df.columns})
            agg.update(method=m, split=split, n_seeds=len(SEEDS))
            rows.append(agg)
        # EKF (single run)
        e = np.load(EKF_NPZ)
        wt = np.load(PRED_DIR / f"seed{SEEDS[0]}_{split}.npz")["wind_true"].astype(np.float64)
        em = metrics(wt, e[split].astype(np.float64))
        em.update({f"{k}_mean": v for k, v in em.items()})
        em.update(method="ekf", split=split, n_seeds=1)
        rows.append(em)

    out = pd.DataFrame(rows)
    out_path = ROOT / "data/figure2/main_table_extended_metrics.csv"
    out.to_csv(out_path, index=False)

    name = {"ekf": "PX4-EKF2", "vanilla_gru": "Vanilla GRU", "pigru": "PI-GRU",
            "kalmannet": "KalmanNet", "pirnn_akf": "PIRNN-AKF"}
    order = ["ekf", "kalmannet", "vanilla_gru", "pigru", "pirnn_akf"]
    for split in SPLITS:
        print(f"\n=== {split} (dir MAE filtered |w_h|>={DIR_MIN_WH} m/s) ===")
        print(f"{'method':14s}{'RMSE':>10}{'MAE':>9}{'N_rmse':>9}{'E_rmse':>9}{'D_rmse':>9}{'dirMAE':>9}{'dirP95':>9}")
        sub = out[out.split == split].set_index("method")
        for m in order:
            r = sub.loc[m]
            sd = r.get('rmse_std', 0.0)
            print(f"{name[m]:14s}{r['rmse_mean']:>10.3f}{r['mae_mean']:>9.3f}{r['north_rmse_mean']:>9.3f}"
                  f"{r['east_rmse_mean']:>9.3f}{r['down_rmse_mean']:>9.3f}{r['dir_mae_mean']:>9.2f}{r['dir_p95_mean']:>9.2f}"
                  f"   (rmse std={sd:.3f})")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
