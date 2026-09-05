"""Moving-block bootstrap 95% CIs and paired-difference tests for the main-table RMSE.

Addresses reviewer M4 (statistical rigor) WITHOUT retraining: operates on the cached
per-sample predictions. Because consecutive windows overlap (temporal correlation),
we use a moving-block bootstrap (contiguous blocks resampled with replacement) rather
than i.i.d. resampling, which would understate uncertainty.

For each method we report the seed-ensemble RMSE and its 95% CI; for the key claims we
report paired RMSE-difference CIs on identical resamples (a CI excluding 0 => significant).
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "data/figure2/fig2_20260526_122624/predictions"
EKF_NPZ = ROOT / "data/figure2/fig2_20260526_122624/ekf_predictions.npz"
KN_DIR = ROOT / "data/baseline_kalmannet"
SEEDS = [26, 42, 2026]
SPLITS = ["test_id", "test_ood"]
BLOCK = 256          # ~5 s of windows; respects temporal correlation
B = 4000             # bootstrap resamples
RNG = np.random.default_rng(20260602)


def load_se(split):
    """Return dict method -> seed-averaged per-sample MSE (axis-meaned) (N,).

    RMSE = sqrt(mean_sample se) equals the pooled multi-seed RMSE, matching Table 1.
    """
    d0 = np.load(PRED_DIR / f"seed{SEEDS[0]}_{split}.npz")
    wt = d0["wind_true"].astype(np.float64)
    n = len(wt)
    out = {}
    for key in ["vanilla_gru", "pigru", "pirnn_akf"]:
        acc = np.zeros(n)
        for s in SEEDS:
            p = np.load(PRED_DIR / f"seed{s}_{split}.npz")[key].astype(np.float64)
            acc += ((p - wt) ** 2).mean(axis=1)
        out[key] = acc / len(SEEDS)
    acc = np.zeros(n)
    for s in SEEDS:
        p = np.load(KN_DIR / f"seed{s}_{split}_kalmannet.npy").astype(np.float64)
        acc += ((p - wt) ** 2).mean(axis=1)
    out["kalmannet"] = acc / len(SEEDS)
    e = np.load(EKF_NPZ)[split].astype(np.float64)
    out["ekf"] = ((e - wt) ** 2).mean(axis=1)
    return out, n


def block_indices(n, block, rng):
    """Build a resampled index vector of length ~n from contiguous blocks."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


def rmse_from_se(se, idx):
    return np.sqrt(se[idx].mean())


def main():
    name = {"ekf": "PX4-EKF2", "kalmannet": "KalmanNet", "vanilla_gru": "Vanilla GRU",
            "pigru": "PI-GRU", "pirnn_akf": "PIRNN-AKF"}
    order = ["ekf", "kalmannet", "vanilla_gru", "pigru", "pirnn_akf"]
    for split in SPLITS:
        se, n = load_se(split)

        # precompute bootstrap index sets once -> paired across methods
        boot_rmse = {m: np.empty(B) for m in order}
        diff_keys = [("pigru", "pirnn_akf"), ("kalmannet", "pigru"), ("kalmannet", "pirnn_akf")]
        boot_diff = {k: np.empty(B) for k in diff_keys}
        for b in range(B):
            idx = block_indices(n, BLOCK, RNG)
            for m in order:
                boot_rmse[m][b] = rmse_from_se(se[m], idx)
            for a, c in diff_keys:
                boot_diff[(a, c)][b] = rmse_from_se(se[a], idx) - rmse_from_se(se[c], idx)

        print(f"\n=== {split}  (moving-block bootstrap, block={BLOCK}, B={B}, 3-axis RMSE) ===")
        print(f"{'method':14s}{'RMSE':>9}{'95% CI':>22}")
        for m in order:
            pt = np.sqrt(se[m].mean())
            lo, hi = np.percentile(boot_rmse[m], [2.5, 97.5])
            print(f"{name[m]:14s}{pt:>9.3f}   [{lo:.3f}, {hi:.3f}]")
        print("  paired RMSE differences (95% CI; excludes 0 => significant):")
        for a, c in diff_keys:
            d = boot_diff[(a, c)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            sig = "significant" if (lo > 0 or hi < 0) else "n.s. (overlaps 0)"
            print(f"    {name[a]} - {name[c]:11s}: {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  {sig}")


if __name__ == "__main__":
    main()
