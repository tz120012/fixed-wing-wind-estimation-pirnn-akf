"""Experiment 9 - paired Wilcoxon signed-rank tests over the 12 transient windows.

Reproduces the significance statements in Section 3.4 (paragraph around Eq./line 498 of
FCGJ-v2.1.md): the tracking-vs-smoothing trade-off is evaluated as a *paired* comparison
across the N_WINDOWS windows produced by ``experiment9_transient_vs_anomaly.py`` (each window
gives one paired sample per method). We use ``scipy.stats.wilcoxon`` (two-sided, exact) on
the per-window metric pairs.

Metrics:
  * anomaly_jitter  -> lower is better (smoothing quality on the injected-spike segment)
  * transient_rmse  -> lower is better (tracking a genuine wind transient, no lag)

Input : data/figure5/akf_experiments/experiment9_transient_vs_anomaly_detail.csv
Output: prints a table and writes experiment9_wilcoxon_results.csv next to the input.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DETAIL_CSV = PROJECT_ROOT / "data/figure5/akf_experiments/experiment9_transient_vs_anomaly_detail.csv"
OUT_CSV = PROJECT_ROOT / "data/figure5/akf_experiments/experiment9_wilcoxon_results.csv"

# (label, metric, method_a, method_b)  ->  test whether method_a differs from method_b.
# Lower is better for both metrics, so a negative median(a-b) means method_a is better.
COMPARISONS = [
    ("Jitter: AKF vs PI-GRU raw", "anomaly_jitter", "PIRNN-AKF", "PI-GRU (Raw)"),
    ("Jitter: AKF vs EMA a=0.9", "anomaly_jitter", "PIRNN-AKF", "EMA a=0.9"),
    ("Transient RMSE: AKF vs PI-GRU raw", "transient_rmse", "PIRNN-AKF", "PI-GRU (Raw)"),
    ("Transient RMSE: AKF vs EMA a=0.9", "transient_rmse", "PIRNN-AKF", "EMA a=0.9"),
    ("Transient RMSE: AKF vs EMA a=0.1", "transient_rmse", "PIRNN-AKF", "EMA a=0.1"),
]


def paired_series(df: pd.DataFrame, metric: str, method: str) -> pd.Series:
    """Return the per-window metric for one method, indexed by window (start_idx, peak_t)."""
    sub = df[df["method"] == method].copy()
    sub = sub.set_index(["start_idx", "peak_t"]).sort_index()
    return sub[metric]


def main() -> None:
    if not DETAIL_CSV.exists():
        sys.exit(f"missing detail CSV: {DETAIL_CSV}\nrun experiment9_transient_vs_anomaly.py first")

    df = pd.read_csv(DETAIL_CSV)
    n_windows = df.groupby("method").size().max()
    print(f"loaded {DETAIL_CSV.name}: {n_windows} paired windows per method\n")

    rows = []
    for label, metric, m_a, m_b in COMPARISONS:
        a = paired_series(df, metric, m_a)
        b = paired_series(df, metric, m_b)
        common = a.index.intersection(b.index)
        a, b = a.loc[common].to_numpy(), b.loc[common].to_numpy()
        diff = a - b
        # two-sided exact Wilcoxon signed-rank on the paired differences
        stat, p = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", mode="exact")
        n_better = int(np.sum(diff < 0))  # method_a lower (better) on this many windows
        rows.append({
            "comparison": label,
            "metric": metric,
            "n_pairs": len(diff),
            "median_a": float(np.median(a)),
            "median_b": float(np.median(b)),
            "median_diff_a_minus_b": float(np.median(diff)),
            "n_a_better": n_better,
            "W": float(stat),
            "p_value": float(p),
        })

    res = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print(res.to_string(index=False))
    res.to_csv(OUT_CSV, index=False)
    print(f"\nsaved -> {OUT_CSV}")

    print("\n--- interpretation (two-sided, alpha=0.05) ---")
    for r in rows:
        verdict = "significant" if r["p_value"] < 0.05 else "NOT significant"
        direction = "AKF lower/better" if r["median_diff_a_minus_b"] < 0 else "AKF higher/worse"
        print(f"  {r['comparison']:<38} p={r['p_value']:.2e}  ({verdict}; {direction}, "
              f"{r['n_a_better']}/{r['n_pairs']} windows)")


if __name__ == "__main__":
    main()
