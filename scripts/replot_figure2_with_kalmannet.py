"""Replot Figure 2 as RMSE + direction-MAE comparison.

Reads cached figure-2 summary for EKF/Vanilla/PI-GRU/PIRNN-AKF and computes the
KalmanNet per-seed RMSE live from its cached predictions. Direction MAE values
are taken from the audited main table so the figure reflects the paper's
"amplitude vs direction" narrative directly.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import BLUE, GRAY, GREEN, ORANGE, RED, apply_style, format_axes, save_figure

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "data/figure2/fig2_20260526_122624"
PRED_DIR = FIG_DIR / "predictions"
KN_DIR = ROOT / "data/baseline_kalmannet"
SEEDS = [26, 42, 2026]
SPLITS = ["test_id", "test_ood"]

METHOD_ORDER = ["EKF", "KalmanNet", "Vanilla GRU", "PI-GRU", "PIRNN-AKF"]
METHOD_LABELS = {"EKF": "PX4-EKF2", "KalmanNet": "KalmanNet", "Vanilla GRU": "Vanilla GRU",
                 "PI-GRU": "PI-GRU", "PIRNN-AKF": "PIRNN-AKF"}
SPLIT_LABELS = {"test_id": "Test-ID", "test_ood": "Test-OOD"}
METHOD_COLORS = {
    "EKF": GRAY,
    "KalmanNet": GREEN,
    "Vanilla GRU": ORANGE,
    "PI-GRU": BLUE,
    "PIRNN-AKF": RED,
}

# Audited direction MAE values from Table 1. Direction MAE is evaluated on
# samples with horizontal wind speed >= 0.5 m/s to avoid arctan2 singularity.
DIR_MAE = {
    ("EKF", "test_id"): 13.08,
    ("EKF", "test_ood"): 24.12,
    ("KalmanNet", "test_id"): 13.20,
    ("KalmanNet", "test_ood"): 5.92,
    ("Vanilla GRU", "test_id"): 3.86,
    ("Vanilla GRU", "test_ood"): 3.18,
    ("PI-GRU", "test_id"): 3.34,
    ("PI-GRU", "test_ood"): 2.13,
    ("PIRNN-AKF", "test_id"): 3.43,
    ("PIRNN-AKF", "test_ood"): 2.14,
}


def rmse(wt, wp):
    return float(np.sqrt(np.mean((wp - wt) ** 2)))


def kalmannet_summary():
    rows = []
    for split in SPLITS:
        vals = []
        for seed in SEEDS:
            wt = np.load(PRED_DIR / f"seed{seed}_{split}.npz")["wind_true"].astype(np.float64)
            wp = np.load(KN_DIR / f"seed{seed}_{split}_kalmannet.npy").astype(np.float64)
            vals.append(rmse(wt, wp))
        rows.append({"method": "KalmanNet", "split": split,
                     "rmse_mean": float(np.mean(vals)), "rmse_std": float(np.std(vals, ddof=0))})
    return pd.DataFrame(rows)


def main():
    summary = pd.read_csv(FIG_DIR / "figure2_metrics_summary.csv")
    summary = summary[summary["method"].isin(["EKF", "Vanilla GRU", "PI-GRU", "PIRNN-AKF"])]
    summary = pd.concat([summary, kalmannet_summary()], ignore_index=True)

    apply_style()
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    x = np.arange(len(METHOD_ORDER))
    width = 0.36
    hatches = {"test_id": "", "test_ood": "//"}

    for ax, metric_name in zip(axs, ["rmse", "direction"]):
        for idx, split in enumerate(SPLITS):
            if metric_name == "rmse":
                sub = summary[summary["split"] == split].set_index("method")
                means = [float(sub.loc[m, "rmse_mean"]) if m in sub.index else np.nan for m in METHOD_ORDER]
                stds = [float(sub.loc[m, "rmse_std"]) if m in sub.index else 0.0 for m in METHOD_ORDER]
                ylabel = "3D Wind RMSE (m/s)"
                title = "(a) Wind-Speed Error"
            else:
                means = [DIR_MAE[(m, split)] for m in METHOD_ORDER]
                stds = [0.0 for _ in METHOD_ORDER]
                ylabel = "Direction MAE (deg)"
                title = "(b) Horizontal Direction Error"

            bars = ax.bar(
                x + (idx - 0.5) * width,
                means,
                width,
                yerr=stds if metric_name == "rmse" else None,
                capsize=3 if metric_name == "rmse" else 0,
                label=SPLIT_LABELS[split],
                color=[METHOD_COLORS[m] for m in METHOD_ORDER],
                alpha=0.92,
                edgecolor="black",
                linewidth=0.8,
            )
            for bar in bars:
                bar.set_hatch(hatches[split])

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=18, ha="right")
        format_axes(ax, grid_axis="y")
        if metric_name == "direction":
            ax.set_ylim(0, 27)
        if metric_name == "rmse":
            from matplotlib.patches import Patch

            legend_handles = [
                Patch(facecolor="0.85", edgecolor="black", linewidth=0.8, hatch=hatches[s] * 2,
                      label=SPLIT_LABELS[s])
                for s in SPLITS
            ]
            ax.legend(
                handles=legend_handles,
                frameon=True,
                edgecolor="black",
                fancybox=False,
                loc="upper right",
                ncol=2,
                fontsize=7,
                handlelength=1.4,
                columnspacing=0.8,
            )

    fig.tight_layout(w_pad=0.8)

    save_figure(fig, FIG_DIR / "figure2_rmse_bar", copy_to_paper="figure2")
    print("replotted Figure 2 with KalmanNet ->", FIG_DIR, "and paper/figures")
    print(summary.sort_values(["split", "method"]).to_string(index=False))


if __name__ == "__main__":
    main()
