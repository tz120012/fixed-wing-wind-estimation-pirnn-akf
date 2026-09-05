"""Re-plot the appendix anomaly robustness summary (Figure B1).

The plot mirrors Appendix Table B1 using direct metrics instead of an undefined
balanced score.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import BLUE, CYAN, GRAY, GREEN, ORANGE, RED, apply_style, format_axes, save_figure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data/figure5"
CSV = OUT_DIR / "figure5_akf_vs_ema_anomaly_sweep_summary.csv"


def main():
    apply_style()
    summary = pd.read_csv(CSV)
    extra = pd.DataFrame([
        {
            "method": "PX4-EKF2",
            "anomaly_window_h_rmse_mps_mean": 1.544,
            "anomaly_window_jitter_mean_mean": 0.179,
            "max_step_jump_mps_mean": 0.892,
            "airspeed_closure_rmse_mps_mean": 3.285,
        },
        {
            "method": "KalmanNet",
            "anomaly_window_h_rmse_mps_mean": 1.422,
            "anomaly_window_jitter_mean_mean": 1.800,
            "max_step_jump_mps_mean": 4.148,
            "airspeed_closure_rmse_mps_mean": 2.699,
        },
    ])
    summary = pd.concat([extra, summary], ignore_index=True)
    order = ["PX4-EKF2", "KalmanNet", "PI-GRU (Raw)", "EMA alpha=0.1", "EMA alpha=0.5", "EMA alpha=0.9", "PIRNN-AKF"]
    summary = summary.set_index("method").loc[order].reset_index()

    label_map = {
        "PX4-EKF2": "PX4-EKF2",
        "KalmanNet": "KalmanNet",
        "PI-GRU (Raw)": "PI-GRU",
        "PIRNN-AKF": "PIRNN-AKF",
        "EMA alpha=0.1": r"EMA ($\alpha=0.1$)",
        "EMA alpha=0.5": r"EMA ($\alpha=0.5$)",
        "EMA alpha=0.9": r"EMA ($\alpha=0.9$)",
    }
    color_map = {
        "PX4-EKF2": GRAY,
        "KalmanNet": GREEN,
        "PI-GRU (Raw)": BLUE,
        "PIRNN-AKF": RED,
    }
    colors = [color_map.get(m, ORANGE if "EMA" in m else CYAN) for m in summary["method"]]

    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.2))
    metrics = [
        ("anomaly_window_h_rmse_mps_mean", "Horizontal RMSE (m/s)", "(a) Tracking Error"),
        ("anomaly_window_jitter_mean_mean", "Jitter Mean", "(b) Output Jitter"),
        ("max_step_jump_mps_mean", "Max Step Jump (m/s)", "(c) Instantaneous Jump"),
        ("airspeed_closure_rmse_mps_mean", "Closure RMSE (m/s)", "(d) Airspeed Closure"),
    ]
    x = np.arange(len(summary))
    labels = [label_map.get(m, m) for m in summary["method"]]
    for ax, (col, ylabel, title) in zip(axs.ravel(), metrics):
        ax.bar(x, summary[col], color=colors, edgecolor="black", linewidth=0.8, width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        format_axes(ax, grid_axis="y")

    fig.tight_layout()
    save_figure(fig, OUT_DIR / "figure5_akf_vs_ema_anomaly_sweep")
    save_figure(fig, PROJECT_ROOT / "paper/figures/figureB1_anomaly_sweep")
    print("re-plotted appendix Figure B1 from cache")


if __name__ == "__main__":
    main()
