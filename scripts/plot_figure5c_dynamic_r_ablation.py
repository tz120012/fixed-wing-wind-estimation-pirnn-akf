from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import BLUE, GRAY, GREEN, PROJECT_ROOT, apply_style, format_axes, save_figure


DATA_PATH = PROJECT_ROOT / "data/figure5/akf_experiments/experiment5_dynamic_qr_ablation_summary.csv"
OUT_BASE = PROJECT_ROOT / "data/figure5/akf_experiments/figure5c_dynamic_r_ablation"

METHODS = ["PI-GRU (Raw)", "AKF fixed Q/R", "AKF dynamic Q/R"]
LABELS = ["PI-GRU", "PIRNN-AKF\nfixed $R$", "PIRNN-AKF\ndynamic $R$"]
COLORS = [GRAY, GREEN, BLUE]


def _percent_drop(before: float, after: float) -> float:
    return 100.0 * (before - after) / before


def main() -> None:
    apply_style()
    summary = pd.read_csv(DATA_PATH).set_index("method").loc[METHODS]

    # "vs PI-GRU" annotations use the CSV's own per-window mean-of-ratios
    # reduction columns (jitter/jump_reduction_vs_pigru_pct_mean), matching
    # how the same percentages are computed and reported in the manuscript
    # text/Table B1. This is deliberately NOT a ratio computed from the two
    # aggregate bar heights (which would give a slightly different number,
    # since mean-of-per-window-ratios != ratio-of-means for a heterogeneous
    # anomaly-window sample).
    dynamic_row = summary.loc["AKF dynamic Q/R"]
    raw_drop_pct = {
        "jitter": float(dynamic_row["jitter_reduction_vs_pigru_pct_mean"]),
        "jump": float(dynamic_row["jump_reduction_vs_pigru_pct_mean"]),
    }

    metrics = [
        ("jitter_mean", "Jitter Mean", "jitter"),
        ("max_jump_mean", "Max Step Jump (m/s)", "jump"),
    ]
    x = np.arange(len(METHODS))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.05))

    for ax, (col, ylabel, kind) in zip(axs, metrics):
        values = summary[col].to_numpy(dtype=float)
        ax.bar(x, values, color=COLORS, edgecolor="black", linewidth=0.8, width=0.68)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.text(
            0.04,
            0.92,
            "Lower values indicate\nstronger smoothing",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )

        raw_drop = raw_drop_pct[kind]
        fixed_drop = _percent_drop(values[1], values[2])
        ax.annotate(
            f"-{raw_drop:.1f}% vs PI-GRU\n-{fixed_drop:.1f}% vs fixed $R$",
            xy=(2, values[2]),
            xytext=(1.28, values.max() * 1.22),
            arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": BLUE},
            fontsize=7,
            color=BLUE,
            ha="left",
            va="bottom",
        )

        y_pad = values.max() * (0.62 if kind == "jitter" else 0.54)
        ax.set_ylim(0, values.max() + y_pad)
        format_axes(ax, grid_axis="y")

    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.22, top=0.86, wspace=0.34)
    save_figure(fig, OUT_BASE)


if __name__ == "__main__":
    main()
