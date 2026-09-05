import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from akf_experiment_utils import (
    PROJECT_ROOT,
    add_relative_metrics,
    aggregate,
    load_context,
    parse_floats,
    parse_starts,
    run_case,
    savefig_all,
    set_plot_style,
)


def plot(summary: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    methods = ["PI-GRU (Raw)", "PIRNN-AKF", "EMA alpha=0.5"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    for method in methods:
        sub = summary[summary["method"] == method].sort_values("strength")
        label = method.replace(" alpha=", r" $\alpha=$")
        axs[0].plot(sub["strength"], sub["h_rmse_mean"], marker="o", label=label)
        axs[1].plot(sub["strength"], sub["jitter_mean"], marker="o", label=label)
        axs[2].plot(sub["strength"], sub["balanced_score"], marker="o", label=label)
    axs[0].set_ylabel("Horizontal RMSE (m/s)")
    axs[1].set_ylabel("Jitter Mean")
    axs[2].set_ylabel("Balanced Score")
    for ax, title in zip(axs, ["(a) Tracking Error", "(b) Jitter", "(c) Balanced Cost"]):
        ax.set_xlabel("Anomaly Strength")
        ax.set_title(title)
        ax.grid(True)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
    axs[2].legend(frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()
    savefig_all(fig, out_dir / "experiment1_dynamic_anomaly_strength")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: anomaly-strength sweep.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_dir", default="data/figure5/akf_experiments")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--starts", default=None)
    parser.add_argument("--n_windows", type=int, default=6)
    parser.add_argument("--anomaly_type", default="gps_spike")
    parser.add_argument("--strengths", default="1,2,3,5,8")
    parser.add_argument("--ema_alphas", default="0.1,0.5,0.9")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_context(args.model)
    starts = parse_starts(args.starts, len(ctx.X), args.window_size, args.n_windows)
    strengths = parse_floats(args.strengths)
    ema_alphas = parse_floats(args.ema_alphas)

    rows = []
    for strength in strengths:
        for start in starts:
            df, _ = run_case(ctx, start, args.window_size, args.anomaly_type, strength, ema_alphas)
            rows.append(df)
    detail = add_relative_metrics(pd.concat(rows, ignore_index=True))
    summary = aggregate(detail, ["strength", "method"])
    detail.to_csv(out_dir / "experiment1_dynamic_anomaly_strength_detail.csv", index=False)
    summary.to_csv(out_dir / "experiment1_dynamic_anomaly_strength_summary.csv", index=False)
    plot(summary, out_dir)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
