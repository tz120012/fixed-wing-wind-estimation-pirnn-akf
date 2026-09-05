import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from akf_experiment_utils import (
    PROJECT_ROOT,
    add_relative_metrics,
    aggregate,
    horizontal_rmse,
    jitter_mean,
    load_context,
    parse_floats,
    parse_starts,
    run_case,
    savefig_all,
    set_plot_style,
)


def regime_score(wind_true: np.ndarray) -> float:
    # High values indicate fast-changing true wind, where over-smoothing is costly.
    return float(np.mean(np.linalg.norm(np.diff(wind_true[:, :2], axis=0), axis=1)))


def plot(summary: pd.DataFrame, out_dir: Path) -> None:
    set_plot_style()
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.9))
    methods = ["PI-GRU (Raw)", "PIRNN-AKF", "EMA alpha=0.1", "EMA alpha=0.5"]
    colors = {"PI-GRU (Raw)": "#7F7F7F", "PIRNN-AKF": "#1F77B4", "EMA alpha=0.1": "#00A087", "EMA alpha=0.5": "#E64B35"}
    for method in methods:
        sub = summary[summary["method"] == method].sort_values("regime")
        if sub.empty:
            continue
        label = method.replace(" alpha=", r" $\alpha=$")
        axs[0].plot(sub["regime"], sub["h_rmse_mean"], marker="o", color=colors.get(method), label=label)
        axs[1].plot(sub["regime"], sub["jitter_mean"], marker="o", color=colors.get(method), label=label)
    axs[0].set_ylabel("Horizontal RMSE (m/s)")
    axs[1].set_ylabel("Jitter Mean")
    for ax, title in zip(axs, ["(a) Tracking Across Regimes", "(b) Smoothing Across Regimes"]):
        ax.set_xlabel("Regime")
        ax.set_title(title)
        ax.grid(True)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
    axs[1].legend(frameon=True, edgecolor="black", fancybox=False)
    fig.tight_layout()
    savefig_all(fig, out_dir / "experiment2_regime_switching")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: low/high dynamics regime switching.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_dir", default="data/figure5/akf_experiments")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--starts", default=None)
    parser.add_argument("--n_windows", type=int, default=12)
    parser.add_argument("--anomaly_type", default="gaussian_burst")
    parser.add_argument("--anomaly_strength", type=float, default=3.0)
    parser.add_argument("--ema_alphas", default="0.1,0.5,0.9")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_context(args.model)
    starts = parse_starts(args.starts, len(ctx.X), args.window_size, args.n_windows)
    scored = []
    for start in starts:
        scored.append((start, regime_score(ctx.wind_true[start:start + args.window_size])))
    median_score = float(np.median([s for _, s in scored]))
    start_to_regime = {start: ("high-dynamics" if score >= median_score else "low-dynamics") for start, score in scored}

    rows = []
    ema_alphas = parse_floats(args.ema_alphas)
    for start, _ in scored:
        df, _ = run_case(ctx, start, args.window_size, args.anomaly_type, args.anomaly_strength, ema_alphas)
        df["regime"] = start_to_regime[start]
        rows.append(df)
    detail = add_relative_metrics(pd.concat(rows, ignore_index=True))
    summary = aggregate(detail, ["regime", "method"])
    detail.to_csv(out_dir / "experiment2_regime_switching_detail.csv", index=False)
    summary.to_csv(out_dir / "experiment2_regime_switching_summary.csv", index=False)
    pd.DataFrame(scored, columns=["start_idx", "true_wind_dynamic_score"]).assign(
        regime=lambda d: d["start_idx"].map(start_to_regime)
    ).to_csv(out_dir / "experiment2_regime_window_scores.csv", index=False)
    plot(summary, out_dir)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
