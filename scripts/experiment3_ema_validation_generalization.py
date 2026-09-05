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
    parse_strings,
    run_case,
    savefig_all,
    set_plot_style,
)


def plot(test_summary: pd.DataFrame, best_alpha: float, out_dir: Path) -> None:
    set_plot_style()
    keep = ["PI-GRU (Raw)", "PIRNN-AKF", f"EMA alpha={best_alpha:g}"]
    sub = test_summary[test_summary["method"].isin(keep)].copy()
    sub["method"] = pd.Categorical(sub["method"], categories=keep, ordered=True)
    sub = sub.sort_values("method")
    labels = ["PI-GRU", "PIRNN-AKF", fr"EMA $\alpha={best_alpha:g}$"]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.7))
    metrics = [("h_rmse_mean", "Horizontal RMSE"), ("jitter_mean", "Jitter"), ("balanced_score", "Balanced Score")]
    for ax, (col, title) in zip(axs, metrics):
        ax.bar(np.arange(len(sub)), sub[col], color=["#7F7F7F", "#1F77B4", "#00A087"], edgecolor="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(sub)))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, axis="y")
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
    fig.tight_layout()
    savefig_all(fig, out_dir / "experiment3_ema_validation_generalization")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: validation-tuned EMA generalization.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_dir", default="data/figure5/akf_experiments")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--starts", default=None)
    parser.add_argument("--n_windows", type=int, default=10)
    parser.add_argument("--val_fraction", type=float, default=0.4)
    parser.add_argument("--anomaly_types", default="gps_spike,tas_spike,sensor_dropout,gaussian_burst")
    parser.add_argument("--anomaly_strength", type=float, default=3.0)
    parser.add_argument("--ema_alphas", default="0.1,0.3,0.5,0.7,0.9")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_context(args.model)
    starts = parse_starts(args.starts, len(ctx.X), args.window_size, args.n_windows)
    split_idx = max(1, int(len(starts) * args.val_fraction))
    val_starts, test_starts = starts[:split_idx], starts[split_idx:]
    anomaly_types = parse_strings(args.anomaly_types)
    ema_alphas = parse_floats(args.ema_alphas)

    rows = []
    for split, split_starts in [("val", val_starts), ("test", test_starts)]:
        for start in split_starts:
            for anomaly_idx, anomaly_type in enumerate(anomaly_types):
                df, _ = run_case(ctx, start, args.window_size, anomaly_type, args.anomaly_strength, ema_alphas, seed=42 + anomaly_idx)
                df["split"] = split
                rows.append(df)
    detail = add_relative_metrics(pd.concat(rows, ignore_index=True))
    val_summary = aggregate(detail[detail["split"] == "val"], ["method"])
    test_summary = aggregate(detail[detail["split"] == "test"], ["method"])
    ema_val = val_summary[val_summary["method"].str.startswith("EMA alpha=")].sort_values("balanced_score")
    best_method = str(ema_val.iloc[0]["method"])
    best_alpha = float(best_method.split("=")[-1])

    detail.to_csv(out_dir / "experiment3_ema_validation_generalization_detail.csv", index=False)
    val_summary.to_csv(out_dir / "experiment3_ema_validation_generalization_val_summary.csv", index=False)
    test_summary.to_csv(out_dir / "experiment3_ema_validation_generalization_test_summary.csv", index=False)
    pd.DataFrame({
        "selected_ema_method": [best_method],
        "selected_alpha": [best_alpha],
        "val_starts": [",".join(map(str, val_starts))],
        "test_starts": [",".join(map(str, test_starts))],
    }).to_csv(out_dir / "experiment3_ema_validation_generalization_selection.csv", index=False)
    plot(test_summary, best_alpha, out_dir)
    print("Selected EMA:", best_method)
    print(test_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
