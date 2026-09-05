import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from akf_experiment_utils import (
    PROJECT_ROOT,
    load_context,
    parse_starts,
    parse_strings,
    run_case,
)
from paper_plot_style import BLUE, GRAY, GREEN, RED, apply_style, format_axes, save_figure


def plot(diag: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    # Plot the representative GPS spike case to show the dynamic-R mechanism.
    cases = diag[["start_idx", "anomaly_type", "strength"]].drop_duplicates()
    gps_cases = cases[cases["anomaly_type"] == "gps_spike"]
    first = (gps_cases if not gps_cases.empty else cases).iloc[0]
    sub = diag[
        (diag["start_idx"] == first["start_idx"])
        & (diag["anomaly_type"] == first["anomaly_type"])
        & (diag["strength"] == first["strength"])
    ].copy()
    # Keep compatibility with cached old diagnostics; regenerated diagnostics already use 50 Hz.
    if sub["time_s"].max() > 25:
        sub["time_s"] = sub["step"] * 0.02

    fig, axs = plt.subplots(3, 1, figsize=(7.0, 4.9), sharex=True)
    axs[0].plot(sub["time_s"], sub["r_scale_mean"], color=RED, linewidth=1.5)
    axs[0].set_ylabel("Mean $R$-scale")
    axs[1].plot(sub["time_s"], sub["akf_weight"], color=BLUE, linewidth=1.5, label="AKF state weight")
    axs[1].plot(sub["time_s"], sub["nn_weight"], color=GREEN, linewidth=1.5, linestyle="--", label="PI-GRU pseudo-measurement weight")
    axs[1].set_ylabel("Fusion Weight")
    axs[1].legend(frameon=True, edgecolor="black", fancybox=False)
    axs[2].plot(sub["time_s"], sub["innovation_norm"], color=GRAY, linewidth=1.5)
    axs[2].set_ylabel("Innovation Norm")
    axs[2].set_xlabel("Time (s)")
    for ax in axs:
        ax.axvspan(450 * 0.02, 550 * 0.02, color=RED, alpha=0.1)
        format_axes(ax)
    fig.tight_layout()
    save_figure(fig, out_dir / "experiment4_akf_internal_diagnostics")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: AKF internal diagnostics under anomalies.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out_dir", default="data/figure5/akf_experiments")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--starts", default="29000")
    parser.add_argument("--anomaly_types", default="gps_spike,gaussian_burst")
    parser.add_argument("--anomaly_strength", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_context(args.model)
    starts = parse_starts(args.starts, len(ctx.X), args.window_size, 1)
    anomaly_types = parse_strings(args.anomaly_types)
    details, diags = [], []
    for start in starts:
        for idx, anomaly_type in enumerate(anomaly_types):
            detail, diag = run_case(ctx, start, args.window_size, anomaly_type, args.anomaly_strength, [], seed=42 + idx)
            details.append(detail)
            diags.append(diag)
    detail_df = pd.concat(details, ignore_index=True)
    diag_df = pd.concat(diags, ignore_index=True)

    # Mechanism summary: compare normal vs anomaly window means.
    rows = []
    for keys, sub in diag_df.groupby(["start_idx", "anomaly_type", "strength"]):
        anom = sub[(sub["step"] >= 450) & (sub["step"] < 550)]
        normal = sub[(sub["step"] < 430) | (sub["step"] >= 570)]
        row = {"start_idx": keys[0], "anomaly_type": keys[1], "strength": keys[2]}
        for col in ["r_scale_mean", "akf_weight", "nn_weight", "innovation_norm"]:
            row[f"{col}_normal_mean"] = float(normal[col].mean())
            row[f"{col}_anomaly_mean"] = float(anom[col].mean())
            row[f"{col}_ratio"] = float(anom[col].mean() / normal[col].mean()) if normal[col].mean() else np.nan
        rows.append(row)
    mechanism = pd.DataFrame(rows)
    detail_df.to_csv(out_dir / "experiment4_akf_internal_diagnostics_detail.csv", index=False)
    diag_df.to_csv(out_dir / "experiment4_akf_internal_diagnostics_timeseries.csv", index=False)
    mechanism.to_csv(out_dir / "experiment4_akf_internal_diagnostics_summary.csv", index=False)
    plot(diag_df, out_dir)
    print(mechanism.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
