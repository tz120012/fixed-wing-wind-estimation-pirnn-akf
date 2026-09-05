"""Experiment 10 - anomaly-strength robustness sweep across multiple anomaly types.

Extends ``experiment1_akf_dynamic_anomaly_strength.py`` (single anomaly type) to a full
sweep over anomaly *type* x *strength*, answering the reviewer question "you only test one
anomaly strength -- where is the robustness boundary?".

For every anomaly type with a strength semantic (GPS speed spike, TAS spike, attitude spike,
Gaussian burst) we inject the anomaly in the middle 45%-55% of each evaluation window at a
range of strengths, and measure, on the anomaly segment:
  * h_rmse   - horizontal tracking RMSE vs ground-truth wind (lower=better, "does it still track?")
  * jitter   - output jitter (lower=better, "is it smooth under the spike?")
  * max_jump - max single-step excursion near the anomaly boundary (leakage proxy)

Methods compared: PI-GRU (raw front-end), PIRNN-AKF (adaptive backend), and fixed EMA
low-pass baselines. Output: detail/summary CSV + a 2-row (tracking / smoothness) x N-column
(anomaly type) figure showing how each method degrades as anomaly strength grows.

Runs on CPU fine (set CUDA_VISIBLE_DEVICES="" if the local CUDA build lacks a matching kernel).
"""
from __future__ import annotations

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
    set_plot_style,
)
from paper_plot_style import BLUE, RED, save_figure

ANOMALY_LABELS = {
    "gps_spike": "GPS speed spike",
    "tas_spike": "TAS spike",
    "attitude_spike": "Attitude spike",
    "gaussian_burst": "Gaussian burst",
    "sensor_dropout": "Sensor dropout",
}

PLOT_METHODS = ["PI-GRU (Raw)", "PIRNN-AKF", "EMA alpha=0.1", "EMA alpha=0.9"]
METHOD_STYLE = {
    "PI-GRU (Raw)": dict(color=BLUE, marker="o", ls="-", label="PI-GRU (raw)"),
    "PIRNN-AKF": dict(color=RED, marker="s", ls="-", label="PIRNN-AKF"),
    "EMA alpha=0.1": dict(color="#9E9E9E", marker="^", ls="--", label=r"EMA $\alpha=0.1$"),
    "EMA alpha=0.9": dict(color="#4D4D4D", marker="v", ls=":", label=r"EMA $\alpha=0.9$"),
}


def plot(summary: pd.DataFrame, anomaly_types: list[str], out_dir: Path) -> None:
    set_plot_style()
    n = len(anomaly_types)
    fig, axs = plt.subplots(2, n, figsize=(3.2 * n, 6.0), squeeze=False)
    panels = [("h_rmse_mean", "Anomaly-window horizontal RMSE (m/s)"),
              ("jitter_mean", "Anomaly-window jitter (m/s)")]
    for col, atype in enumerate(anomaly_types):
        sub_a = summary[summary["anomaly_type"] == atype]
        for row, (metric, ylabel) in enumerate(panels):
            ax = axs[row][col]
            for method in PLOT_METHODS:
                s = sub_a[sub_a["method"] == method].sort_values("strength")
                if s.empty:
                    continue
                st = METHOD_STYLE[method]
                ax.plot(s["strength"], s[metric], color=st["color"], marker=st["marker"],
                        ls=st["ls"], label=st["label"], markersize=5, lw=1.6)
            ax.grid(True, alpha=0.3, ls="--")
            ax.tick_params(axis="both", direction="in", top=True, right=True)
            if row == 0:
                ax.set_title(ANOMALY_LABELS.get(atype, atype), fontweight="bold")
            if row == 1:
                ax.set_xlabel("Anomaly strength")
            if col == 0:
                ax.set_ylabel(ylabel)
    axs[0][n - 1].legend(frameon=True, edgecolor="black", fancybox=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, out_dir / "experiment10_anomaly_strength_multitype")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 10: multi-type anomaly-strength sweep.")
    parser.add_argument("--model", required=True, help="model dir containing best_model.pth (relative to project root)")
    parser.add_argument("--out_dir", default="data/figureD1_anomaly_strength")
    parser.add_argument("--window_size", type=int, default=1000)
    parser.add_argument("--starts", default=None)
    parser.add_argument("--n_windows", type=int, default=6)
    parser.add_argument("--anomaly_types", default="gps_spike,tas_spike,attitude_spike,gaussian_burst")
    parser.add_argument("--strengths", default="0.5,1,2,3,5,8")
    parser.add_argument("--ema_alphas", default="0.1,0.5,0.9")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_context(args.model)
    starts = parse_starts(args.starts, len(ctx.X), args.window_size, args.n_windows)
    strengths = parse_floats(args.strengths)
    anomaly_types = parse_strings(args.anomaly_types)
    ema_alphas = parse_floats(args.ema_alphas)

    rows = []
    for atype in anomaly_types:
        for strength in strengths:
            for start in starts:
                df, _ = run_case(ctx, start, args.window_size, atype, strength, ema_alphas)
                rows.append(df)
        print(f"[done] anomaly_type={atype}")

    detail = add_relative_metrics(pd.concat(rows, ignore_index=True))
    summary = aggregate(detail, ["anomaly_type", "strength", "method"])
    detail.to_csv(out_dir / "experiment10_anomaly_strength_detail.csv", index=False)
    summary.to_csv(out_dir / "experiment10_anomaly_strength_summary.csv", index=False)
    plot(summary, anomaly_types, out_dir)

    pd.set_option("display.width", 240)
    key = summary[summary["method"].isin(PLOT_METHODS)][
        ["anomaly_type", "strength", "method", "h_rmse_mean", "jitter_mean",
         "max_jump_mean", "jitter_reduction_vs_pigru_pct_mean"]
    ]
    print(key.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
