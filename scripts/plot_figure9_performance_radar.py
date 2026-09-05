"""[补充材料图 · 非正文图号] 多视角性能雷达图。

注意：本脚本输出到 ``data/figure9/``，文件名中的 "figure9" 为历史编号，与
主线论文 FCGJ-v2.1.md 正文的"图 9 / 图 10"**没有对应关系**——正文图 9 为
HITL 实验平台照片，正文图 10 为数据集重播时序 + 推理延迟 CDF
（见 ``paper/figures/figure10.*``）。本雷达图仅作补充/探索性可视化，
未纳入正文，整理复现材料时请勿与正文图混淆。
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
}

METHOD_ORDER = ["EKF", "Vanilla GRU", "PI-GRU", "PIRNN-AKF"]
COLORS = {
    "EKF": "#7F7F7F",
    "Vanilla GRU": "#E64B35",
    "PI-GRU": "#4DBBD5",
    "PIRNN-AKF": "#1F77B4",
}
LINESTYLES = {
    "EKF": ":",
    "Vanilla GRU": "--",
    "PI-GRU": "-.",
    "PIRNN-AKF": "-",
}


def inverse_minmax(values: pd.Series) -> pd.Series:
    """Convert an error metric to a 0-1 score where larger is better."""
    vmin = values.min()
    vmax = values.max()
    if np.isclose(vmax, vmin):
        return pd.Series(np.ones(len(values)), index=values.index)
    return (vmax - values) / (vmax - vmin)


def minmax(values: pd.Series) -> pd.Series:
    vmin = values.min()
    vmax = values.max()
    if np.isclose(vmax, vmin):
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - vmin) / (vmax - vmin)


def build_split_scores(raw: pd.DataFrame, split: str) -> pd.DataFrame:
    """Build normalized scores for a single split from raw Figure 2 metrics."""
    methods = [m for m in METHOD_ORDER if m in set(raw["method"])]
    sub = raw[(raw["split"] == split) & (raw["method"].isin(methods))]
    metrics = (
        sub.groupby("method")[["rmse", "mae", "rmse_n", "rmse_e", "rmse_d"]]
        .mean()
        .loc[methods]
    )

    scores = pd.DataFrame(index=metrics.index)
    scores["Overall RMSE"] = inverse_minmax(metrics["rmse"])
    scores["Overall MAE"] = inverse_minmax(metrics["mae"])
    scores["North RMSE"] = inverse_minmax(metrics["rmse_n"])
    scores["East RMSE"] = inverse_minmax(metrics["rmse_e"])
    scores["Down RMSE"] = inverse_minmax(metrics["rmse_d"])
    return scores


def plot_radar(ax, scores: pd.DataFrame, title: str) -> None:
    labels = list(scores.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    for method, values in scores.iterrows():
        vals = values.to_numpy(dtype=float)
        vals = np.concatenate([vals, [vals[0]]])
        ax.plot(
            angles,
            vals,
            linewidth=2.2 if method == "PIRNN-AKF" else 1.6,
            linestyle=LINESTYLES.get(method, "-"),
            color=COLORS.get(method, "black"),
            label=method,
        )
        ax.fill(
            angles,
            vals,
            color=COLORS.get(method, "black"),
            alpha=0.11 if method == "PIRNN-AKF" else 0.035,
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_title(title, pad=16)
    ax.grid(True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 9: normalized multi-metric performance radar."
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default="data/figure2/fig2_20260526_122624/figure2_metrics_summary.csv",
    )
    parser.add_argument(
        "--raw-metrics-csv",
        type=str,
        default="data/figure2/fig2_20260526_122624/figure2_metrics_raw.csv",
    )
    parser.add_argument("--out-dir", type=str, default="data/figure9")
    args = parser.parse_args()

    metrics_csv = PROJECT_ROOT / args.metrics_csv
    raw_metrics_csv = PROJECT_ROOT / args.raw_metrics_csv
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing Figure 2 summary metrics: {metrics_csv}")
    if not raw_metrics_csv.exists():
        raise FileNotFoundError(f"Missing Figure 2 raw metrics: {raw_metrics_csv}")

    raw = pd.read_csv(raw_metrics_csv)
    id_scores = build_split_scores(raw, "test_id")
    ood_scores = build_split_scores(raw, "test_ood")
    id_scores.to_csv(out_dir / "figure9_test_id_radar_scores.csv")
    ood_scores.to_csv(out_dir / "figure9_test_ood_radar_scores.csv")

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), subplot_kw={"polar": True})
    plot_radar(axes[0], id_scores, "(a) Test-ID Performance")
    plot_radar(axes[1], ood_scores, "(b) Test-OOD Performance")

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.04),
        ncol=4,
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, wspace=0.32)
    png_path = out_dir / "figure9_performance_radar.png"
    svg_path = out_dir / "figure9_performance_radar.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 9 to {png_path}")


if __name__ == "__main__":
    main()
