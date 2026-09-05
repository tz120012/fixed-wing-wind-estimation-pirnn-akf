from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FPS_TO_MPS = 0.3048

WIND_COLS = [
    "/fdm/jsbsim/atmosphere/wind-north-fps",
    "/fdm/jsbsim/atmosphere/wind-east-fps",
    "/fdm/jsbsim/atmosphere/wind-down-fps",
    "wind_regime",
]

SPLITS = [
    ("train", "Train"),
    ("val", "Val"),
    ("test_id", "Test-ID"),
    ("test_ood", "Test-OOD"),
]

COLORS = {
    "Train": "#7F7F7F",
    "Val": "#00A087",
    "Test-ID": "#1F77B4",
    "Test-OOD": "#E64B35",
}

SPEED_BINS = [0.0, 1.0, 2.0, 3.2, 5.0, np.inf]
SPEED_BIN_LABELS = ["0-1", "1-2", "2-3.2", "3.2-5", ">5"]
SPEED_BIN_COLORS = ["#DCEAF6", "#A6CEE3", "#1F78B4", "#FB9A99", "#E31A1C"]

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.linewidth": 1.1,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
}


def clean_axes(ax) -> None:
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def load_split(split_dir: Path, split_label: str, row_stride: int) -> tuple[pd.DataFrame, dict]:
    files = sorted(split_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {split_dir}")

    frames = []
    total_rows = 0
    for csv_path in files:
        df = pd.read_csv(csv_path, usecols=lambda c: c in WIND_COLS)
        total_rows += len(df)
        if row_stride > 1:
            df = df.iloc[::row_stride].copy()
        df["split"] = split_label
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data["wind_north_mps"] = data["/fdm/jsbsim/atmosphere/wind-north-fps"] * FPS_TO_MPS
    data["wind_east_mps"] = data["/fdm/jsbsim/atmosphere/wind-east-fps"] * FPS_TO_MPS
    data["wind_down_mps"] = data["/fdm/jsbsim/atmosphere/wind-down-fps"] * FPS_TO_MPS
    data["wind_h_mps"] = np.hypot(data["wind_north_mps"], data["wind_east_mps"])
    data["wind_3d_mps"] = np.sqrt(
        data["wind_north_mps"] ** 2
        + data["wind_east_mps"] ** 2
        + data["wind_down_mps"] ** 2
    )
    data["wind_dir_rad"] = np.mod(
        np.arctan2(data["wind_east_mps"], data["wind_north_mps"]),
        2 * np.pi,
    )

    summary = {
        "split": split_label,
        "n_files": len(files),
        "n_rows_raw": total_rows,
        "n_rows_plotted": len(data),
    }
    return data, summary


def sample_for_scatter(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def add_split_summary_table(ax, summary: pd.DataFrame) -> None:
    ax.axis("off")
    table_data = summary[
        [
            "split",
            "n_files",
            "n_rows_raw",
            "wind_h_mean",
            "wind_h_p50",
            "wind_h_p95",
        ]
    ].copy()
    table_data["n_rows_raw"] = (table_data["n_rows_raw"] / 1000).map(lambda v: f"{v:.1f}k")
    for col in ["wind_h_mean", "wind_h_p50", "wind_h_p95"]:
        table_data[col] = table_data[col].map(lambda v: f"{v:.2f}")
    table_data.columns = ["Split", "Files", "Rows", "Mean", "P50", "P95"]

    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(weight="bold")
    ax.set_title("(e) Split Summary")


def plot_wind_rose(ax, df: pd.DataFrame, title: str) -> None:
    direction_edges = np.linspace(0, 2 * np.pi, 17)
    direction_centers = (direction_edges[:-1] + direction_edges[1:]) / 2
    width = np.diff(direction_edges) * 0.88
    bottom = np.zeros(len(direction_centers))
    total = max(len(df), 1)

    for lo, hi, label, color in zip(
        SPEED_BINS[:-1],
        SPEED_BINS[1:],
        SPEED_BIN_LABELS,
        SPEED_BIN_COLORS,
    ):
        mask = (df["wind_h_mps"] >= lo) & (df["wind_h_mps"] < hi)
        counts, _ = np.histogram(df.loc[mask, "wind_dir_rad"], bins=direction_edges)
        percentages = counts / total * 100.0
        ax.bar(
            direction_centers,
            percentages,
            width=width,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            align="center",
            label=f"{label} m/s",
        )
        bottom += percentages

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(title, pad=14)
    ax.set_rlabel_position(135)
    ax.grid(True, alpha=0.35, linestyle="--")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 8-2: wind-speed distribution and ID/OOD split overview."
    )
    parser.add_argument("--data-csv-dir", type=str, default="data/data_csv")
    parser.add_argument("--out-dir", type=str, default="data/figure8")
    parser.add_argument("--row-stride", type=int, default=5)
    parser.add_argument("--max-scatter-per-split", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_csv_dir = PROJECT_ROOT / args.data_csv_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    split_frames = []
    summaries = []
    for split_name, split_label in SPLITS:
        frame, summary = load_split(data_csv_dir / split_name, split_label, args.row_stride)
        split_frames.append(frame)
        summaries.append(summary)

    data = pd.concat(split_frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    stats = (
        data.groupby("split")["wind_h_mps"]
        .agg(
            wind_h_mean="mean",
            wind_h_std="std",
            wind_h_min="min",
            wind_h_p05=lambda s: np.percentile(s, 5),
            wind_h_p50="median",
            wind_h_p95=lambda s: np.percentile(s, 95),
            wind_h_max="max",
        )
        .reset_index()
    )
    summary = summary.merge(stats, on="split")
    summary.to_csv(out_dir / "figure8-2_wind_distribution_summary.csv", index=False)

    regime_counts = (
        data.groupby(["split", "wind_regime"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    regime_counts.to_csv(out_dir / "figure8-2_wind_regime_counts.csv", index=False)

    plt.rcParams.update(STYLE)
    fig = plt.figure(figsize=(14.2, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.04, 1.0])
    ax_rose_id = fig.add_subplot(gs[0, 0], projection="polar")
    ax_rose_ood = fig.add_subplot(gs[0, 1], projection="polar")
    ax_cdf = fig.add_subplot(gs[0, 2])
    ax_violin = fig.add_subplot(gs[1, 0:2])
    ax_table = fig.add_subplot(gs[1, 2])

    plot_wind_rose(
        ax_rose_id,
        data[data["split"] == "Test-ID"],
        "(a) Test-ID Wind Rose",
    )
    plot_wind_rose(
        ax_rose_ood,
        data[data["split"] == "Test-OOD"],
        "(b) Test-OOD Wind Rose",
    )
    handles, labels = ax_rose_ood.get_legend_handles_labels()
    ax_rose_ood.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.82, -0.10),
        frameon=True,
        edgecolor="black",
        fancybox=False,
        fontsize=8,
    )

    for _, split_label in SPLITS:
        sub = data[data["split"] == split_label]["wind_h_mps"].to_numpy()
        x, y = empirical_cdf(sub)
        ax_cdf.plot(
            x,
            y,
            linewidth=2.0,
            color=COLORS[split_label],
            linestyle="--" if split_label in {"Train", "Val"} else "-",
            label=split_label,
        )
    ax_cdf.axvspan(1.0, 3.2, color=COLORS["Test-ID"], alpha=0.08, label="ID nominal band")
    ax_cdf.axvspan(3.5, 5.0, color=COLORS["Test-OOD"], alpha=0.08, label="OOD extrapolation band")
    ax_cdf.set_title("(c) Horizontal Wind Speed CDF")
    ax_cdf.set_xlabel("Horizontal Wind Speed (m/s)")
    ax_cdf.set_ylabel("Cumulative Probability")
    ax_cdf.set_xlim(0, max(6.0, data["wind_h_mps"].quantile(0.995) * 1.05))
    ax_cdf.set_ylim(0, 1.0)
    clean_axes(ax_cdf)

    order = [label for _, label in SPLITS]
    values = [data.loc[data["split"] == label, "wind_h_mps"].to_numpy() for label in order]
    parts = ax_violin.violinplot(
        values,
        positions=np.arange(1, len(order) + 1),
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, split_label in zip(parts["bodies"], order):
        body.set_facecolor(COLORS[split_label])
        body.set_edgecolor("black")
        body.set_alpha(0.28)
        body.set_linewidth(1.0)

    for pos, split_label in enumerate(order, start=1):
        sub = data.loc[data["split"] == split_label, "wind_h_mps"]
        q1, median, q3 = np.percentile(sub, [25, 50, 75])
        ax_violin.plot([pos - 0.18, pos + 0.18], [median, median], color="black", linewidth=2.0)
        ax_violin.plot([pos, pos], [q1, q3], color="black", linewidth=1.6)
        ax_violin.scatter(
            pos,
            sub.mean(),
            s=26,
            color=COLORS[split_label],
            edgecolor="black",
            zorder=3,
        )
    ax_violin.set_xticks(np.arange(1, len(order) + 1))
    ax_violin.set_xticklabels(order, rotation=12)
    ax_violin.set_title("(d) Split-Level Speed Distribution")
    ax_violin.set_ylabel("Horizontal Wind Speed (m/s)")
    ax_violin.set_ylim(0, max(6.0, data["wind_h_mps"].quantile(0.995) * 1.05))
    clean_axes(ax_violin)

    add_split_summary_table(ax_table, summary)

    fig.tight_layout()
    png_path = out_dir / "figure8-2_wind_distribution.png"
    svg_path = out_dir / "figure8-2_wind_distribution.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 8-2 to {png_path}")
    print(f"Saved Figure 8-2 to {svg_path}")


if __name__ == "__main__":
    main()
