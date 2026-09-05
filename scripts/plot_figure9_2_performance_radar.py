"""[补充材料图 · 非正文图号] 多视角性能雷达图（v2）。

注意：本脚本输出到 ``data/figure9/``，文件名中的 "figure9" 为历史编号，与
主线论文 FCGJ-v2.1.md 正文的"图 9 / 图 10"**没有对应关系**——正文图 9 为
HITL 实验平台照片，正文图 10 为数据集重播时序 + 推理延迟 CDF
（见 ``paper/figures/figure10.*``）。本雷达图仅作补充/探索性可视化，
未纳入正文，整理复现材料时请勿与正文图混淆。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

METHOD_ORDER = [
    "EKF",
    "Vanilla GRU",
    "PI-GRU",
    "PI-GRU + Fixed KF",
    "PIRNN-AKF",
]

COLORS = {
    "EKF": "#7F7F7F",
    "Vanilla GRU": "#E64B35",
    "PI-GRU": "#4DBBD5",
    "PI-GRU + Fixed KF": "#00A087",
    "PIRNN-AKF": "#1F77B4",
}

LINESTYLES = {
    "EKF": ":",
    "Vanilla GRU": "--",
    "PI-GRU": "-.",
    "PI-GRU + Fixed KF": (0, (3, 1, 1, 1)),
    "PIRNN-AKF": "-",
}

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "axes.linewidth": 1.1,
    "grid.alpha": 0.40,
    "grid.linestyle": "--",
}

# Supplemental OOD metrics used by the manuscript sections on physical
# consistency, temporal smoothness, and edge deployment. Accuracy is loaded
# from Figure 2 metrics when available.
SUPPLEMENTAL_METRICS = {
    "EKF": {
        "closure_rmse_mps": 0.760,
        "jitter_mean": 0.0102,
        "latency_ms": 0.20,
    },
    "Vanilla GRU": {
        "closure_rmse_mps": 0.704,
        "jitter_mean": 0.0090,
        "latency_ms": 0.78,
    },
    "PI-GRU": {
        "closure_rmse_mps": 0.631,
        "jitter_mean": 0.00700,
        "latency_ms": 0.84,
    },
    "PI-GRU + Fixed KF": {
        "closure_rmse_mps": 0.598,
        "jitter_mean": 0.00476,
        "latency_ms": 0.87,
    },
    "PIRNN-AKF": {
        "closure_rmse_mps": 0.549,
        "jitter_mean": 0.00514,
        "latency_ms": 0.884,
    },
}


def inverse_minmax(values: pd.Series) -> pd.Series:
    """Normalize an error/cost metric to a score where larger is better."""
    vmin = float(values.min())
    vmax = float(values.max())
    if np.isclose(vmax, vmin):
        return pd.Series(np.ones(len(values)), index=values.index)
    return (vmax - values) / (vmax - vmin)


def read_ood_rmse(summary_csv: Path) -> pd.Series:
    summary = pd.read_csv(summary_csv)
    sub = summary[summary["split"] == "test_ood"].copy()
    rmse = sub.set_index("method")["rmse_mean"]
    missing = [m for m in METHOD_ORDER if m not in rmse.index]
    if missing:
        raise ValueError(f"Missing OOD RMSE for methods: {missing}")
    return rmse.loc[METHOD_ORDER]


def read_hitl_latency(hitl_dir: Path) -> dict[str, float]:
    """Read Raspberry Pi HITL latency logs when available."""
    mapping = {
        "hitl_data_20260416_210242.csv": "PI-GRU",
        "hitl_data_20260416_214252.csv": "PIRNN-AKF",
    }
    latency = {}
    for filename, method in mapping.items():
        path = hitl_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=lambda c: c == "inference_ms")
        if len(df) > 100:
            df = df.iloc[100:]
        if not df.empty:
            latency[method] = float(df["inference_ms"].dropna().mean())
    if "PI-GRU" in latency:
        latency["Vanilla GRU"] = min(latency["PI-GRU"], SUPPLEMENTAL_METRICS["Vanilla GRU"]["latency_ms"])
    if "PIRNN-AKF" in latency:
        latency["PI-GRU + Fixed KF"] = latency["PIRNN-AKF"]
    return latency


def build_raw_metrics(summary_csv: Path, hitl_dir: Path) -> pd.DataFrame:
    ood_rmse = read_ood_rmse(summary_csv)
    measured_latency = read_hitl_latency(hitl_dir)

    rows = []
    for method in METHOD_ORDER:
        supplemental = SUPPLEMENTAL_METRICS[method]
        rows.append(
            {
                "method": method,
                "ood_rmse_mps": float(ood_rmse.loc[method]),
                "closure_rmse_mps": supplemental["closure_rmse_mps"],
                "jitter_mean": supplemental["jitter_mean"],
                "latency_ms": measured_latency.get(method, supplemental["latency_ms"]),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def build_scores(raw: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=raw.index)
    scores["Accuracy\n(OOD RMSE)"] = inverse_minmax(raw["ood_rmse_mps"])
    scores["Physical\nClosure"] = inverse_minmax(raw["closure_rmse_mps"])
    scores["Temporal\nSmoothness"] = inverse_minmax(raw["jitter_mean"])
    scores["Real-Time\nEfficiency"] = inverse_minmax(raw["latency_ms"])
    return scores


def plot_radar(ax, scores: pd.DataFrame) -> None:
    labels = list(scores.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.concatenate([angles, [angles[0]]])

    for method in METHOD_ORDER:
        vals = scores.loc[method].to_numpy(dtype=float)
        vals = np.concatenate([vals, [vals[0]]])
        ax.plot(
            closed_angles,
            vals,
            linewidth=2.4 if method == "PIRNN-AKF" else 1.6,
            linestyle=LINESTYLES[method],
            color=COLORS[method],
            label=method,
        )
        ax.fill(
            closed_angles,
            vals,
            color=COLORS[method],
            alpha=0.13 if method == "PIRNN-AKF" else 0.035,
        )

    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_title("Normalized Multi-Metric Comparison on Test-OOD", pad=20)
    ax.grid(True)


def add_raw_metric_table(ax, raw: pd.DataFrame) -> None:
    ax.axis("off")
    table_data = raw.reset_index().copy()
    table_data["ood_rmse_mps"] = table_data["ood_rmse_mps"].map(lambda v: f"{v:.3f}")
    table_data["closure_rmse_mps"] = table_data["closure_rmse_mps"].map(lambda v: f"{v:.3f}")
    table_data["jitter_mean"] = table_data["jitter_mean"].map(lambda v: f"{v:.5f}")
    table_data["latency_ms"] = table_data["latency_ms"].map(lambda v: f"{v:.3f}")
    table_data.columns = ["Method", "OOD RMSE", "Closure RMSE", "Jitter", "Latency"]

    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.8)
    table.scale(1.0, 1.25)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#333333")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(weight="bold")
    ax.set_title("Raw Metrics Used for Normalization")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 9-2: normalized accuracy/physics/smoothness/realtime radar."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="data/figure2/fig2_20260526_122624/figure2_metrics_summary.csv",
    )
    parser.add_argument("--hitl-dir", type=str, default="HITL/logs_in_rasbpi")
    parser.add_argument("--out-dir", type=str, default="data/figure9")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = build_raw_metrics(PROJECT_ROOT / args.summary_csv, PROJECT_ROOT / args.hitl_dir)
    scores = build_scores(raw)
    raw.to_csv(out_dir / "figure9-2_radar_raw_metrics.csv")
    scores.to_csv(out_dir / "figure9-2_radar_scores.csv")

    plt.rcParams.update(STYLE)
    fig, ax_radar = plt.subplots(figsize=(7.2, 6.4), subplot_kw={"polar": True})
    plot_radar(ax_radar, scores)
    ax_radar.tick_params(pad=12)

    handles, labels = ax_radar.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=True,
        edgecolor="black",
        fancybox=False,
    )
    fig.text(
        0.5,
        0.075,
        "Scores are min-max normalized to [0, 1]; larger values indicate better performance.",
        ha="center",
        va="center",
        fontsize=9,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)

    png_path = out_dir / "figure9-2_performance_radar.png"
    svg_path = out_dir / "figure9-2_performance_radar.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 9-2 to {png_path}")
    print(f"Saved Figure 9-2 to {svg_path}")


if __name__ == "__main__":
    main()
