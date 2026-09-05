"""Plot a publication-ready overview of the ten-session HITL campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from paper_plot_style import (  # noqa: E402
    BLUE,
    CYAN,
    GRAY,
    ORANGE,
    RED,
    apply_style,
    format_axes,
    save_figure,
)


WARMUP_S = 20.0
CONTROL_PERIOD_MS = 20.0
REQUIRED_LOOP_HZ = 50.0
MAX_DEADLINE_MISS_PERCENT = 10.0


def _load_sessions(sessions_root: Path) -> pd.DataFrame:
    metrics_path = sessions_root / "campaign_summary/session_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    expected = [
        f"{condition}_{index:02d}"
        for condition in ("id", "ood")
        for index in range(1, 6)
    ]
    by_id = metrics.set_index("session_id")
    missing = sorted(set(expected) - set(by_id.index))
    if missing:
        raise ValueError(f"Missing campaign metrics for sessions: {missing}")

    rows: list[dict[str, float | str]] = []
    for session_id in expected:
        row = by_id.loc[session_id].to_dict()
        validation_path = sessions_root / session_id / "summary/validation.json"
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if validation.get("valid") is not True:
            raise ValueError(f"{session_id} did not pass frozen validation")

        aligned_path = sessions_root / session_id / "aligned/aligned.csv"
        columns = [
            "pi_monotonic_ns",
            "estimated_wind_n_mps",
            "estimated_wind_e_mps",
            "estimated_wind_d_mps",
            "truth_wind_n_mps",
            "truth_wind_e_mps",
            "truth_wind_d_mps",
        ]
        frame = pd.read_csv(aligned_path, usecols=columns)
        monotonic_ns = frame["pi_monotonic_ns"].to_numpy(float)
        keep = monotonic_ns - monotonic_ns[0] >= WARMUP_S * 1e9
        retained = frame.loc[keep]
        estimate = retained[
            [
                "estimated_wind_n_mps",
                "estimated_wind_e_mps",
                "estimated_wind_d_mps",
            ]
        ].to_numpy(float)
        truth = retained[
            ["truth_wind_n_mps", "truth_wind_e_mps", "truth_wind_d_mps"]
        ].to_numpy(float)

        row["session_id"] = session_id
        row["condition"] = session_id.split("_")[0]
        row["truth_horizontal_wind_mps"] = float(
            validation["metrics"]["horizontal_truth_wind_mps"]["median"]
        )
        row["magnitude_ratio"] = float(
            np.mean(np.linalg.norm(estimate, axis=1))
            / np.mean(np.linalg.norm(truth, axis=1))
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _condition_style(condition: str) -> tuple[str, str, str]:
    if condition == "id":
        return BLUE, "^", "HITL ID"
    return RED, "s", "HITL OOD"


def _plot_condition_points(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    *,
    show_band: bool = False,
) -> None:
    for condition in ("id", "ood"):
        group = data.loc[data["condition"] == condition]
        color, marker, _ = _condition_style(condition)
        x = group["truth_horizontal_wind_mps"].to_numpy(float)
        y = group[metric].to_numpy(float)
        if show_band:
            mean = float(np.mean(y))
            sd = float(np.std(y, ddof=1))
            left, right = float(np.min(x) - 0.18), float(np.max(x) + 0.18)
            ax.fill_between(
                [left, right],
                mean - sd,
                mean + sd,
                color=color,
                alpha=0.10,
                linewidth=0,
                zorder=0,
            )
            ax.hlines(
                mean,
                left,
                right,
                color=color,
                linestyle=(0, (4, 2)),
                linewidth=1.15,
                zorder=1,
            )
        ax.scatter(
            x,
            y,
            s=35,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=4,
        )


def _panel_title(ax: plt.Axes, label: str, title: str) -> None:
    ax.set_title(f"({label}) {title}", loc="left", fontweight="bold", pad=5)


def _annotate_threshold(
    ax: plt.Axes,
    y: float,
    label: str,
    *,
    color: str = ORANGE,
) -> None:
    ax.axhline(y, color=color, linestyle=(0, (5, 2)), linewidth=1.15, zorder=1)
    ax.annotate(
        label,
        xy=(0.985, y),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=7,
        color=color,
        fontweight="bold",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sessions-root",
        default="HITL/sessions",
        help="Directory containing the ten canonical HITL sessions.",
    )
    parser.add_argument(
        "--output",
        default="HITL/sessions/campaign_summary/figureF1_hitl_session_overview",
        help="Output path without extension.",
    )
    args = parser.parse_args()

    sessions_root = Path(args.sessions_root)
    if not sessions_root.is_absolute():
        sessions_root = PROJECT_ROOT / sessions_root
    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_ROOT / output

    data = _load_sessions(sessions_root)
    source_path = output.with_name(f"{output.name}_source_data.csv")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(source_path, index=False)

    apply_style()
    plt.rcParams.update(
        {
            "axes.titlesize": 9.3,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.60, 5.15), sharex=True)
    ax_rmse, ax_direction, ax_ratio = axes[0]
    ax_latency, ax_rate, ax_miss = axes[1]
    wind = data["truth_horizontal_wind_mps"].to_numpy(float)

    _plot_condition_points(ax_rmse, data, "rmse_3d_mps", show_band=True)
    _panel_title(ax_rmse, "a", "3D wind error")
    ax_rmse.set_ylabel("3D RMSE (m/s)")
    ax_rmse.set_ylim(0.35, 1.38)

    _plot_condition_points(
        ax_direction, data, "direction_mae_deg", show_band=True
    )
    _panel_title(ax_direction, "b", "Direction error")
    ax_direction.set_ylabel("Direction MAE (deg)")
    ax_direction.set_ylim(0, 30)

    _plot_condition_points(ax_ratio, data, "magnitude_ratio", show_band=True)
    _annotate_threshold(ax_ratio, 1.0, "Ideal ratio = 1")
    _panel_title(ax_ratio, "c", "Magnitude fidelity")
    ax_ratio.set_ylabel("Estimate / truth magnitude")
    ax_ratio.set_ylim(0.62, 1.25)

    inference_p95 = data["inference_latency_p95_ms"].to_numpy(float)
    companion_p95 = data["companion_latency_p95_ms"].to_numpy(float)
    ax_latency.vlines(
        wind,
        inference_p95,
        companion_p95,
        color="0.72",
        linewidth=1.15,
        zorder=2,
    )
    ax_latency.scatter(
        wind,
        inference_p95,
        s=30,
        marker="o",
        color=GRAY,
        edgecolor="black",
        linewidth=0.45,
        label=r"Inference $p_{95}$",
        zorder=3,
    )
    ax_latency.scatter(
        wind,
        companion_p95,
        s=34,
        marker="D",
        color=CYAN,
        edgecolor="black",
        linewidth=0.45,
        label=r"Companion $p_{95}$",
        zorder=3,
    )
    ax_latency.axhline(
        CONTROL_PERIOD_MS,
        color=ORANGE,
        linestyle=(0, (5, 2)),
        linewidth=1.15,
        label="20 ms budget",
        zorder=1,
    )
    _panel_title(ax_latency, "d", "Processing latency")
    ax_latency.set_ylabel(r"Latency $p_{95}$ (ms)")
    ax_latency.set_ylim(10, 23)
    handles, labels = ax_latency.get_legend_handles_labels()
    legend_order = (0, 2, 1)
    ax_latency.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=6.5,
        handlelength=1.1,
        columnspacing=0.65,
        handletextpad=0.35,
        borderaxespad=0.25,
    )

    _plot_condition_points(ax_rate, data, "loop_hz")
    for condition in ("id", "ood"):
        group = data.loc[data["condition"] == condition]
        color, _, _ = _condition_style(condition)
        ax_rate.vlines(
            group["truth_horizontal_wind_mps"],
            REQUIRED_LOOP_HZ,
            group["loop_hz"],
            color=color,
            linewidth=1.0,
            alpha=0.55,
            zorder=2,
        )
    _annotate_threshold(
        ax_rate, REQUIRED_LOOP_HZ, "50 Hz requirement", color=ORANGE
    )
    _panel_title(ax_rate, "e", "Estimator update rate")
    ax_rate.set_ylabel("Achieved loop rate (Hz)")
    ax_rate.set_ylim(45, 90)

    data["deadline_miss_percent"] = 100.0 * data["deadline_miss_ratio"]
    _plot_condition_points(ax_miss, data, "deadline_miss_percent")
    for condition in ("id", "ood"):
        group = data.loc[data["condition"] == condition]
        color, _, _ = _condition_style(condition)
        ax_miss.vlines(
            group["truth_horizontal_wind_mps"],
            0,
            group["deadline_miss_percent"],
            color=color,
            linewidth=1.0,
            alpha=0.55,
            zorder=2,
        )
    _annotate_threshold(
        ax_miss,
        MAX_DEADLINE_MISS_PERCENT,
        "10% acceptance limit",
        color=ORANGE,
    )
    _panel_title(ax_miss, "f", "Deadline misses")
    ax_miss.set_ylabel("Deadline-miss ratio (%)")
    ax_miss.set_ylim(0, 11.5)

    for ax in axes.ravel():
        ax.set_xlim(1.0, 7.8)
        ax.set_xticks(np.arange(1, 8, 1))
        format_axes(ax, grid_axis="y")

    condition_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markersize=6,
            markerfacecolor=BLUE,
            markeredgecolor="white",
            label="HITL ID (5 sessions)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=6,
            markerfacecolor=RED,
            markeredgecolor="white",
            label="HITL OOD (5 sessions)",
        ),
        Patch(
            facecolor=GRAY,
            alpha=0.14,
            edgecolor="none",
            label=r"Condition mean $\pm$ SD (a--c)",
        ),
    ]
    fig.legend(
        handles=condition_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.982),
        ncol=3,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.supxlabel("Median horizontal truth wind (m/s)", y=0.045, fontsize=9.5)
    fig.text(
        0.5,
        0.012,
        "All 10 sessions passed frozen validation; each session lasted 350 s "
        "with the first 20 s excluded.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="0.30",
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.992,
        bottom=0.115,
        top=0.895,
        wspace=0.34,
        hspace=0.31,
    )

    paper_name = "figureF1_hitl_session_overview"
    save_figure(fig, output, copy_to_paper=paper_name)
    save_figure(
        fig,
        PROJECT_ROOT / "Paper_2/MDPI_template_APA/figures" / paper_name,
    )
    plt.close(fig)
    print(f"Saved figure: {output}.png/.pdf/.svg")
    print(f"Saved source data: {source_path}")


if __name__ == "__main__":
    main()
