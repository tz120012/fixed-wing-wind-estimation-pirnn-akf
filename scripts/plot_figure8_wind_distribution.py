import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import paper_evidence_chain_eval as evidence


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

COLORS = {
    "Train": "#7F7F7F",
    "Test-ID": "#1F77B4",
    "Test-OOD": "#E64B35",
}


def load_wind(data_dir: Path, split: str, scaler_y) -> np.ndarray:
    y_path = data_dir / f"y_{split}.npy"
    if not y_path.exists():
        raise FileNotFoundError(f"Missing label file: {y_path}")
    y = np.load(y_path)
    return evidence.denorm_y(y, scaler_y)[:, :3]


def sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if n <= max_points:
        return np.arange(n)
    return rng.choice(n, size=max_points, replace=False)


def apply_clean_axes(ax) -> None:
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Figure 8: wind-field distribution across dataset splits."
    )
    parser.add_argument("--data-dir", type=str, default="data/dataset_new_processed")
    parser.add_argument("--out-dir", type=str, default="data/figure8")
    parser.add_argument("--max-points", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    _, scaler_y = evidence.load_norm_params(data_dir)
    wind = {
        "Train": load_wind(data_dir, "train", scaler_y),
        "Test-ID": load_wind(data_dir, "test_id", scaler_y),
        "Test-OOD": load_wind(data_dir, "test_ood", scaler_y),
    }

    mag = {name: np.linalg.norm(arr[:, :2], axis=1) for name, arr in wind.items()}

    plt.rcParams.update(STYLE)
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2))

    # (a) Horizontal wind-vector coverage. Keep this panel sparse and clean.
    for name, arr in wind.items():
        point_budget = args.max_points if name != "Train" else args.max_points // 2
        idx = sample_indices(len(arr), point_budget, args.seed)
        zorder = {"Train": 1, "Test-ID": 2, "Test-OOD": 3}[name]
        alpha = {"Train": 0.12, "Test-ID": 0.30, "Test-OOD": 0.32}[name]
        size = {"Train": 4, "Test-ID": 7, "Test-OOD": 7}[name]
        axs[0].scatter(
            arr[idx, 0],
            arr[idx, 1],
            s=size,
            alpha=alpha,
            color=COLORS[name],
            edgecolor="none",
            label=name,
            zorder=zorder,
        )
    axs[0].set_title("(a) Horizontal Wind Coverage")
    axs[0].set_xlabel("North Wind (m/s)")
    axs[0].set_ylabel("East Wind (m/s)")
    axs[0].axis("equal")
    axs[0].legend(loc="upper right", frameon=True, edgecolor="black", fancybox=False)
    apply_clean_axes(axs[0])

    # (b) Horizontal wind speed CDF, highlighting the OOD shift without noisy histograms.
    for name, values in mag.items():
        x, y = empirical_cdf(values)
        axs[1].plot(
            x,
            y,
            linewidth=2.0,
            color=COLORS[name],
            linestyle="-" if name != "Train" else "--",
            label=name,
        )
    axs[1].axvspan(1.0, 3.2, color="#1F77B4", alpha=0.08)
    axs[1].axvspan(3.5, 5.0, color="#E64B35", alpha=0.08)
    axs[1].set_title("(b) Horizontal Wind Speed CDF")
    axs[1].set_xlabel("Horizontal Wind Speed (m/s)")
    axs[1].set_ylabel("Cumulative Probability")
    axs[1].set_xlim(0, 8.2)
    axs[1].set_ylim(0, 1.0)
    apply_clean_axes(axs[1])

    # (c) Compact distribution summary using violin + quartile markers.
    order = ["Train", "Test-ID", "Test-OOD"]
    values = [mag[name] for name in order]
    parts = axs[2].violinplot(
        values,
        positions=np.arange(1, len(order) + 1),
        widths=0.75,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, name in zip(parts["bodies"], order):
        body.set_facecolor(COLORS[name])
        body.set_edgecolor("black")
        body.set_alpha(0.30)
        body.set_linewidth(1.0)

    for pos, name in enumerate(order, start=1):
        q1, median, q3 = np.percentile(mag[name], [25, 50, 75])
        axs[2].plot([pos - 0.18, pos + 0.18], [median, median], color="black", linewidth=2.0)
        axs[2].plot([pos, pos], [q1, q3], color="black", linewidth=1.6)
        axs[2].scatter(pos, np.mean(mag[name]), s=26, color=COLORS[name], edgecolor="black", zorder=3)

    axs[2].set_xticks(np.arange(1, len(order) + 1))
    axs[2].set_xticklabels(order, rotation=15)
    axs[2].set_title("(c) Wind Speed Summary")
    axs[2].set_xlabel("")
    axs[2].set_ylabel("Horizontal Wind Speed (m/s)")
    axs[2].set_ylim(0, 8.2)
    apply_clean_axes(axs[2])

    fig.tight_layout()
    png_path = out_dir / "figure8_wind_distribution.png"
    svg_path = out_dir / "figure8_wind_distribution.svg"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved Figure 8 to {png_path}")


if __name__ == "__main__":
    main()
