from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DPI = 600
BLUE = "#1F77B4"
RED = "#E64B35"
GREEN = "#00A087"
CYAN = "#4DBBD5"
GRAY = "#7F7F7F"
ORANGE = "#D55E00"


def apply_style() -> None:
    """Apply a consistent MDPI/Drones-like academic figure style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "patch.linewidth": 0.8,
        "grid.alpha": 0.30,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def format_axes(axes: Iterable[plt.Axes] | plt.Axes, grid_axis: str = "both") -> None:
    """Use inward ticks and a full four-sided box for all axes."""
    if isinstance(axes, np.ndarray):
        axes = axes.ravel().tolist()
    elif not isinstance(axes, (list, tuple)):
        axes = [axes]
    for ax in axes:
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        if grid_axis:
            ax.grid(True, axis=grid_axis)


def save_figure(fig: plt.Figure, base: Path, *, copy_to_paper: str | None = None) -> None:
    """Save PNG, PDF, and SVG consistently, optionally mirroring to paper/figures."""
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if ext in {"png", "pdf"}:
            kwargs["dpi"] = DPI
        fig.savefig(base.with_suffix(f".{ext}"), **kwargs)

    if copy_to_paper:
        paper_base = PROJECT_ROOT / "Paper_2/figures" / copy_to_paper
        paper_base.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf", "svg"):
            kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
            if ext in {"png", "pdf"}:
                kwargs["dpi"] = DPI
            fig.savefig(paper_base.with_suffix(f".{ext}"), **kwargs)
