"""Assemble the main-text Figure 5 from the typical anomaly-response panel.

Dynamic-R ablation is kept as an appendix figure to avoid overloading the main
time-series figure.
"""
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from paper_plot_style import apply_style, save_figure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = PROJECT_ROOT / "data/figure5/figure5_anomaly_composite"

PANELS = [
    ("GPS spike response", PROJECT_ROOT / "data/figure5/figure5-2_time_series.png"),
]
FIG_WIDTH_IN = 7.0   # column/page width target
LABEL_GAP_IN = 0.20  # vertical room above each panel for its descriptive label


def main():
    apply_style()

    imgs, rel_h = [], []
    for label, path in PANELS:
        if not path.exists():
            raise FileNotFoundError(path)
        im = mpimg.imread(path)
        imgs.append(im)
        h, w = im.shape[0], im.shape[1]
        rel_h.append(h / w)  # height when width normalized to 1

    panel_heights_in = [FIG_WIDTH_IN * r for r in rel_h]
    total_h = sum(panel_heights_in) + LABEL_GAP_IN * len(PANELS)

    fig = plt.figure(figsize=(FIG_WIDTH_IN, total_h))
    height_ratios = []
    for ph in panel_heights_in:
        height_ratios.append(LABEL_GAP_IN)  # label strip
        height_ratios.append(ph)            # image
    gs = GridSpec(len(PANELS) * 2, 1, height_ratios=height_ratios,
                  hspace=0.04, left=0.02, right=0.98, top=0.995, bottom=0.005)

    for i, ((label, _), im) in enumerate(zip(PANELS, imgs)):
        lab_ax = fig.add_subplot(gs[i * 2, 0])
        lab_ax.axis("off")
        lab_ax.text(
            0.5,
            0.02,
            label,
            fontsize=10,
            fontweight="normal",
            ha="center",
            va="bottom",
            transform=lab_ax.transAxes,
        )
        img_ax = fig.add_subplot(gs[i * 2 + 1, 0])
        img_ax.imshow(im, aspect="auto", interpolation="lanczos")
        img_ax.axis("off")

    save_figure(fig, OUT_BASE, copy_to_paper="figure5")
    print(f"saved -> {OUT_BASE}.png / .pdf / .svg  (figure size {FIG_WIDTH_IN:.1f} x {total_h:.1f} in) and copied to paper/figures")


if __name__ == "__main__":
    main()
