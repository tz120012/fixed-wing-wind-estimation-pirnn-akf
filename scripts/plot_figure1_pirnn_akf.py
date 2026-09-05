#!/usr/bin/env python3
"""Publication schematic for the PIRNN-AKF signal flow."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "figure1" / "python"
PAPER_DIRS = (
    ROOT / "Paper_2" / "figures",
    ROOT / "Paper_2" / "MDPI_template_APA" / "figures",
)

C = {
    "ink": "#263238", "grey": "#607D8B", "panel": "#F8FAFB",
    "panel_edge": "#D4DDE2", "sensor": "#EEF3F6", "sensor_edge": "#78909C",
    "net": "#E8F1FA", "net_edge": "#3775BA", "akf": "#E7F5F1",
    "akf_edge": "#2A8C78", "train": "#FCEDEA", "train_edge": "#C96A5A",
    "blue": "#0F4D92", "white": "#FFFFFF",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 6.2,
})


def box(ax, x, y, w, h, text, fc, ec, fs=5.6, bold=False, ls="-", z=3):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.009",
        facecolor=fc, edgecolor=ec, linewidth=0.9, linestyle=ls, zorder=z,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
        color=C["ink"], fontweight="bold" if bold else "normal",
        linespacing=1.15, zorder=z + 1,
    )


def panel(ax, x, y, w, h, n, title, subtitle):
    box(ax, x, y, w, h, "", C["panel"], C["panel_edge"], z=0)
    ax.text(x + .010, y + h - .025, n, fontsize=8, fontweight="bold",
            color=C["blue"], ha="left", va="top")
    ax.text(x + .038, y + h - .024, title, fontsize=7.2, fontweight="bold",
            color=C["ink"], ha="left", va="top")
    ax.text(x + .010, y + h - .057, subtitle, fontsize=5.1, color=C["grey"],
            ha="left", va="top")


def flow(ax, start, end, label, color, fs=4.7, rad=0, offset=(0, 0), ls="-", lw=1):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=7, linewidth=lw,
        color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        shrinkA=1.5, shrinkB=1.5, zorder=5,
    ))
    if label:
        ax.text(
            (start[0] + end[0]) / 2 + offset[0],
            (start[1] + end[1]) / 2 + offset[1],
            label, ha="center", va="center", fontsize=fs, color=color,
            bbox=dict(fc="white", ec="none", pad=.35, alpha=.94), zorder=6,
        )


def elbow(ax, points, label, color, fs=4.6, label_xy=None, ls="-", lw=1):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=4)
    flow(ax, points[-2], points[-1], "", color, ls=ls, lw=lw)
    if label:
        x, y = label_xy or (
            (points[0][0] + points[1][0]) / 2,
            (points[0][1] + points[1][1]) / 2,
        )
        ax.text(x, y, label, ha="center", va="center", fontsize=fs, color=color,
                bbox=dict(fc="white", ec="none", pad=.35, alpha=.94), zorder=6)


def build():
    fig, ax = plt.subplots(figsize=(7.2, 5.30))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    panel(ax, .015, .33, .185, .64, "1", "Onboard signals",
          "navigation • air-data • control")
    panel(ax, .215, .33, .310, .64, "2", "Physics-informed GRU",
          "temporal inference • physics-informed training")
    panel(ax, .540, .33, .445, .64, "3", "Adaptive Kalman fusion",
          "sequential updates • same-step output fusion")

    sensors = [
        (.030, .790, "GPS\n" + r"$\mathbf{V}_{g,k}$", r"$\mathbf{V}_{g,k}$"),
        (.030, .695, "Pitot tube\n" + r"$V_{\mathrm{TAS},k}$", r"$V_{\mathrm{TAS},k}$"),
        (.030, .600, "IMU / attitude\n" + r"$\phi,\theta,\psi,\mathbf{\Omega},\mathbf{a}$",
         r"$\phi,\theta,\psi,\mathbf{\Omega},\mathbf{a}$"),
        (.030, .505, "Control\n" + r"$\mathbf{u}_k,u_{\mathrm{thr},k}$",
         r"$\mathbf{u}_k,u_{\mathrm{thr},k}$"),
    ]
    for x, y, text, _ in sensors:
        box(ax, x, y, .112, .058, text, C["sensor"], C["sensor_edge"], fs=5.2)
    busx = .165
    ax.plot([busx, busx], [.525, .819], color=C["sensor_edge"], lw=1)
    for x, y, _, label in sensors:
        flow(ax, (x + .112, y + .029), (busx, y + .029), label,
             C["sensor_edge"], fs=4.4, offset=(0, .014))

    box(ax, .230, .635, .095, .095, "Sliding window\n" + r"$\mathbf{X}_{k-T+1:k}$"
        + "\n41-D, 50 Hz", C["net"], C["net_edge"], fs=4.8, bold=True)
    box(ax, .350, .635, .085, .095, "2-layer GRU\n" + r"$\mathbf{h}_k$"
        + "\ntemporal state", C["net"], C["net_edge"], fs=4.8, bold=True)
    elbow(ax, [(busx, .525), (busx, .470), (.212, .470), (.230, .682)],
          r"normalized $\mathbf{X}_{k-T+1:k}$", C["net_edge"], label_xy=(.200, .490))
    flow(ax, (.325, .682), (.350, .682), r"$\mathbf{X}_{k-T+1:k}$",
         C["net_edge"], offset=(0, .016))

    box(ax, .458, .570, .055, .280,
        "Multi-task\nheads\n\n"
        + r"$\mathbf{w}_{\mathrm{NN},k}$" + "\n"
        + r"$\mathbf{q}/\mathbf{r}_{\mathrm{scale},k}$" + "\n"
        + r"$\Delta\alpha,\Delta\beta,s_{\mathrm{TAS}}$" + "\n"
        + r"$s_k$",
        C["white"], C["net_edge"], fs=4.5, bold=True)
    flow(ax, (.435, .682), (.458, .682), r"$\mathbf{h}_k$",
         C["net_edge"], fs=4.3, offset=(0, .015))

    # AKF modules.
    box(ax, .558, .775, .115, .090, "Kinematic measurement\n"
        + r"$\mathbf{w}_{\mathrm{kin},k}\rightarrow\mathbf{w}^{\mathrm{stab}}_{\mathrm{kin},k}$"
        + "\n" + r"$\rho_k,m_k$",
        C["akf"], C["akf_edge"], fs=4.0, bold=True)
    box(ax, .704, .775, .118, .090, "Adaptive covariance\n"
        + r"$\mathbf{Q}^{\mathrm{eff}}_k$" + "\n"
        + r"$\mathbf{R}_{\mathrm{kin},k},\mathbf{R}_{\mathrm{NN},k}$",
        C["akf"], C["akf_edge"], fs=4.1, bold=True)
    box(ax, .558, .540, .115, .088, "State prediction\n"
        + r"$\hat{\mathbf{x}}^-_k,\mathbf{P}^-_k$" + "\nlimited NN increment",
        C["akf"], C["akf_edge"], fs=4.6, bold=True)
    box(ax, .704, .540, .118, .088, "Sequential updates\n1  "
        + r"$\mathbf{w}^{\mathrm{stab}}_{\mathrm{kin},k}$" + "\n2  "
        + r"$\mathbf{w}_{\mathrm{NN},k}$",
        C["akf"], C["akf_edge"], fs=4.6, bold=True)
    box(ax, .850, .540, .098, .088, "Same-step fusion\n"
        + r"$\omega_k\hat{\mathbf{x}}_{k|k}$" + "\n"
        + r"$+(1-\omega_k)\mathbf{w}_{\mathrm{NN},k}$",
        C["akf"], C["akf_edge"], fs=4.4, bold=True)

    # Labelled PI-GRU outputs and reused raw channels.
    flow(ax, (.513, .805), (.558, .805), r"$\mathbf{w}_{\mathrm{NN},k}$",
         C["net_edge"], fs=4.3, offset=(0, .014))
    elbow(ax, [(.513, .745), (.682, .745), (.682, .820), (.704, .820)],
          r"$\mathbf{q}_{\mathrm{scale},k},\mathbf{r}_{\mathrm{scale},k}$",
          C["net_edge"], fs=4.1, label_xy=(.600, .756))
    elbow(ax, [(.513, .685), (.540, .685), (.540, .850), (.558, .850)],
          r"$\Delta\alpha,\Delta\beta,s_{\mathrm{TAS}}$",
          C["net_edge"], fs=4.1, label_xy=(.540, .760))
    elbow(ax, [(.513, .625), (.687, .625), (.687, .790), (.704, .790)],
          r"$s_k$", C["net_edge"], fs=4.2, label_xy=(.610, .638))
    elbow(ax, [(busx, .819), (busx, .885), (.615, .885), (.615, .865)],
          r"$\mathbf{V}_{g,k},V_{\mathrm{TAS},k},\phi_k,\theta_k,\psi_k;"
          r"\ m_k=f(\mathbf{\Omega}_k,\mathbf{a}_k,\mathbf{u}_k,u_{\mathrm{thr},k})$",
          C["sensor_edge"], fs=4.1, label_xy=(.425, .874))

    # Ordered AKF signal flow.
    flow(ax, (.673, .820), (.704, .820), r"$\rho_k,d_k,m_k,c_{\mathrm{out}}$",
         C["akf_edge"], fs=4.0, offset=(0, -.017))
    elbow(ax, [(.513, .805), (.530, .805), (.530, .584), (.558, .584)],
          r"$\Delta\mathbf{w}_{\mathrm{NN},k}$", C["net_edge"],
          label_xy=(.530, .690))
    elbow(ax, [(.763, .775), (.763, .690), (.615, .690), (.615, .628)],
          r"$\mathbf{Q}^{\mathrm{eff}}_k$", C["akf_edge"], label_xy=(.690, .703))
    flow(ax, (.763, .775), (.763, .628),
         r"$\mathbf{R}_{\mathrm{kin},k},\mathbf{R}_{\mathrm{NN},k}$",
         C["akf_edge"], fs=4.2, offset=(.027, 0))
    flow(ax, (.673, .584), (.704, .584), r"$\hat{\mathbf{x}}^-_k,\mathbf{P}^-_k$",
         C["akf_edge"], fs=4.3, offset=(0, .016))
    elbow(ax, [(.615, .775), (.615, .710), (.735, .710), (.735, .628)],
          r"$\mathbf{z}_{\mathrm{kin},k}=\mathbf{w}^{\mathrm{stab}}_{\mathrm{kin},k}$",
          C["akf_edge"], fs=4.2, label_xy=(.675, .722))
    elbow(ax, [(.513, .805), (.520, .805), (.520, .505), (.763, .505), (.763, .540)],
          r"$\mathbf{z}_{\mathrm{NN},k}=\mathbf{w}_{\mathrm{NN},k}$",
          C["net_edge"], fs=4.2, label_xy=(.640, .493))
    flow(ax, (.822, .584), (.850, .584), r"$\hat{\mathbf{x}}_{k|k},\mathbf{P}_{k|k}$",
         C["akf_edge"], fs=4.2, offset=(0, .016))
    elbow(ax, [(.513, .625), (.525, .625), (.525, .470), (.899, .470), (.899, .540)],
          r"$s_k,\rho_k,p_k\ \rightarrow\ \omega_k$", C["net_edge"],
          fs=4.3, label_xy=(.710, .458))

    out = Circle((.970, .584), .018, fc=C["blue"], ec=C["blue"], zorder=5)
    ax.add_patch(out)
    flow(ax, (.948, .584), (.952, .584), "", C["akf_edge"])
    ax.text(.970, .584, r"$\hat{\mathbf{w}}_k$", color="white", fontsize=5.2,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(.970, .552, "local 3-D wind", color=C["blue"], fontsize=4.6,
            ha="center", va="top")

    # Training-only branch: one compact objective block prevents a dashboard-like layout.
    box(ax, .215, .045, .470, .230, "", C["train"], C["train_edge"], ls="--", z=0)
    ax.text(.228, .252, "TRAINING ONLY", fontsize=6.0, fontweight="bold",
            color=C["train_edge"], ha="left", va="top")
    box(ax, .255, .095, .390, .100,
        "Physics-informed objective\n"
        + r"$\mathcal{L}_{\mathrm{total}}="
        + r"\lambda_{\mathrm{wind}}\mathcal{L}_{\mathrm{data}}+"
        + r"\lambda_{\mathrm{physics}}\mathcal{L}_{\mathrm{physics}}+"
        + r"\lambda_{\mathrm{mag}}\mathcal{L}_{\mathrm{mag}}+"
        + r"\lambda_{\mathrm{dir}}\mathcal{L}_{\mathrm{dir}}$"
        + "\n"
        + r"$+\lambda_{\mathrm{collapse}}\mathcal{L}_{\mathrm{collapse}}"
        + r"+\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}}$",
        C["white"], C["train_edge"], fs=4.4, bold=True)
    elbow(ax, [(busx, .525), (.205, .525), (.205, .220), (.320, .220), (.320, .195)],
          r"$\mathbf{V}_{g},V_{\mathrm{TAS}},\phi,\theta,\psi$",
          C["train_edge"], fs=4.0, label_xy=(.270, .220), ls="--")
    elbow(ax, [(.485, .570), (.485, .220), (.450, .220), (.450, .195)],
          r"$\mathbf{w}_{\mathrm{NN}},\Delta\alpha,\Delta\beta,s_{\mathrm{TAS}}$",
          C["train_edge"], fs=3.9, label_xy=(.450, .208), ls="--")
    elbow(ax, [(.645, .145), (.670, .145), (.670, .295), (.392, .295), (.392, .635)],
          r"$\nabla_{\theta}\mathcal{L}_{\mathrm{total}}$"
          + "\nfreeze → unfreeze", C["train_edge"],
          fs=4.2, label_xy=(.535, .295), ls="--", lw=1.1)
    ax.text(.590, .220, r"$\mathbf{W}^{\mathrm{true}}$ + weak-wind mask",
            ha="center", va="center", fontsize=4.0, color=C["train_edge"],
            bbox=dict(fc="white", ec="none", pad=.35, alpha=.94), zorder=6)
    flow(ax, (.590, .218), (.590, .195), "", C["train_edge"], ls="--")

    # Small visual key.
    ax.plot([.715, .770], [.085, .085], color=C["net_edge"], lw=1.2)
    ax.text(.775, .085, "online inference", fontsize=4.8, color=C["net_edge"], va="center")
    ax.plot([.855, .910], [.085, .085], color=C["train_edge"], lw=1.2, ls="--")
    ax.text(.915, .085, "training only", fontsize=4.8, color=C["train_edge"], va="center")

    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for directory in PAPER_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
    fig = build()
    bases = [OUT / "figure1", *(directory / "figure1" for directory in PAPER_DIRS)]
    for base in bases:
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=.03)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.03)
        fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=.03)
        fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
