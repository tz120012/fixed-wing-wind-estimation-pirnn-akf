from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "pictures"
OUT_DIR.mkdir(exist_ok=True)


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 180,
    "savefig.dpi": 300,
})


COLORS = {
    "input": "#E8F1FA",
    "backbone": "#DFF2E1",
    "head": "#FFF3CD",
    "physics": "#F8D7DA",
    "akf": "#EADCF8",
    "output": "#D1ECF1",
    "line": "#3A3A3A",
}


def add_box(ax, xy, w, h, text, fc, ec="#555555", fontsize=10, lw=1.2):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
        zorder=3,
    )
    return box


def arrow(ax, start, end, text=None, rad=0.0, color=None, lw=1.4, style="-|>"):
    color = color or COLORS["line"]
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(arr)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.05, text, ha="center", va="bottom", fontsize=8, color=color)
    return arr


def save(fig, name):
    png = OUT_DIR / f"{name}.png"
    svg = OUT_DIR / f"{name}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {svg}")


def figure_1_architecture():
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis("off")

    ax.text(
        7.5,
        7.65,
        "Figure 1. PI-GRU and PIRNN-AKF Overall Architecture",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    # Main pipeline
    add_box(
        ax,
        (0.45, 3.7),
        2.2,
        1.1,
        "Sliding window input\nX[t-T+1:t]\n41 flight features",
        COLORS["input"],
    )
    add_box(
        ax,
        (3.1, 3.7),
        2.0,
        1.1,
        "GRU backbone\n2 layers\nhidden state h_t",
        COLORS["backbone"],
    )
    add_box(
        ax,
        (5.65, 3.7),
        1.8,
        1.1,
        "Shared latent\nfeature z_t",
        COLORS["backbone"],
    )

    # Heads
    head_x = 8.15
    head_w = 2.2
    heads = [
        ((head_x, 5.95), "Wind head\nw_NN = [N,E,D]", COLORS["head"]),
        ((head_x, 4.65), "Q-scale head\nq_scale [N,E,D]", COLORS["head"]),
        ((head_x, 3.35), "R-scale head\nr_scale [GPS,TAS,ATT]", COLORS["head"]),
        ((head_x, 2.05), "Angle/TAS head\n[Delta alpha,\n Delta beta, s_TAS]", COLORS["head"]),
        ((head_x, 0.75), "Confidence head\ns_k in [0,1]", COLORS["head"]),
    ]
    for xy, txt, fc in heads:
        add_box(ax, xy, head_w, 0.9, txt, fc, fontsize=9)

    # Loss and AKF modules
    add_box(
        ax,
        (11.15, 5.6),
        2.8,
        1.2,
        "Training losses\nData loss + Physics loss\n+ weak-wind regularizers",
        COLORS["physics"],
        fontsize=9,
    )
    add_box(
        ax,
        (11.15, 2.35),
        2.8,
        1.6,
        "Adaptive Kalman Filter\nstate prediction\nQ/R re-scaling\nposterior fusion",
        COLORS["akf"],
        fontsize=9,
    )
    add_box(
        ax,
        (11.45, 0.55),
        2.2,
        0.9,
        "Final wind output\nw_AKF",
        COLORS["output"],
        fontsize=10,
    )

    # Arrows
    arrow(ax, (2.65, 4.25), (3.1, 4.25))
    arrow(ax, (5.1, 4.25), (5.65, 4.25))
    for y in [6.4, 5.1, 3.8, 2.5, 1.2]:
        arrow(ax, (7.45, 4.25), (8.15, y), rad=0.08)

    # Ground truth labels (Training only)
    add_box(
        ax,
        (0.2, 6.7),
        2.2,
        0.7,
        "Ground truth label\nW_true (Training only)",
        "#F2F2F2",
        fontsize=8.5,
    )
    arrow(ax, (2.4, 7.05), (11.15, 6.6), text="supervised data loss", rad=-0.05, color="#8A2D2D")

    # Explicit state slicing from the 45-dim input
    add_box(
        ax,
        (3.5, 6.4),
        2.8,
        0.9,
        "Kinematic state slice X_t\n(V_g, TAS, attitude)\nextracted from 45 dims",
        COLORS["input"],
        fontsize=8.5,
    )
    arrow(ax, (2.0, 4.8), (3.5, 6.85), rad=0.0, color="#444444")
    ax.text(2.65, 5.9, "slice", ha="center", va="center", fontsize=8, color="#444444", rotation=50)
    
    # Slice to Loss
    arrow(ax, (6.3, 6.85), (11.15, 6.3), rad=-0.02, color="#8A2D2D")
    ax.text(7.2, 6.85, "physics residual", ha="center", va="center", fontsize=8, color="#8A2D2D", rotation=-6)
    
    # Slice to AKF
    arrow(ax, (6.3, 6.6), (11.15, 3.8), rad=0.02, color="#5B3B8A")
    ax.text(7.2, 6.18, "kinematic states", ha="center", va="center", fontsize=8, color="#5B3B8A", rotation=-30)

    # Arrows from network heads to Loss
    arrow(ax, (10.35, 6.4), (11.15, 6.25), color="#8A2D2D") # from Wind head
    arrow(ax, (10.35, 2.5), (11.15, 5.85), color="#8A2D2D", rad=-0.12) # from Angle/TAS head

    # Indicate Gradient Backpropagation
    arrow(ax, (12.55, 6.2), (12.55, 7.3), style="<|-", lw=2, color="#D9534F")
    arrow(ax, (12.55, 7.3), (4.1, 7.3), style="-", lw=2, color="#D9534F")
    arrow(ax, (4.1, 7.3), (4.1, 4.25), style="-|>", lw=2, color="#D9534F")
    ax.text(8.0, 7.4, "Gradient Backpropagation (updates network weights)", ha="center", va="bottom", fontsize=9, color="#D9534F", fontweight="bold")

    # AKF inputs
    arrow(ax, (10.35, 6.4), (11.15, 3.55), text="pseudo-measurement", rad=-0.12, color="#5B3B8A")
    arrow(ax, (10.35, 5.1), (11.15, 3.25), color="#5B3B8A", rad=-0.05)
    arrow(ax, (10.35, 3.8), (11.15, 3.0), color="#5B3B8A", rad=0.02)
    arrow(ax, (10.35, 1.2), (11.15, 2.6), color="#5B3B8A", rad=0.08)
    arrow(ax, (12.55, 2.35), (12.55, 1.45), color="#006C7A")

    # Notes
    ax.text(
        1.55,
        3.1,
        "Features include NED velocity,\nbody velocity, IMU, attitude,\nangular rates, controls and TAS.",
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
    )
    ax.text(
        12.55,
        4.25,
        "Inference stage uses wind,\nq/r scales and confidence\nfor online smoothing.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )

    save(fig, "Figure_1_PIGRU_PIRNN_AKF_architecture")


def figure_2_physics_loss():
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(
        7.5,
        8.05,
        "Figure 2. Velocity-Triangle Physics Loss Computation Path",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    # Top sensor/model inputs
    add_box(ax, (0.55, 6.55), 2.15, 0.85, "GPS ground velocity\nV_g", COLORS["input"])
    add_box(ax, (3.05, 6.55), 2.15, 0.85, "Network wind\nw_NN", COLORS["head"])
    add_box(ax, (5.55, 6.55), 2.15, 0.85, "Attitude matrix\nR_b->n", COLORS["input"])
    add_box(ax, (8.05, 6.55), 2.15, 0.85, "Pitot airspeed\nTAS_meas", COLORS["input"])
    add_box(ax, (10.55, 6.55), 2.15, 0.85, "Angle/TAS outputs\nDelta alpha,\nDelta beta, s_TAS", COLORS["head"], fontsize=8.5)

    # Main computation path
    add_box(
        ax,
        (1.35, 4.9),
        2.9,
        0.9,
        "NED air-relative velocity\nV_a,n^base = V_g - w_NN",
        COLORS["physics"],
        fontsize=9,
    )
    add_box(
        ax,
        (4.75, 4.9),
        2.85,
        0.9,
        "Transform to body frame\nV_a,b^base = R_n->b V_a,n^base\n= [u,v,w]^T",
        COLORS["physics"],
        fontsize=8.5,
    )
    add_box(
        ax,
        (8.1, 4.9),
        2.6,
        0.9,
        "Baseline aero angles\nV_T, alpha=atan2(w,u)\nbeta=asin(v/V_T)",
        COLORS["physics"],
        fontsize=8.5,
    )
    add_box(
        ax,
        (11.15, 4.9),
        2.6,
        0.9,
        "Corrected angles\nalpha~=alpha+Delta alpha\nbeta~=beta+Delta beta",
        COLORS["physics"],
        fontsize=8.5,
    )

    add_box(
        ax,
        (2.0, 3.0),
        3.2,
        1.05,
        "Reconstruct body airspeed\nV~_a,b = (s_TAS*TAS_meas)\n[cos alpha~ cos beta~,\n sin beta~, sin alpha~ cos beta~]^T",
        COLORS["akf"],
        fontsize=8.5,
    )
    add_box(
        ax,
        (6.1, 3.0),
        3.0,
        1.05,
        "Rotate to NED frame\nV~_a,n = R_b->n V~_a,b",
        COLORS["akf"],
        fontsize=9,
    )
    add_box(
        ax,
        (10.0, 3.0),
        3.25,
        1.05,
        "Physics residual\nr_phys = (V_g - w_NN) - V~_a,n",
        COLORS["physics"],
        fontsize=9,
    )
    add_box(
        ax,
        (5.25, 1.05),
        4.5,
        0.95,
        "Physics loss\nL_physics = mean(||r_phys||^2)\n+ regularization terms",
        "#FADADD",
        fontsize=10,
    )

    # Arrows top into path
    arrow(ax, (1.65, 6.55), (2.3, 5.8), color="#444444")
    arrow(ax, (4.1, 6.55), (3.2, 5.8), color="#444444")
    arrow(ax, (6.6, 6.55), (5.9, 5.8), color="#444444")
    arrow(ax, (9.15, 6.55), (3.25, 4.05), color="#444444", rad=0.15)
    arrow(ax, (11.65, 6.55), (12.4, 5.8), color="#444444")

    # Main arrows
    arrow(ax, (4.25, 5.35), (4.75, 5.35))
    arrow(ax, (7.6, 5.35), (8.1, 5.35))
    arrow(ax, (10.7, 5.35), (11.15, 5.35))
    arrow(ax, (12.45, 4.9), (4.0, 4.05), color="#5B3B8A", rad=0.22)
    arrow(ax, (5.2, 3.55), (6.1, 3.55))
    arrow(ax, (9.1, 3.55), (10.0, 3.55))
    arrow(ax, (11.6, 3.0), (8.25, 2.0), color="#8A2D2D", rad=0.1)

    # residual also needs base air velocity
    arrow(ax, (2.8, 4.9), (11.1, 4.05), text="base airspeed term", color="#8A2D2D", rad=-0.08)

    # Warm-up note
    add_box(
        ax,
        (0.7, 1.0),
        3.35,
        1.25,
        "Warm-up freezing strategy\nfor first E_wu epochs:\nDelta alpha=0, Delta beta=0,\ns_TAS=1",
        "#F2F2F2",
        fontsize=8.5,
    )
    arrow(ax, (2.4, 2.25), (11.35, 5.0), text="blocks shortcut early", color="#8A2D2D", rad=-0.15)

    ax.text(
        12.1,
        1.55,
        "The loss encourages consistency\nbetween kinematic airspeed\nand sensor-constrained airspeed.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#444444",
    )

    save(fig, "Figure_2_velocity_triangle_physics_loss")


if __name__ == "__main__":
    figure_1_architecture()
    figure_2_physics_loss()
