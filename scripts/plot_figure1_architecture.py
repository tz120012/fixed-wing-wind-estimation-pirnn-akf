"""Figure 1 - PI-GRU front-end + PIRNN-AKF back-end architecture schematic.

Redrawn (v2.3 figure-audit) to fix omissions in the original TikZ diagram
(paper/figures/figure1.tex), cross-checked against the manuscript text
(Sections 2.2-2.3, Table M1) and the model implementation
(src/2_pigru_module.py, src/7_sitl_closedloop_eval.py):

  1. The multi-task output head has FIVE branches, not three: wind head,
     dynamic-R head, diagnostic-Q head, angle/airspeed-scale compensation
     head [Delta-alpha, Delta-beta, s_TAS], and confidence head s_k. The
     angle-compensation and confidence heads were entirely missing from the
     original figure even though the manuscript text (line ~168, Table M1)
     explicitly lists all five as the network's output heads.
  2. The confidence-gated EMA smoothing stage (Eq. 29-30) that turns the
     raw Kalman posterior into the deployed w_AKF,k was missing; the caption
     text explicitly says this mechanism is "shown in Figure 1's inference
     fusion branch", which was not true of the original diagram.
  3. The "weak-wind regularization" loss box was wired from the dynamic-R
     head (r_scale) in the original diagram; Eq. 17-19 show the anti-collapse
     hinge / magnitude / direction losses operate on the wind head output
     (w_NN), not on r_scale. Fixed to originate from the wind head.
  4. The uncertainty-scale regularization term L_reg (Eq. 20; see
     calculate_scale_regularization() in src/3_train_pigru.py) was not
     represented at all; added as its own training-only loss box fed by the
     q_scale and r_scale heads.

v2.3 simplification pass (reduce box/line count for print legibility, add
explicit data-flow labels on every edge instead of packing inline formulas
into box text): the four separate training-loss boxes were collapsed into
one aggregated "Training-only losses (Eq. 16-20)" node fed by labeled edges
carrying the exact per-head variable that each loss term consumes; the
separate "Delay" buffer box was replaced by a self-loop on the EMA box
labeled with the fed-back variable w_AKF,k-1; the standalone "Posterior
wind" output box was replaced by a labeled terminal arrow; and inline
covariance/update formulas were removed from box interiors (they remain in
Eq. 21-30 and Appendix F) so each box only carries a short name while the
connecting arrows carry the variable names. A new dark dashed backprop arrow
from the aggregated loss node back into the GRU backbone makes the
training closed-loop explicit (forward pass -> loss -> backprop -> weight
update), which the previous version left implicit.

Run:
    python scripts/plot_figure1_architecture.py
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.path import Path as MPath

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper_plot_style import BLUE, GRAY, GREEN, RED, apply_style, save_figure

FRONT_FILL = "#F3F7FB"
STAGE_EDGE = "#8C8C8C"
BOX_EDGE = "#595959"
BACKPROP = "#333333"


def rounded_box(ax, cx, cy, w, h, *, edgecolor=BOX_EDGE, facecolor="white",
                 lw=1.0, linestyle="-", zorder=4, pad=0.045):
    box = mpatches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={pad},rounding_size=0.09",
        linewidth=lw, edgecolor=edgecolor, facecolor=facecolor,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(box)
    return box


def label(ax, cx, cy, text, *, fontsize=8.0, zorder=5, color="black"):
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
             zorder=zorder, color=color, linespacing=1.4)


def elbow(ax, pts, *, color, lw=1.15, style="-", zorder=3, mutation_scale=8.5):
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(pts) - 1)
    path = MPath(pts, codes)
    patch = mpatches.PathPatch(
        path, edgecolor=color, facecolor="none", lw=lw, linestyle=style,
        zorder=zorder, capstyle="round", joinstyle="round",
    )
    ax.add_patch(patch)
    p1, p2 = pts[-2], pts[-1]
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.001,
                         shrinkA=0.0, shrinkB=0.0, mutation_scale=mutation_scale),
        zorder=zorder + 1,
    )


def edge_label(ax, x, y, text, *, color="black", fontsize=6.6, zorder=6, ha="center"):
    """Small data-flow label placed on/near an edge, with a white halo box
    so it stays legible where it crosses other lines."""
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize, color=color,
             zorder=zorder,
             bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                       edgecolor="none", alpha=0.92))


def stage_box(ax, x0, x1, y0, y1, title, *, zorder=1):
    box = mpatches.FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.0,rounding_size=0.16",
        linewidth=1.0, edgecolor=STAGE_EDGE, facecolor=FRONT_FILL,
        zorder=zorder,
    )
    ax.add_patch(box)
    ax.text(x0 + 0.22, y1 - 0.30, title, ha="left", va="center", fontsize=9.3, zorder=zorder + 1)
    return box


def main():
    apply_style()
    fig, ax = plt.subplots(figsize=(17.6, 9.4))
    ax.set_xlim(-0.6, 22.6)
    ax.set_ylim(-4.85, 3.75)
    ax.set_aspect("equal")
    ax.axis("off")

    BOX_W, BOX_H = 2.3, 0.9
    HEAD_W, HEAD_H = 2.55, 0.72

    # ---- stage backgrounds ----
    stage_box(ax, -0.5, 14.05, -2.15, 3.55, "PI-GRU front-end", zorder=1)
    stage_box(ax, 14.5, 22.4, -1.75, 2.15, "AKF back-end (inference only, no gradient)", zorder=1)
    stage_box(ax, -0.5, 22.4, -4.65, -2.55, "Training-only constraints (used only while training)", zorder=1)

    # ================= front-end trunk =================
    x_sens, x_win, x_gru, x_lat = 1.25, 3.85, 6.45, 9.05
    y_trunk = 0.9
    rounded_box(ax, x_sens, y_trunk, BOX_W, BOX_H)
    label(ax, x_sens, y_trunk, "Onboard sensors\nGPS/IMU/pitot/AHRS", fontsize=7.3)
    rounded_box(ax, x_win, y_trunk, BOX_W, BOX_H)
    label(ax, x_win, y_trunk, "Sliding window\n41-D, $z$-norm", fontsize=7.6)
    rounded_box(ax, x_gru, y_trunk, BOX_W, BOX_H)
    label(ax, x_gru, y_trunk, "GRU backbone", fontsize=8.2)
    rounded_box(ax, x_lat, y_trunk, BOX_W, BOX_H)
    label(ax, x_lat, y_trunk, "Shared latent", fontsize=8.2)

    elbow(ax, [(x_sens + BOX_W / 2, y_trunk), (x_win - BOX_W / 2, y_trunk)], color=BLUE)
    elbow(ax, [(x_win + BOX_W / 2, y_trunk), (x_gru - BOX_W / 2, y_trunk)], color=BLUE)
    edge_label(ax, (x_win + BOX_W / 2 + x_gru - BOX_W / 2) / 2, y_trunk + 0.32,
               r"$\mathbf{X}_{k-T+1:k}$", color=BLUE)
    elbow(ax, [(x_gru + BOX_W / 2, y_trunk), (x_lat - BOX_W / 2, y_trunk)], color=BLUE)
    edge_label(ax, (x_gru + BOX_W / 2 + x_lat - BOX_W / 2) / 2, y_trunk + 0.32, "$h_k$", color=BLUE)

    # ================= five output heads =================
    x_head = 12.25
    y_wind, y_r, y_conf, y_q, y_ang = 2.6, 1.55, 0.5, -0.55, -1.6

    rounded_box(ax, x_head, y_wind, HEAD_W, HEAD_H, edgecolor=BLUE, lw=1.3)
    label(ax, x_head, y_wind, "Wind head", fontsize=8.0)
    rounded_box(ax, x_head, y_r, HEAD_W, HEAD_H, edgecolor=BLUE, lw=1.3)
    label(ax, x_head, y_r, "Dynamic-$R$ head", fontsize=8.0)
    rounded_box(ax, x_head, y_conf, HEAD_W, HEAD_H, edgecolor=BLUE, lw=1.3)
    label(ax, x_head, y_conf, "Confidence head", fontsize=8.0)
    rounded_box(ax, x_head, y_q, HEAD_W, HEAD_H, edgecolor=GRAY, lw=1.15, linestyle="--")
    label(ax, x_head, y_q, "Diagnostic-$Q$ head", fontsize=8.0)
    rounded_box(ax, x_head, y_ang, HEAD_W, HEAD_H, edgecolor=BLUE, lw=1.3)
    label(ax, x_head, y_ang, "Angle / TAS-scale head", fontsize=7.6)

    # Staggered branch lanes (distinct x per head) so the five overlapping
    # verticals do not paint over one another where their y-spans coincide.
    x_split0 = x_lat + BOX_W / 2 + 0.18
    branches = [
        (y_wind, BLUE, "-", x_split0 + 0.00),
        (y_r, BLUE, "-", x_split0 + 0.10),
        (y_conf, BLUE, "-", x_split0 + 0.20),
        (y_q, GRAY, "--", x_split0 + 0.30),
        (y_ang, BLUE, "-", x_split0 + 0.40),
    ]
    for y_h, c, ls, x_sp in branches:
        elbow(ax, [(x_lat + BOX_W / 2, y_trunk), (x_sp, y_trunk), (x_sp, y_h),
                    (x_head - HEAD_W / 2, y_h)], color=c, style=ls)
    edge_label(ax, x_split0 - 0.02, y_trunk + 0.30, "$\\mathbf{z}_k$", color=BLUE, ha="left", fontsize=7.0)

    # ================= AKF back-end =================
    x_akf, y_akf = 16.55, 0.5
    AKF_W, AKF_H = 2.85, 1.3
    rounded_box(ax, x_akf, y_akf, AKF_W, AKF_H, edgecolor=GREEN, lw=1.4)
    label(ax, x_akf, y_akf, "Dynamic-$R$ AKF\npredict + update", fontsize=8.2)

    y_wind_entry = y_akf + 0.28
    elbow(ax, [(x_head + HEAD_W / 2, y_wind), (x_akf - AKF_W / 2 - 0.35, y_wind),
                (x_akf - AKF_W / 2 - 0.35, y_wind_entry), (x_akf - AKF_W / 2, y_wind_entry)], color=GREEN)
    edge_label(ax, x_head + HEAD_W / 2 + 0.15, y_wind + 0.26, r"$\mathbf{w}_{\mathrm{NN},k}$",
               color=GREEN, ha="left", fontsize=7.0)
    elbow(ax, [(x_head + HEAD_W / 2, y_r), (x_akf - AKF_W / 2, y_r)], color=GREEN)
    edge_label(ax, (x_head + HEAD_W / 2 + x_akf - AKF_W / 2) / 2, y_r + 0.24, r"$\mathbf{r}_{\mathrm{scale},k}$",
               color=GREEN, fontsize=7.0)
    elbow(ax, [(x_head + HEAD_W / 2, y_q), (x_akf - AKF_W / 2 - 0.35, y_q),
                (x_akf - AKF_W / 2 - 0.35, y_akf - 0.5), (x_akf - AKF_W / 2, y_akf - 0.5)],
          color=GRAY, style="--")
    edge_label(ax, x_head + HEAD_W / 2 + 0.15, y_q - 0.26, r"$\mathbf{q}_{\mathrm{scale},k}$",
               color=GRAY, ha="left", fontsize=7.0)

    x_ema, y_ema = 19.75, y_akf
    EMA_W, EMA_H = 2.75, 1.3
    rounded_box(ax, x_ema, y_ema, EMA_W, EMA_H, edgecolor=GREEN, lw=1.4)
    label(ax, x_ema, y_ema, "Confidence-gated\nEMA", fontsize=8.2)

    elbow(ax, [(x_akf + AKF_W / 2, y_akf), (x_ema - EMA_W / 2, y_ema)], color=GREEN)
    edge_label(ax, (x_akf + AKF_W / 2 + x_ema - EMA_W / 2) / 2, y_akf + 0.24, r"$\hat{\mathbf{x}}_{k|k}$",
               color=GREEN, fontsize=7.0)

    # confidence head -> EMA box: dip below the AKF box so the line does not
    # cut through it on its way to the EMA box.
    y_conf_bus = y_akf - AKF_H / 2 - 0.35
    x_conf_drop = x_akf - AKF_W / 2 - 1.0
    elbow(ax, [(x_head + HEAD_W / 2, y_conf), (x_conf_drop, y_conf), (x_conf_drop, y_conf_bus),
                (x_ema - EMA_W / 2 - 0.5, y_conf_bus), (x_ema - EMA_W / 2 - 0.5, y_ema - 0.42),
                (x_ema - EMA_W / 2, y_ema - 0.42)], color=BLUE)
    edge_label(ax, x_head + HEAD_W / 2 + 0.15, y_conf - 0.26, "$s_k$", color=BLUE, ha="left", fontsize=7.0)

    # feedback self-loop replacing the old separate "Delay" box
    loop_y = y_ema + EMA_H / 2
    elbow(ax, [(x_ema - 0.55, loop_y), (x_ema - 0.55, loop_y + 0.55),
               (x_ema + 0.55, loop_y + 0.55), (x_ema + 0.55, loop_y)], color=GREEN, style="--")
    edge_label(ax, x_ema, loop_y + 0.72, r"$\mathbf{w}_{\mathrm{AKF},k-1}$ (previous step)",
               color=GREEN, fontsize=6.8)

    # terminal output label instead of a separate "Posterior wind" box
    x_out_arrow = x_ema + EMA_W / 2 + 1.15
    elbow(ax, [(x_ema + EMA_W / 2, y_ema), (x_out_arrow, y_ema)], color=GREEN)
    edge_label(ax, x_ema + EMA_W / 2 + 0.12, y_ema + 0.26, r"$\hat{\mathbf{w}}_{\mathrm{AKF},k}$",
               color=GREEN, ha="left", fontsize=7.2)
    ax.text(x_out_arrow + 0.08, y_ema, "deployed\nwind output", ha="left", va="center",
            fontsize=7.6, zorder=5, linespacing=1.3)

    # ================= training-only constraints (aggregated) =================
    y_train = -3.55
    KIN_W, KIN_H = 2.55, 0.9
    LOSS_W, LOSS_H = 6.6, 1.65
    x_kin, x_loss = 2.9, 10.6

    rounded_box(ax, x_kin, y_train, KIN_W, KIN_H, edgecolor=BOX_EDGE)
    label(ax, x_kin, y_train, "Kinematic slice\n$\\mathbf{v}_g$, TAS, attitude", fontsize=7.4)

    rounded_box(ax, x_loss, y_train, LOSS_W, LOSS_H, edgecolor=RED, lw=1.4)
    label(ax, x_loss, y_train,
          "Training-only losses (Eq. 16\u201320)\n"
          "physics closure \u00b7 data \u00b7 weak-wind reg. \u00b7 uncertainty-scale reg.",
          fontsize=7.6)

    elbow(ax, [(x_kin + KIN_W / 2, y_train), (x_loss - LOSS_W / 2, y_train)], color=RED, style="--")
    edge_label(ax, (x_kin + KIN_W / 2 + x_loss - LOSS_W / 2) / 2, y_train + 0.26,
               "kinematic pseudo-meas.", color=RED, fontsize=6.8)

    y_bus_bot = -4.35
    elbow(ax, [(x_win, y_trunk - BOX_H / 2), (x_win, y_bus_bot), (x_kin, y_bus_bot),
                (x_kin, y_train - KIN_H / 2)], color=RED, style="--")

    y_bus_top = -2.35
    x_wind_lane = x_head - HEAD_W / 2 - 0.35
    elbow(ax, [(x_head - HEAD_W / 2, y_wind), (x_wind_lane, y_wind), (x_wind_lane, y_bus_top),
                (x_loss - 1.6, y_bus_top), (x_loss - 1.6, y_train + LOSS_H / 2)], color=RED, style="--")
    edge_label(ax, x_wind_lane + 0.1, y_bus_top - 0.28, r"$\mathbf{w}_{\mathrm{NN},k}$",
               color=RED, ha="left", fontsize=6.8)

    x_ang_lane = x_head - HEAD_W / 2 - 0.7
    elbow(ax, [(x_head - HEAD_W / 2, y_ang), (x_ang_lane, y_ang), (x_ang_lane, y_bus_top + 0.22),
                (x_loss - 0.35, y_bus_top + 0.22), (x_loss - 0.35, y_train + LOSS_H / 2)], color=RED, style="--")
    edge_label(ax, x_ang_lane - 0.1, y_ang - 0.26, r"$[\Delta\alpha,\Delta\beta,s_{\mathrm{TAS}}]$",
               color=RED, ha="right", fontsize=6.8)

    x_q_lane = x_head - HEAD_W / 2 - 0.4
    elbow(ax, [(x_head - HEAD_W / 2, y_q), (x_q_lane, y_q), (x_q_lane, y_bus_top - 0.18),
                (x_loss + 0.6, y_bus_top - 0.18), (x_loss + 0.6, y_train + LOSS_H / 2)], color=RED, style="--")
    x_r_lane = x_head + HEAD_W / 2 + 0.6
    elbow(ax, [(x_head + HEAD_W / 2, y_r), (x_r_lane, y_r), (x_r_lane, y_bus_top - 0.36),
                (x_loss + 1.7, y_bus_top - 0.36), (x_loss + 1.7, y_train + LOSS_H / 2)], color=RED, style="--")
    edge_label(ax, x_r_lane + 0.1, y_r + 0.26, r"$\mathbf{q}_{\mathrm{scale},k},\mathbf{r}_{\mathrm{scale},k}$",
               color=RED, ha="left", fontsize=6.8)

    # ---- explicit training closed loop: aggregated loss -> backprop -> GRU ----
    x_bp_drop = x_loss - LOSS_W / 2 - 0.5
    elbow(ax, [(x_loss - LOSS_W / 2, y_train + 0.3), (x_bp_drop, y_train + 0.3),
                (x_bp_drop, y_trunk - BOX_H / 2 - 0.55), (x_gru, y_trunk - BOX_H / 2 - 0.55),
                (x_gru, y_trunk - BOX_H / 2)], color=BACKPROP, lw=1.7, style=(0, (2, 2)))
    edge_label(ax, x_bp_drop + 0.15, (y_train + y_trunk) / 2 - 0.3,
               "backprop:\nupdate $\\theta$ (Adam)", color=BACKPROP, ha="left", fontsize=7.2)

    # ---- legend ----
    legend_items = [
        (BLUE, "-", "inference path"),
        (RED, "--", "training-only loss path"),
        (GREEN, "-", "AKF / EMA fusion"),
        (GRAY, "--", "diagnostic / optional signal"),
        (BACKPROP, (0, (2, 2)), "backprop / weight update"),
    ]
    lx0, ly0 = -0.3, -4.55
    for i, (c, ls, txt) in enumerate(legend_items):
        xs = lx0 + i * 4.6
        ax.plot([xs, xs + 0.5], [ly0, ly0], color=c, lw=1.6, linestyle=ls, solid_capstyle="butt")
        ax.text(xs + 0.65, ly0, txt, ha="left", va="center", fontsize=7.6)

    fig.tight_layout()
    out_base = PROJECT_ROOT / "data" / "figure1" / "figure1_architecture"
    save_figure(fig, out_base, copy_to_paper="figure1")
    print(f"Saved to {out_base} and paper/figures/figure1.*")


if __name__ == "__main__":
    main()
