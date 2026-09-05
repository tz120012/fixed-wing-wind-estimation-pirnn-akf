#!/usr/bin/env python3
"""生成论文主图（基于 dir_metrics_summary.json 已计算的指标）

  fig_dir_compare:    5 模型 × 分箱 dir_MAE 对比 + 几何下界（双子图：test_id / test_ood）
  fig_dir_diagnostics: 多视角 dir 指标雷达 + 柱状图（PI-GRU vs PX4-EKF2）
  fig_rmse_compare:   分箱 horizontal RMSE 与样本数（双子图）
  fig_overview:       核心结果一图概览（论文 Highlight）
"""
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

PROJ = Path(__file__).resolve().parent.parent
SUMMARY = PROJ / "data/evaluation/dir_metrics_summary.json"
FIG_DIR = PROJ / "paper/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.titlesize": 11,
})

MODELS_ORDER = ["PX4-EKF2", "Exp-A", "Exp-B", "Exp-C-v1", "Exp-C"]
COLORS = {
    "PX4-EKF2": "#7F7F7F",
    "Exp-C-v1": "#A6CEE3",
    "Exp-A":    "#33A02C",
    "Exp-B":    "#FB9A99",
    "Exp-C":    "#1F78B4",
    "Vanilla GRU": "#FF7F00",
}
LABELS = {
    "PX4-EKF2": "PX4-EKF2 (baseline)",
    "Exp-C-v1": "Exp-C-v1 (PI-GRU, ac=1.0/dir=0.5, old DL)",
    "Exp-C":    "Exp-C (PI-GRU, ac=1.0/dir=0.5)",
    "Exp-A":    "Exp-A (PI-GRU, ac=1.0/dir=0.1)",
    "Exp-B":    "Exp-B (PI-GRU, ac=5.0/dir=0.5)",
    "Vanilla GRU": "Vanilla GRU (no PI)",
}


def load_summary():
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def fig_dir_compare(summary):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    splits = ["test_id", "test_ood"]
    titles = ["Test-ID (4 strata × 80/10/10)", "Test-OOD (3 high-wind runs)"]

    for ax, split, title in zip(axes, splits, titles):
        # 几何下界（用 PX4-EKF2 的 by_bin 表示，所有模型 bin 相同）
        bins = summary["PX4-EKF2"]["per_split"][split]["by_bin"]
        x_labels = [b["label"] for b in bins]
        x = np.arange(len(x_labels))
        geom = [b["geom_lower"] for b in bins]
        n_per_bin = [b["n"] for b in bins]
        # 灰色阴影：几何下界（dir_err_min ≈ arctan(rmse_h / |w_h|)）
        ax.fill_between(x, 0, geom, color="lightgray", alpha=0.4, label="Geometric lower bound", zorder=0)
        # 各模型 dir_MAE
        for name in MODELS_ORDER:
            if name not in summary:
                continue
            mb = summary[name]["per_split"][split]["by_bin"]
            y = [b["dir_mae"] for b in mb]
            marker = "o" if name == "PX4-EKF2" else "s" if "Exp-C" in name else "^"
            lw = 2.2 if name == "Exp-C" else 1.4
            ax.plot(x, y, marker=marker, color=COLORS[name], lw=lw,
                    markersize=7 if name == "Exp-C" else 5, alpha=0.9,
                    label=LABELS[name])

        # 在底部加每个 bin 的样本数（次要轴）
        ax2 = ax.twinx()
        ax2.bar(x, n_per_bin, color="#E0E0E0", alpha=0.45, width=0.55, zorder=-1)
        ax2.set_ylabel("Samples (count, gray bars)", color="#909090", fontsize=9)
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", colors="#909090", labelsize=8)
        ax2.spines["right"].set_color("#909090")

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15)
        ax.set_xlabel(r"$|\mathbf{w}_h|$ bin (m/s)")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel("Direction MAE (°)")
            ax.legend(loc="upper right", framealpha=0.92)

    fig.suptitle("Wind direction MAE by horizontal-wind magnitude bin "
                "(5 models compared; geometric lower bound shaded)",
                fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "fig_dir_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def fig_dir_diagnostics(summary):
    """6 个多视角 dir 指标的横向柱状图：raw / eng / conf / clean / evil%
    PI-GRU (Exp-C) vs PX4-EKF2 baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=False)
    metrics = [("raw_dir", "raw dir MAE (°)"),
               ("eng_dir", r"eng dir MAE (°), $|w_h|\geq1.5$"),
               ("clean_dir", "clean dir MAE (°), |Δθ|<30°"),
               ("evil_pct_pct", "evil% (Δθ>30°)")]

    for ax, split in zip(axes, ["test_id", "test_ood"]):
        names = MODELS_ORDER
        n_metric = len(metrics)
        n_model = len(names)
        bar_w = 0.18
        x = np.arange(n_metric)
        for j, name in enumerate(names):
            if name not in summary:
                continue
            r = summary[name]["per_split"][split]
            vals = [
                r["raw_dir"],
                r["eng_dir"],
                r["clean_dir"],
                r["evil_pct"] * 100,  # 转成百分比
            ]
            offset = (j - (n_model - 1) / 2) * bar_w
            ax.bar(x + offset, vals, bar_w, label=LABELS[name], color=COLORS[name],
                   edgecolor="black", lw=0.4, alpha=0.9)
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=7,
                       rotation=90 if name == "PX4-EKF2" else 0)

        ax.set_xticks(x)
        ax.set_xticklabels([m[1] for m in metrics], rotation=12)
        ax.set_ylabel("Value (° or %)")
        ax.set_title(f"{split}", fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.92, ncol=1)

    fig.suptitle("Multi-perspective direction diagnostics: 5 models compared", 
                fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig_dir_diagnostics.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def fig_rmse_compare(summary):
    """分箱水平 RMSE 对比 + 几何下界估计。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=False)
    splits = ["test_id", "test_ood"]

    for ax, split in zip(axes, splits):
        bins = summary["PX4-EKF2"]["per_split"][split]["by_bin"]
        x_labels = [b["label"] for b in bins]
        x = np.arange(len(x_labels))
        for name in MODELS_ORDER:
            if name not in summary:
                continue
            mb = summary[name]["per_split"][split]["by_bin"]
            y = [b["rmse_h"] for b in mb]
            marker = "o" if name == "PX4-EKF2" else "s" if "Exp-C" in name else "^"
            lw = 2.2 if name == "Exp-C" else 1.3
            ax.plot(x, y, marker=marker, color=COLORS[name], lw=lw,
                    markersize=7 if name == "Exp-C" else 5,
                    label=LABELS[name], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=15)
        ax.set_xlabel(r"$|\mathbf{w}_h|$ bin (m/s)")
        ax.set_title(f"{split}", fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.set_axisbelow(True)
        if ax is axes[0]:
            ax.set_ylabel("Horizontal wind RMSE (m/s)")
            ax.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Horizontal-wind RMSE by magnitude bin",
                fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "fig_rmse_compare.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def fig_overview(summary):
    """论文 Highlight：四象限单图。
        (TL) 总 3D RMSE（5 模型 × 2 split 分组柱状图）
        (TR) raw dir MAE（同上）
        (BL) Test-ID 分箱 dir_MAE（核心证据）
        (BR) Test-ID 分箱 RMSE_h（次要证据）
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    names = MODELS_ORDER
    splits = ["test_id", "test_ood"]
    bar_w = 0.36
    x = np.arange(len(names))

    # TL: 3D RMSE
    ax = axes[0, 0]
    for j, sp in enumerate(splits):
        vals = [summary[n]["per_split"][sp]["rmse_3d"] for n in names]
        offset = (j - 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w, label=sp,
               color="#1F78B4" if sp == "test_id" else "#E31A1C",
               edgecolor="black", lw=0.4, alpha=0.85)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("3D RMSE (m/s)")
    ax.set_title("Overall 3D RMSE", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # TR: raw dir MAE
    ax = axes[0, 1]
    for j, sp in enumerate(splits):
        vals = [summary[n]["per_split"][sp]["raw_dir"] for n in names]
        offset = (j - 0.5) * bar_w
        ax.bar(x + offset, vals, bar_w, label=sp,
               color="#1F78B4" if sp == "test_id" else "#E31A1C",
               edgecolor="black", lw=0.4, alpha=0.85)
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("Raw direction MAE (°)")
    ax.set_title("Raw direction MAE", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    # BL: 分箱 dir_MAE on test_id
    ax = axes[1, 0]
    bins = summary["PX4-EKF2"]["per_split"]["test_id"]["by_bin"]
    xb = np.arange(len(bins))
    geom = [b["geom_lower"] for b in bins]
    ax.fill_between(xb, 0, geom, color="lightgray", alpha=0.45, label="Geometric lower bound")
    for name in names:
        mb = summary[name]["per_split"]["test_id"]["by_bin"]
        y = [b["dir_mae"] for b in mb]
        marker = "o" if name == "PX4-EKF2" else "s" if "Exp-C" in name else "^"
        lw = 2.2 if name == "Exp-C" else 1.3
        ax.plot(xb, y, marker=marker, color=COLORS[name], lw=lw,
                markersize=6, label=name)
    ax.set_xticks(xb)
    ax.set_xticklabels([b["label"] for b in bins], rotation=15)
    ax.set_xlabel(r"$|\mathbf{w}_h|$ (m/s)")
    ax.set_ylabel("Direction MAE (°)")
    ax.set_title("Test-ID: dir_MAE vs wind magnitude", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    # BR: 分箱 horizontal RMSE on test_id
    ax = axes[1, 1]
    for name in names:
        mb = summary[name]["per_split"]["test_id"]["by_bin"]
        y = [b["rmse_h"] for b in mb]
        marker = "o" if name == "PX4-EKF2" else "s" if "Exp-C" in name else "^"
        lw = 2.2 if name == "Exp-C" else 1.3
        ax.plot(xb, y, marker=marker, color=COLORS[name], lw=lw,
                markersize=6, label=name)
    ax.set_xticks(xb)
    ax.set_xticklabels([b["label"] for b in bins], rotation=15)
    ax.set_xlabel(r"$|\mathbf{w}_h|$ (m/s)")
    ax.set_ylabel("Horizontal wind RMSE (m/s)")
    ax.set_title("Test-ID: RMSE_h vs wind magnitude", fontweight="bold")
    ax.legend(loc="upper left", fontsize=8); ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("PI-GRU (Exp-C) vs PX4-EKF2 baseline — Overview on stratified test sets",
                fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "fig_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")


def main():
    print("=" * 80)
    print("  论文主图生成（基于 dir_metrics_summary.json）")
    print("=" * 80)
    s = load_summary()
    print(f"  载入: {SUMMARY}")
    print(f"  模型: {list(s.keys())}")
    fig_dir_compare(s)
    fig_dir_diagnostics(s)
    fig_rmse_compare(s)
    fig_overview(s)
    print("\n  ✓ 所有图已输出至 paper/figures/")


if __name__ == "__main__":
    main()
