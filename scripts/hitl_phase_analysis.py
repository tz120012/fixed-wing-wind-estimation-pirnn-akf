#!/usr/bin/env python3
"""HITL 半实物实验 — 按相位分阶段重新统计 + 论文级图表生成。

输入: HITL/aligned/hitl_aligned_master.csv  (47,499 行 × 81 列)

输出:
  - paper/figures/fig14_hitl_phase_timeseries.png 各 session 风速时序与真值对比
  - paper/figures/fig15_hitl_latency_violin.png   后端 × phase 的端到端时延小提琴图
  - paper/figures/fig_hitl_error_phase.png       各 phase 的 RMSE/dir_err 分组柱状图
  - paper/tables/table_hitl_phase_summary.md     表格形式的分阶段摘要
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
CSV = PROJ / "HITL/aligned/hitl_aligned_master.csv"
FIG_DIR = PROJ / "paper/figures"
TBL_DIR = PROJ / "paper/tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

PHASE_ORDER = ["steady", "gust_light", "gust_strong", "packet_loss"]
PHASE_LABELS = {"steady": "Steady", "gust_light": "Light Gust",
                "gust_strong": "Strong Gust", "packet_loss": "Packet Loss"}
PHASE_COLOR = {"steady": "#9DC3E6", "gust_light": "#FFD966",
               "gust_strong": "#F4B183", "packet_loss": "#C5E0B4",
               "warmup": "#D9D9D9"}
BACKEND_COLOR = {"ONNX CPU": "#2E75B6", "PyTorch CPU": "#C00000"}


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV, low_memory=False)
    # 真值有效掩码（剔除 warmup 起飞前的零风段）
    df["truth_valid"] = (df[["wind_north_ms", "wind_east_ms", "wind_down_ms"]]
                        .abs().sum(axis=1) > 0.01)
    df["pred_h_mag"] = np.sqrt(df["wind_n"] ** 2 + df["wind_e"] ** 2)
    df["truth_h_mag"] = np.sqrt(df["wind_north_ms"] ** 2 + df["wind_east_ms"] ** 2)
    df["truth_3d_mag"] = np.sqrt(df["wind_north_ms"] ** 2 + df["wind_east_ms"] ** 2 + df["wind_down_ms"] ** 2)
    # 分量误差
    df["err_n"] = df["wind_n"] - df["wind_north_ms"]
    df["err_e"] = df["wind_e"] - df["wind_east_ms"]
    df["err_d"] = df["wind_d"] - df["wind_down_ms"]
    df["err_3d"] = np.sqrt(df["err_n"] ** 2 + df["err_e"] ** 2 + df["err_d"] ** 2)
    df["err_mag"] = np.sqrt(df["pred_h_mag"] ** 2 + df["wind_d"] ** 2) - df["truth_3d_mag"]
    # 风向误差（仅水平）
    dir_p = np.degrees(np.arctan2(df["wind_e"], df["wind_n"]))
    dir_t = np.degrees(np.arctan2(df["wind_east_ms"], df["wind_north_ms"]))
    df["err_dir"] = ((dir_p - dir_t + 180) % 360) - 180
    return df


def phase_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for backend in ["ONNX CPU", "PyTorch CPU"]:
        for ph in PHASE_ORDER:
            sub = df[(df["backend"] == backend) & (df["jsbsim_phase_name"] == ph) & df["truth_valid"]]
            if len(sub) == 0:
                continue
            rmse_3d = float(np.sqrt(np.mean(sub["err_3d"] ** 2)))
            rmse_n = float(np.sqrt(np.mean(sub["err_n"] ** 2)))
            rmse_e = float(np.sqrt(np.mean(sub["err_e"] ** 2)))
            rmse_d = float(np.sqrt(np.mean(sub["err_d"] ** 2)))
            mag_rmse = float(np.sqrt(np.mean(sub["err_mag"] ** 2)))
            dir_mae = float(np.mean(np.abs(sub["err_dir"])))
            lat = sub["latency_e2e_ms"].dropna()
            rows.append({
                "backend": backend, "phase": ph, "n": len(sub),
                "rmse_3d": rmse_3d, "rmse_n": rmse_n, "rmse_e": rmse_e, "rmse_d": rmse_d,
                "mag_rmse": mag_rmse, "dir_mae": dir_mae,
                "lat_p50": float(lat.quantile(0.50)) if len(lat) else float("nan"),
                "lat_p95": float(lat.quantile(0.95)) if len(lat) else float("nan"),
                "lat_max": float(lat.max()) if len(lat) else float("nan"),
                "inf_p50": float(sub["inference_ms"].quantile(0.50)),
                "inf_p95": float(sub["inference_ms"].quantile(0.95)),
            })
    return pd.DataFrame(rows)


def write_summary_md(tbl: pd.DataFrame, df: pd.DataFrame):
    out = TBL_DIR / "table_hitl_phase_summary.md"
    lines = ["# HITL 半实物实验 — 端到端时延摘要（精度评估见 §6 离线测试）", ""]
    lines.append("> 重要说明：本节 HITL session 采集于 2026-04-16，伴机加载的是**早期 within_file_temporal 划分**下训练的 PI-GRU。")
    lines.append("> 由于 HITL 现场注入的是 ~5 m/s + 湍流的 OOD 强风，与早期训练分布严重不符，**在线估计精度不应作为算法精度证据**。")
    lines.append("> 本节仅作为：(1) 端到端时延的证据；(2) 真实异步通信链路可行性的证据；(3) 多后端对比（ONNX vs PyTorch CPU）。")
    lines.append("> 算法精度证据全部来自 §6 离线评估（stratified test_id/test_ood）。")
    lines.append("")
    lines.append(f"- 总对齐样本: **{len(df):,}** 行")
    lines.append(f"- 真值有效（剔 warmup）: **{int(df['truth_valid'].sum()):,}** 行 ({100 * df['truth_valid'].mean():.1f}%)")
    lines.append(f"- Sessions: **4** (PyTorch CPU × 2, ONNX CPU × 2)")
    lines.append(f"- 时间对齐 p95: **11.98 ms** (truth vs raspi)")
    lines.append("")
    lines.append("## 表 1 — 各 phase × backend 推理时延与端到端时延（核心证据）")
    lines.append("")
    lines.append("| Backend | Phase | n | inf p50 (ms) | inf p95 (ms) | E2E p50 (ms) | E2E p95 (ms) | E2E max (ms) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in tbl.iterrows():
        lines.append(f"| {r['backend']} | {PHASE_LABELS[r['phase']]} | {r['n']:,} "
                    f"| {r['inf_p50']:.2f} | {r['inf_p95']:.2f} "
                    f"| {r['lat_p50']:.2f} | {r['lat_p95']:.2f} | {r['lat_max']:.2f} |")
    # 总体
    lines.append("")
    lines.append("## 表 2 — 各 backend 总体时延（剔除 warmup）")
    lines.append("")
    lines.append("| Backend | n | inf p50/p95 (ms) | E2E p50 (ms) | E2E p95 (ms) | E2E max (ms) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for backend in ["ONNX CPU", "PyTorch CPU"]:
        sub = df[(df["backend"] == backend) & df["truth_valid"]]
        lat = sub["latency_e2e_ms"].dropna()
        inf = sub["inference_ms"].dropna()
        lines.append(f"| {backend} | {len(sub):,} | {inf.quantile(0.50):.2f}/{inf.quantile(0.95):.2f} "
                    f"| {lat.quantile(0.50):.2f} | {lat.quantile(0.95):.2f} | {lat.max():.2f} |")

    lines.append("")
    lines.append("## 附 — 在线估计与真值的统计差距（仅作 OOD 现象记录，不作精度结论）")
    lines.append("")
    lines.append("| Backend | Phase | n | 3D RMSE | dir MAE |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in tbl.iterrows():
        lines.append(f"| {r['backend']} | {PHASE_LABELS[r['phase']]} | {r['n']:,} "
                    f"| {r['rmse_3d']:.2f} m/s | {r['dir_mae']:.1f}° |")
    lines.append("")
    lines.append("> 上述误差较 §6 离线值高约一个量级，原因：(a) HITL 注入的真值风强为 ~5 m/s 含 ±1.8 m/s 湍流，"
                "训练集弱-中风段 [0.8, 2.5] m/s 不覆盖该区间；(b) 伴机模型为早期版本。"
                "本节将其作为 \"早期模型在 OOD + 湍流下的鲁棒不崩溃证据\"，主精度结论以 §6 stratified test 为准。")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  ✓ 摘要表: {out}")


def plot_timeseries(df: pd.DataFrame):
    """每个 session 一行：风速模 / 风向 / 时延 vs runtime。"""
    fig, axes = plt.subplots(4, 3, figsize=(16, 11), sharex="row")
    sessions = sorted(df["session_idx"].unique())
    backend_label = {1: "PyTorch CPU", 2: "ONNX CPU", 3: "PyTorch CPU", 4: "ONNX CPU"}

    for row, sess in enumerate(sessions):
        sub = df[df["session_idx"] == sess].copy().reset_index(drop=True)
        sub = sub[sub["truth_valid"]]
        if len(sub) == 0:
            continue
        # 用 jsbsim_sim_time_s 做 X 轴（session 内对齐到 0）
        t = sub["jsbsim_sim_time_s"].values
        t = t - t[0]

        # phase 颜色背景
        for ax_idx in range(3):
            ax = axes[row, ax_idx]
            for ph in PHASE_ORDER + ["warmup"]:
                mask = sub["jsbsim_phase_name"] == ph
                if not mask.any():
                    continue
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                # 找连续段
                edges = np.where(np.diff(idx) > 1)[0]
                segs = np.split(idx, edges + 1)
                for s in segs:
                    ax.axvspan(t[s[0]], t[s[-1]], color=PHASE_COLOR[ph], alpha=0.25, zorder=0)

        # col 0: 风速模
        ax = axes[row, 0]
        ax.plot(t, sub["truth_3d_mag"], color="#404040", lw=0.8, alpha=0.9, label="JSBSim Truth")
        ax.plot(t, np.sqrt(sub["wind_n"] ** 2 + sub["wind_e"] ** 2 + sub["wind_d"] ** 2),
                color=BACKEND_COLOR[backend_label[sess]], lw=0.8, alpha=0.85, label="PI-GRU online")
        ax.set_ylabel(f"S{sess} ({backend_label[sess]})\n|w| (m/s)")
        ax.grid(True, alpha=0.3, linestyle="--")
        if row == 0:
            ax.legend(loc="upper right")
            ax.set_title("Wind speed magnitude")

        # col 1: 风向（atan2 NE）
        ax = axes[row, 1]
        dir_t = np.degrees(np.arctan2(sub["wind_east_ms"], sub["wind_north_ms"]))
        dir_p = np.degrees(np.arctan2(sub["wind_e"], sub["wind_n"]))
        ax.plot(t, dir_t, color="#404040", lw=0.7, alpha=0.85)
        ax.plot(t, dir_p, color=BACKEND_COLOR[backend_label[sess]], lw=0.7, alpha=0.85)
        ax.set_ylabel("dir (°)")
        ax.set_ylim(-180, 180)
        ax.grid(True, alpha=0.3, linestyle="--")
        if row == 0:
            ax.set_title("Horizontal direction")

        # col 2: 端到端时延
        ax = axes[row, 2]
        ax.plot(t, sub["latency_e2e_ms"], color=BACKEND_COLOR[backend_label[sess]],
                lw=0.5, alpha=0.7)
        ax.set_ylabel("E2E lat (ms)")
        ax.set_yscale("log")
        ax.axhline(10, color="gray", ls=":", lw=0.8)
        ax.grid(True, alpha=0.3, linestyle="--", which="both")
        if row == 0:
            ax.set_title("End-to-end latency")

    for ax in axes[-1, :]:
        ax.set_xlabel("Sim time within session (s)")
    # phase legend
    handles = [plt.Rectangle((0, 0), 1, 1, fc=PHASE_COLOR[p], alpha=0.4) for p in PHASE_ORDER]
    fig.legend(handles, [PHASE_LABELS[p] for p in PHASE_ORDER],
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("HITL 4 sessions — Online estimate vs JSBSim truth, with phase shading",
                 y=1.03, fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "fig14_hitl_phase_timeseries.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 时序图: {out}")


def plot_latency_violin(df: pd.DataFrame):
    """各 phase × backend 的 E2E 时延小提琴图。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    sub = df[df["truth_valid"]].copy()

    for ax, backend in zip(axes, ["ONNX CPU", "PyTorch CPU"]):
        data, positions, colors = [], [], []
        for i, ph in enumerate(PHASE_ORDER):
            d = sub[(sub["backend"] == backend) & (sub["jsbsim_phase_name"] == ph)]["latency_e2e_ms"].dropna()
            if len(d) > 50:
                data.append(d.values)
                positions.append(i)
                colors.append(PHASE_COLOR[ph])
        if not data:
            continue
        parts = ax.violinplot(data, positions=positions, widths=0.7,
                              showmedians=True, showextrema=False)
        for pc, c in zip(parts["bodies"], colors):
            pc.set_facecolor(c)
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)
            pc.set_linewidth(0.5)
        # 在 violin 上覆盖 box 的 p50/p95
        for pos, d in zip(positions, data):
            p50 = np.percentile(d, 50)
            p95 = np.percentile(d, 95)
            ax.plot([pos - 0.18, pos + 0.18], [p50, p50], color="black", lw=1.5)
            ax.plot([pos - 0.12, pos + 0.12], [p95, p95], color="red", lw=1.0)
        ax.axhline(10, color="gray", ls=":", lw=1, label="10 ms threshold")
        ax.set_xticks(range(len(PHASE_ORDER)))
        ax.set_xticklabels([PHASE_LABELS[p] for p in PHASE_ORDER])
        ax.set_title(f"{backend}", fontweight="bold")
        ax.set_ylabel("End-to-end latency (ms)" if ax is axes[0] else "")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, linestyle="--", which="both")
        # legend (only first axis)
        if ax is axes[0]:
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0], [0], color="black", lw=1.5, label="median (p50)"),
                Line2D([0], [0], color="red", lw=1.0, label="p95"),
                Line2D([0], [0], color="gray", ls=":", lw=1, label="10 ms threshold"),
            ], loc="upper left")

    fig.suptitle("End-to-End Latency by Phase × Backend (HITL)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "fig15_hitl_latency_violin.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 时延小提琴图: {out}")


def plot_error_bar(df: pd.DataFrame, tbl: pd.DataFrame):
    """各 phase × backend 的 RMSE/dir_MAE 分组柱状图。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    metrics = [("rmse_3d", "3D RMSE (m/s)"),
               ("mag_rmse", "Magnitude RMSE (m/s)"),
               ("dir_mae", "Direction MAE (°)")]
    width = 0.36
    x = np.arange(len(PHASE_ORDER))

    for ax, (col, label) in zip(axes, metrics):
        for j, backend in enumerate(["ONNX CPU", "PyTorch CPU"]):
            vals = []
            for ph in PHASE_ORDER:
                row = tbl[(tbl["backend"] == backend) & (tbl["phase"] == ph)]
                vals.append(float(row[col].iloc[0]) if len(row) else np.nan)
            offset = (j - 0.5) * width
            ax.bar(x + offset, vals, width, label=backend,
                   color=BACKEND_COLOR[backend], edgecolor="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([PHASE_LABELS[p] for p in PHASE_ORDER], rotation=15)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        if ax is axes[0]:
            ax.legend(loc="upper left")

    fig.suptitle("HITL Online Estimation Error by Phase × Backend", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "fig_hitl_error_phase.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 误差柱状图: {out}")


def main():
    print("=" * 80)
    print("  HITL 半实物实验 — 分阶段重统计")
    print("=" * 80)
    df = load()
    print(f"\n  载入 {len(df):,} 行")
    print(f"  真值有效（剔 warmup）: {int(df['truth_valid'].sum()):,}")

    tbl = phase_summary_table(df)
    print("\n  ▼ 分阶段摘要")
    print(tbl.to_string(index=False, float_format="%.3f"))

    write_summary_md(tbl, df)
    print("\n  开始绘图...")
    plot_timeseries(df)
    plot_latency_violin(df)
    plot_error_bar(df, tbl)
    print("\n  ✓ HITL 重统计完成")


if __name__ == "__main__":
    main()
