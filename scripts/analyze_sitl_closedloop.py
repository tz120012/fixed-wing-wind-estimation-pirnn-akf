"""
analyze_sitl_closedloop.py  ─  SITL 闭环实验结果对比分析脚本
=============================================================
读取 7_sitl_closedloop_eval.py 生成的两组 CSV（baseline & pirnn_akf），
生成论文 §3.5.2 所需的对比图表与数值摘要。

输出
----
  1. comparison_timeseries.png  ─  时序对比（XTE、空速误差、升降舵、风估计）
  2. comparison_boxplot.png     ─  按阶段分组箱线图（XTE / 空速误差 / 舵面 jitter）
  3. comparison_summary.txt     ─  数值摘要表（均值 ± 标准差，各阶段分层）
  4. wind_accuracy.png          ─  风估计精度对比（PIRNN-AKF vs EKF2 vs 真值）

典型运行命令
------------
  python3 scripts/analyze_sitl_closedloop.py \\
      --baseline SITL/sitl_baseline_20260603_*.csv \\
      --pirnn    SITL/sitl_pirnn_akf_20260603_*.csv \\
      --output   SITL/comparison

  # 或者只分析单组（对比将只显示有数据的那组）
  python3 scripts/analyze_sitl_closedloop.py \\
      --pirnn SITL/sitl_pirnn_akf_20260603_120000.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── 复用项目图表样式 ──────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    from paper_plot_style import apply_style, DPI, BLUE, RED, GREEN, GRAY, ORANGE
except ImportError:
    def apply_style(): pass
    DPI = 300
    BLUE, RED, GREEN, GRAY, ORANGE = "#1F77B4", "#E64B35", "#00A087", "#7F7F7F", "#D55E00"

CYAN = "#4DBBD5"

# 五阶段顺序与显示名
PHASES = ["warmup", "steady", "gust_light", "gust_strong", "packet_loss"]
PHASE_LABEL = {
    "warmup":       "Warmup",
    "steady":       "Steady",
    "gust_light":   "Gust-Light",
    "gust_strong":  "Gust-Strong",
    "packet_loss":  "Pkt-Loss",
}

# 跳过 warmup 阶段（飞机还在爬升）
EVAL_PHASES = ["steady", "gust_light", "gust_strong", "packet_loss"]


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载与预处理
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 数值列强制转换（'nan' 字符串 → NaN）
    num_cols = [c for c in df.columns if c not in ("phase",)]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    # 过滤 warmup 阶段不参与控制评估（xtrack 尚未有意义）
    return df


def compute_jitter(series: pd.Series) -> pd.Series:
    """二阶差分均值（舵面抖动指标），按行对齐返回 NaN 填充的 Series。"""
    arr = series.to_numpy(dtype=float)
    d2 = np.full_like(arr, np.nan)
    if len(arr) >= 3:
        d2[2:] = np.abs(np.diff(arr, n=2))
    return pd.Series(d2, index=series.index)


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """补充派生列：elevator_jitter、wind_est_err（PIRNN-AKF vs 真值）、wind_ekf2_err。"""
    df = df.copy()
    if "elevator_raw" in df.columns:
        df["elevator_jitter"] = compute_jitter(df["elevator_raw"])
    if "throttle_raw" in df.columns:
        df["throttle_jitter"] = compute_jitter(df["throttle_raw"])

    # 风场估计误差（3D RMSE，逐行）
    for prefix, cols in [
        ("est", ["wind_est_n", "wind_est_e", "wind_est_d"]),
        ("ekf2", ["wind_ekf2_n", "wind_ekf2_e"]),
    ]:
        gt_cols = ["wind_gt_n", "wind_gt_e"] if prefix == "ekf2" else \
                  ["wind_gt_n", "wind_gt_e", "wind_gt_d"]
        if all(c in df.columns for c in cols) and all(c in df.columns for c in gt_cols):
            sq_err = sum(
                (df[e] - df[g]) ** 2
                for e, g in zip(cols, gt_cols)
            )
            df[f"wind_{prefix}_rmse_inst"] = np.sqrt(sq_err)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 图 1：时序对比
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(
    df_base: pd.DataFrame | None,
    df_pirnn: pd.DataFrame | None,
    output_path: str,
):
    apply_style()
    rows = 4
    fig, axes = plt.subplots(rows, 1, figsize=(10, 9), sharex=True)
    fig.subplots_adjust(hspace=0.35)

    def shade_phases(ax, df):
        """在轴上绘制阶段背景色。"""
        phase_colors = {
            "warmup": "#f0f0f0",
            "steady": "#e8f4f8",
            "gust_light": "#fff3cd",
            "gust_strong": "#ffe0e0",
            "packet_loss": "#e8e0f0",
        }
        if "runtime_s" not in df.columns or "phase" not in df.columns:
            return
        for ph, color in phase_colors.items():
            mask = df["phase"] == ph
            if not mask.any():
                continue
            t_start = df.loc[mask, "runtime_s"].min()
            t_end = df.loc[mask, "runtime_s"].max()
            ax.axvspan(t_start, t_end, color=color, alpha=0.4, zorder=0)

    ref_df = df_pirnn if df_pirnn is not None else df_base

    for ax in axes:
        if ref_df is not None:
            shade_phases(ax, ref_df)

    # ── 子图 0: XTE ──
    ax = axes[0]
    ax.set_ylabel("XTE [m]")
    ax.set_title("Lateral Cross-Track Error")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    if df_base is not None and "xtrack_error" in df_base.columns:
        ax.plot(df_base["runtime_s"], df_base["xtrack_error"].abs(),
                color=GRAY, lw=0.8, alpha=0.8, label="Baseline (EKF2)")
    if df_pirnn is not None and "xtrack_error" in df_pirnn.columns:
        ax.plot(df_pirnn["runtime_s"], df_pirnn["xtrack_error"].abs(),
                color=BLUE, lw=0.9, alpha=0.9, label="PIRNN-AKF")
    ax.legend(loc="upper right", fontsize=8)

    # ── 子图 1: 空速误差 ──
    ax = axes[1]
    ax.set_ylabel("Airspeed Err [m/s]")
    ax.set_title("Airspeed Tracking Error (|actual − target|)")
    ax.axhline(0, color="k", lw=0.6, ls="--")
    if df_base is not None and "airspeed_err" in df_base.columns:
        ax.plot(df_base["runtime_s"], df_base["airspeed_err"].abs(),
                color=GRAY, lw=0.8, alpha=0.8, label="Baseline (EKF2)")
    if df_pirnn is not None and "airspeed_err" in df_pirnn.columns:
        ax.plot(df_pirnn["runtime_s"], df_pirnn["airspeed_err"].abs(),
                color=BLUE, lw=0.9, alpha=0.9, label="PIRNN-AKF")
    ax.legend(loc="upper right", fontsize=8)

    # ── 子图 2: 升降舵 jitter ──
    ax = axes[2]
    ax.set_ylabel("Elevator Jitter\n[PWM diff²]")
    ax.set_title("Elevator Actuator Jitter (2nd-order difference)")
    if df_base is not None and "elevator_jitter" in df_base.columns:
        ax.plot(df_base["runtime_s"], df_base["elevator_jitter"],
                color=GRAY, lw=0.8, alpha=0.7, label="Baseline (EKF2)")
    if df_pirnn is not None and "elevator_jitter" in df_pirnn.columns:
        ax.plot(df_pirnn["runtime_s"], df_pirnn["elevator_jitter"],
                color=BLUE, lw=0.9, alpha=0.8, label="PIRNN-AKF")
    ax.legend(loc="upper right", fontsize=8)

    # ── 子图 3: 风估计精度（PIRNN-AKF 模式下有意义）──
    ax = axes[3]
    ax.set_ylabel("Wind Est. Err [m/s]")
    ax.set_title("Wind Estimation Error (Inst. RMSE vs. JSBSim Truth)")
    ax.set_xlabel("Runtime [s]")
    if df_pirnn is not None:
        if "wind_est_rmse_inst" in df_pirnn.columns:
            ax.plot(df_pirnn["runtime_s"], df_pirnn["wind_est_rmse_inst"],
                    color=BLUE, lw=0.9, alpha=0.9, label="PIRNN-AKF vs truth")
        if "wind_ekf2_rmse_inst" in df_pirnn.columns:
            ax.plot(df_pirnn["runtime_s"], df_pirnn["wind_ekf2_rmse_inst"],
                    color=RED, lw=0.8, alpha=0.8, ls="--", label="EKF2 vs truth (horiz)")
    ax.legend(loc="upper right", fontsize=8)

    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 时序对比图: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 图 2：按阶段箱线图
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplot(
    df_base: pd.DataFrame | None,
    df_pirnn: pd.DataFrame | None,
    output_path: str,
):
    apply_style()
    metrics = [
        ("xtrack_error",     "|XTE| [m]",             True),   # (列名, y轴标签, 取绝对值)
        ("airspeed_err",     "|Airspeed Err| [m/s]",  True),
        ("elevator_jitter",  "Elevator Jitter",        False),
    ]
    n_metrics = len(metrics)
    n_phases = len(EVAL_PHASES)

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.5))
    if n_metrics == 1:
        axes = [axes]

    x_base = np.arange(n_phases) * 3.0
    width = 1.0

    for ax, (col, ylabel, take_abs) in zip(axes, metrics):
        ax.set_title(ylabel, fontsize=9, pad=4)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x_base + width / 2)
        ax.set_xticklabels(
            [PHASE_LABEL[p] for p in EVAL_PHASES], rotation=30, ha="right", fontsize=8
        )
        ax.grid(axis="y", alpha=0.3)

        for i, phase in enumerate(EVAL_PHASES):
            base_vals = []
            pirnn_vals = []

            if df_base is not None and col in df_base.columns:
                s = df_base.loc[df_base["phase"] == phase, col].dropna()
                base_vals = s.abs().values if take_abs else s.values

            if df_pirnn is not None and col in df_pirnn.columns:
                s = df_pirnn.loc[df_pirnn["phase"] == phase, col].dropna()
                pirnn_vals = s.abs().values if take_abs else s.values

            pos = x_base[i]
            for k, (vals, color) in enumerate(
                [(base_vals, GRAY), (pirnn_vals, BLUE)]
            ):
                if len(vals) == 0:
                    continue
                bp = ax.boxplot(
                    vals,
                    positions=[pos + k * width],
                    widths=width * 0.8,
                    patch_artist=True,
                    medianprops=dict(color="white", lw=1.5),
                    whiskerprops=dict(color=color, lw=0.8),
                    capprops=dict(color=color, lw=0.8),
                    flierprops=dict(
                        marker=".", markersize=2, color=color, alpha=0.4
                    ),
                    boxprops=dict(facecolor=color, alpha=0.75),
                    showfliers=True,
                )

    # 图例
    legend_handles = [
        mpatches.Patch(facecolor=GRAY, alpha=0.75, label="Baseline (EKF2)"),
        mpatches.Patch(facecolor=BLUE, alpha=0.75, label="PIRNN-AKF"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Control Performance by Wind Phase: PIRNN-AKF vs. EKF2 Baseline",
        fontsize=10,
        y=1.06,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 箱线图: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 图 3：风估计精度对比
# ─────────────────────────────────────────────────────────────────────────────

def plot_wind_accuracy(df_pirnn: pd.DataFrame, output_path: str):
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    # 左图：逐阶段 RMSE 柱状图
    ax = axes[0]
    rmse_est, rmse_ekf2 = [], []
    valid_phases = []
    for phase in EVAL_PHASES:
        sub = df_pirnn[df_pirnn["phase"] == phase]
        if "wind_est_rmse_inst" in sub.columns and sub["wind_est_rmse_inst"].notna().any():
            rmse_est.append(sub["wind_est_rmse_inst"].median())
            valid_phases.append(phase)
        else:
            rmse_est.append(np.nan)
            valid_phases.append(phase)
        if "wind_ekf2_rmse_inst" in sub.columns and sub["wind_ekf2_rmse_inst"].notna().any():
            rmse_ekf2.append(sub["wind_ekf2_rmse_inst"].median())
        else:
            rmse_ekf2.append(np.nan)

    x = np.arange(len(valid_phases))
    bw = 0.35
    bars1 = ax.bar(x - bw / 2, rmse_ekf2, bw, color=RED, alpha=0.75, label="EKF2 (horiz)")
    bars2 = ax.bar(x + bw / 2, rmse_est, bw, color=BLUE, alpha=0.75, label="PIRNN-AKF (3D)")
    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABEL[p] for p in valid_phases], rotation=20, ha="right")
    ax.set_ylabel("Median Inst. RMSE [m/s]")
    ax.set_title("Wind Estimation Accuracy by Phase")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 右图：EKF2 vs PIRNN-AKF vs 真值 水平分量时序（稳态+阵风段）
    ax = axes[1]
    eval_mask = df_pirnn["phase"].isin(["steady", "gust_light", "gust_strong"])
    sub = df_pirnn[eval_mask].copy()
    if not sub.empty:
        t = sub["runtime_s"]
        if "wind_gt_n" in sub.columns and "wind_gt_e" in sub.columns:
            wgt_mag = np.sqrt(sub["wind_gt_n"] ** 2 + sub["wind_gt_e"] ** 2)
            ax.plot(t, wgt_mag, color="k", lw=1.0, label="Truth |wind_h|", alpha=0.9)
        if "wind_ekf2_n" in sub.columns and "wind_ekf2_e" in sub.columns:
            wekf_mag = np.sqrt(sub["wind_ekf2_n"] ** 2 + sub["wind_ekf2_e"] ** 2)
            ax.plot(t, wekf_mag, color=RED, lw=0.9, ls="--", alpha=0.8, label="EKF2 |wind_h|")
        if "wind_est_n" in sub.columns and "wind_est_e" in sub.columns:
            west_mag = np.sqrt(sub["wind_est_n"] ** 2 + sub["wind_est_e"] ** 2)
            ax.plot(t, west_mag, color=BLUE, lw=1.0, alpha=0.9, label="PIRNN-AKF |wind_h|")
        ax.set_xlabel("Runtime [s]")
        ax.set_ylabel("Horiz. Wind Speed [m/s]")
        ax.set_title("Horizontal Wind Magnitude (Steady→Gust)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ 风估计精度图: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 数值摘要表
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(
    df_base: pd.DataFrame | None,
    df_pirnn: pd.DataFrame | None,
    output_path: str,
):
    lines = []
    lines.append("=" * 80)
    lines.append("  SITL 闭环对比摘要 — PIRNN-AKF vs. EKF2 Baseline")
    lines.append("=" * 80)

    metrics = [
        ("xtrack_error",         "|XTE|",           True,  "m"),
        ("airspeed_err",         "|Airspeed Err|",   True,  "m/s"),
        ("elevator_jitter",      "Elevator Jitter",  False, "PWM"),
        ("throttle_jitter",      "Throttle Jitter",  False, "PWM"),
        ("wind_est_rmse_inst",   "Wind Est RMSE",    False, "m/s"),
        ("wind_ekf2_rmse_inst",  "EKF2 Wind RMSE",   False, "m/s"),
    ]

    for phase in ["ALL"] + EVAL_PHASES:
        lines.append(f"\n── 阶段: {PHASE_LABEL.get(phase, phase)} ──")
        header = f"  {'Metric':<22} {'Baseline (mean±std)':>22}  {'PIRNN-AKF (mean±std)':>22}  {'Δ%':>8}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for col, label, take_abs, unit in metrics:
            def _stats(df):
                if df is None or col not in df.columns:
                    return None
                if phase == "ALL":
                    s = df.loc[df["phase"].isin(EVAL_PHASES), col].dropna()
                else:
                    s = df.loc[df["phase"] == phase, col].dropna()
                if take_abs:
                    s = s.abs()
                if len(s) == 0:
                    return None
                return s.mean(), s.std()

            s_base = _stats(df_base)
            s_pirnn = _stats(df_pirnn)

            b_str = f"{s_base[0]:.3f}±{s_base[1]:.3f} {unit}" if s_base else "  --"
            p_str = f"{s_pirnn[0]:.3f}±{s_pirnn[1]:.3f} {unit}" if s_pirnn else "  --"

            delta_str = "--"
            if s_base and s_pirnn and abs(s_base[0]) > 1e-9:
                delta_pct = (s_pirnn[0] - s_base[0]) / abs(s_base[0]) * 100
                sign = "+" if delta_pct > 0 else ""
                delta_str = f"{sign}{delta_pct:.1f}%"

            lines.append(
                f"  {label+'('+unit+')' :<22} {b_str:>22}  {p_str:>22}  {delta_str:>8}"
            )

    lines.append("\n" + "=" * 80)
    lines.append("注：Δ% = (PIRNN-AKF − Baseline) / |Baseline| × 100%；负值表示改善。")
    lines.append("    风场估计 RMSE 列仅在 pirnn_akf 模式下有数据。")
    lines.append("    EKF2 wind RMSE 仅含水平分量（EKF2 不输出垂向风估计）。")
    lines.append("=" * 80)

    text = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(text)
    print(f"  ✓ 数值摘要: {output_path}")
    print("\n" + text)


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="SITL 闭环实验结果分析脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        default=None,
        nargs="+",
        help="baseline 模式 CSV 路径（可 glob 匹配多文件，自动合并）",
    )
    parser.add_argument(
        "--pirnn",
        default=None,
        nargs="+",
        help="pirnn_akf 模式 CSV 路径（可 glob 匹配多文件，自动合并）",
    )
    parser.add_argument(
        "--output",
        default="SITL/comparison",
        help="输出文件路径前缀（不含扩展名），默认 SITL/comparison",
    )
    return parser.parse_args()


def _load_multi(paths: list[str] | None) -> pd.DataFrame | None:
    if not paths:
        return None
    dfs = []
    for p in paths:
        try:
            df = load_csv(p)
            dfs.append(df)
            print(f"  加载: {p}  ({len(df)} 行)")
        except Exception as exc:
            print(f"  ⚠ 跳过 {p}: {exc}")
    if not dfs:
        return None
    merged = pd.concat(dfs, ignore_index=True)
    # 按运行时间排序（多文件合并时重排）
    if "runtime_s" in merged.columns:
        merged.sort_values("runtime_s", inplace=True, ignore_index=True)
    return merged


def main():
    args = _parse_args()
    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print("\n── 加载数据 ──")
    df_base = _load_multi(args.baseline)
    df_pirnn = _load_multi(args.pirnn)

    if df_base is None and df_pirnn is None:
        print("错误：必须至少提供 --baseline 或 --pirnn 其中一个 CSV 文件。")
        sys.exit(1)

    # 补充派生列
    if df_base is not None:
        df_base = enrich(df_base)
    if df_pirnn is not None:
        df_pirnn = enrich(df_pirnn)

    print("\n── 生成图表 ──")
    plot_timeseries(df_base, df_pirnn, str(out_prefix) + "_timeseries.png")
    plot_boxplot(df_base, df_pirnn, str(out_prefix) + "_boxplot.png")
    if df_pirnn is not None:
        plot_wind_accuracy(df_pirnn, str(out_prefix) + "_wind_accuracy.png")

    print("\n── 数值摘要 ──")
    compute_summary(df_base, df_pirnn, str(out_prefix) + "_summary.txt")

    print(f"\n所有输出已保存至: {out_prefix}_*.{{png,txt}}")


if __name__ == "__main__":
    main()
