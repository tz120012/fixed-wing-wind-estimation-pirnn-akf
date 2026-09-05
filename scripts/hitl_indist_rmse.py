#!/usr/bin/env python3
"""HITL 单段（分布内）硬件 RMSE 统计。

用途：单相位 HITL 实验（wind_config.txt 单段稳态风，phase_name="unknown"）下，
不依赖 phase 标签，直接从对齐主表计算整段分布内风速估计 RMSE / 幅值比 / 时延，
产出可回填论文表 6「真实 HITL 硬件链路」列的指标。

输入：HITL/aligned/hitl_aligned_master.csv  （由 HITL/align_hitl_timestamps.py 生成）
输出：paper/tables/table_hitl_indist_rmse.md

真值有效样本：|truth| > 0.01（自动剔除 warmup / <50m 起飞段的零风），
并可用 --max-truth-mag 限定"分布内"上界（默认 4.57 m/s，与训练水平风上界一致）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
DEFAULT_MASTER = PROJ / "HITL/aligned/hitl_aligned_master.csv"
DEFAULT_OUT = PROJ / "paper/tables/table_hitl_indist_rmse.md"

TRUTH = ["wind_north_ms", "wind_east_ms", "wind_down_ms"]
EST_FUSED = ["wind_n", "wind_e", "wind_d"]
EST_RAW = ["wind_nn_n", "wind_nn_e", "wind_nn_d"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HITL 单段分布内硬件 RMSE")
    p.add_argument("--master", type=str, default=str(DEFAULT_MASTER),
                   help="对齐主表 CSV 路径")
    p.add_argument("--out", type=str, default=str(DEFAULT_OUT),
                   help="输出 Markdown 表路径")
    p.add_argument("--max-truth-mag", type=float, default=4.57,
                   help="分布内水平风上界 (m/s)；超过者视为 OOD，单独统计（默认 4.57）")
    p.add_argument("--min-truth-mag", type=float, default=0.01,
                   help="真值有效下界 (m/s)，剔除 warmup 零风段（默认 0.01）")
    return p.parse_args()


def _rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")


def _stats(df: pd.DataFrame, est_cols) -> dict:
    if not all(c in df.columns for c in est_cols):
        return {}
    e = df[est_cols].to_numpy(dtype=float)
    g = df[TRUTH].to_numpy(dtype=float)
    err = e - g
    est_h = np.sqrt(e[:, 0] ** 2 + e[:, 1] ** 2)
    est_3d = np.sqrt((e ** 2).sum(axis=1))
    tru_h = np.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2)
    tru_3d = np.sqrt((g ** 2).sum(axis=1))
    dir_e = np.degrees(np.arctan2(e[:, 1], e[:, 0]))
    dir_g = np.degrees(np.arctan2(g[:, 1], g[:, 0]))
    dir_err = ((dir_e - dir_g + 180) % 360) - 180
    return {
        "n": int(len(df)),
        "rmse_n": _rmse(err[:, 0]),
        "rmse_e": _rmse(err[:, 1]),
        "rmse_d": _rmse(err[:, 2]),
        "rmse_3d": float(np.sqrt(np.mean((err ** 2).sum(axis=1)))) if len(df) else float("nan"),
        "ratio_h": float(est_h.mean() / tru_h.mean()) if tru_h.mean() > 1e-6 else float("nan"),
        "ratio_3d": float(est_3d.mean() / tru_3d.mean()) if tru_3d.mean() > 1e-6 else float("nan"),
        "dir_mae": float(np.mean(np.abs(dir_err))),
        "truth_h_mean": float(tru_h.mean()),
    }


def main() -> None:
    args = parse_args()
    master = Path(args.master)
    if not master.exists():
        raise FileNotFoundError(f"未找到对齐主表: {master}\n请先运行 HITL/align_hitl_timestamps.py")

    df = pd.read_csv(master, low_memory=False)
    for c in TRUTH + EST_FUSED:
        if c not in df.columns:
            raise KeyError(f"主表缺少列: {c}")

    tru_h = np.sqrt(df["wind_north_ms"] ** 2 + df["wind_east_ms"] ** 2)
    tru_abs_sum = df[TRUTH].abs().sum(axis=1)
    valid = tru_abs_sum > args.min_truth_mag
    indist = valid & (tru_h <= args.max_truth_mag)
    ood = valid & (tru_h > args.max_truth_mag)

    df_indist = df[indist].copy()
    n_valid = int(valid.sum())
    n_indist = int(indist.sum())
    n_ood = int(ood.sum())

    print("=" * 70)
    print(" HITL 单段分布内硬件 RMSE")
    print("=" * 70)
    print(f"主表: {master}")
    print(f"总行数 {len(df):,} | 真值有效 {n_valid:,} | 分布内 {n_indist:,} | OOD(>{args.max_truth_mag}m/s) {n_ood:,}")
    if n_ood > 0:
        print(f"  ! 有 {n_ood:,} 帧真值水平风 > {args.max_truth_mag} m/s（超训练分布），已从分布内统计剔除。")
    if n_indist == 0:
        print("  ✗ 无分布内有效样本，无法统计。检查风场配置/起飞是否成功。")
        return

    fused = _stats(df_indist, EST_FUSED)
    raw = _stats(df_indist, EST_RAW)

    # 时延
    inf = df_indist["inference_ms"].dropna() if "inference_ms" in df_indist else pd.Series(dtype=float)
    e2e = df_indist["latency_e2e_ms"].dropna() if "latency_e2e_ms" in df_indist else pd.Series(dtype=float)

    def q(s, p):
        return float(np.percentile(s, p)) if len(s) else float("nan")

    # ---------- 控制台输出 ----------
    print(f"\n分布内真值水平风均值: {fused.get('truth_h_mean', float('nan')):.3f} m/s")
    print("\n[融合估计 wind_n/e/d (AKF+EMA)]")
    print(f"  RMSE  N={fused['rmse_n']:.3f}  E={fused['rmse_e']:.3f}  D={fused['rmse_d']:.3f}  "
          f"| 3D均值={fused['rmse_3d']:.3f} m/s")
    print(f"  幅值比 水平={fused['ratio_h']:.3f}  3D={fused['ratio_3d']:.3f}  | 风向MAE={fused['dir_mae']:.1f}°")
    if raw:
        print("\n[原始 PI-GRU 输出 wind_nn_n/e/d (融合前)]")
        print(f"  RMSE  N={raw['rmse_n']:.3f}  E={raw['rmse_e']:.3f}  D={raw['rmse_d']:.3f}  "
              f"| 3D均值={raw['rmse_3d']:.3f} m/s | 幅值比水平={raw['ratio_h']:.3f}")
    print(f"\n时延: 推理 mean={inf.mean():.2f} p95={q(inf,95):.2f} ms | "
          f"端到端 mean={e2e.mean():.2f} p95={q(e2e,95):.2f} ms" if len(inf) else "\n时延: 无 inference_ms 列")

    # ---------- 写 Markdown ----------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# HITL 单段分布内硬件 RMSE（可回填论文表 6「真实 HITL 硬件链路」列）\n")
    lines.append(f"- 对齐主表: `{master.name}`")
    lines.append(f"- 样本: 总 {len(df):,} 行，真值有效 {n_valid:,}，**分布内 {n_indist:,}**"
                 f"（水平风 ≤ {args.max_truth_mag} m/s），OOD 剔除 {n_ood:,}")
    lines.append(f"- 分布内真值水平风均值: {fused['truth_h_mean']:.3f} m/s\n")
    lines.append("| 指标 | 融合估计 (AKF+EMA) | 原始 PI-GRU |")
    lines.append("|---|---|---|")
    def cell(d, k, fmt="{:.3f}"):
        return fmt.format(d[k]) if d and k in d else "—"
    lines.append(f"| RMSE$_N$ (m/s) | {cell(fused,'rmse_n')} | {cell(raw,'rmse_n')} |")
    lines.append(f"| RMSE$_E$ (m/s) | {cell(fused,'rmse_e')} | {cell(raw,'rmse_e')} |")
    lines.append(f"| RMSE$_D$ (m/s) | {cell(fused,'rmse_d')} | {cell(raw,'rmse_d')} |")
    lines.append(f"| 均值 3D RMSE (m/s) | {cell(fused,'rmse_3d')} | {cell(raw,'rmse_3d')} |")
    lines.append(f"| 幅值比（水平） | {cell(fused,'ratio_h')} | {cell(raw,'ratio_h')} |")
    lines.append(f"| 风向 MAE (°) | {cell(fused,'dir_mae','{:.1f}')} | {cell(raw,'dir_mae','{:.1f}')} |")
    if len(inf):
        lines.append(f"| 单步推理延迟 mean/p95 (ms) | {inf.mean():.2f} / {q(inf,95):.2f} | 同左 |")
    if len(e2e):
        lines.append(f"| 端到端延迟 mean/p95 (ms) | {e2e.mean():.2f} / {q(e2e,95):.2f} | 同左 |")
    lines.append("")
    lines.append("> RMSE 定义：`sqrt(mean(Σ(est−truth)²))`，仅在真值有效且分布内的样本上统计。")
    lines.append("> 幅值比 = 估计水平风均值 / 真值水平风均值（1.0 为无系统性欠/过估）。")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✓ 已写出: {out}")


if __name__ == "__main__":
    main()
