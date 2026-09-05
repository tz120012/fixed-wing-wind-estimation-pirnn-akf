"""验证 HITL 会话（Raspberry Pi5 端）与仿真主机风真值日志（JSBSim 端）
是否共享同一个 wall-clock 时间基准。

思路：不直接假设两边时钟同步，而是用一个双方都能独立观测到的物理事件
做交叉验证锚点——仿真主机侧标记的 "packet_loss" 阶段（人为丢包故障注入）
理应在 Pi5 侧表现为通信延迟升高/采样间隔变大。如果按 wall_time_usec ==
timestamp*1e6 直接对齐后，Pi5 侧在对应时间窗口确实出现了通信异常，
说明两边时钟基本同源可信；否则说明存在偏移，需要先估计偏移量或放弃
事后对齐方案。

用法：
    .venv/bin/python scripts/verify_hitl_clock_alignment.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wind_truth", default="HITL/JSBSim_truth_wind/wind_truth_20260416_204152.csv")
    ap.add_argument("--hitl_data", default="HITL/logs_in_rasbpi/hitl_data_20260416_214252.csv")
    args = ap.parse_args()

    wt = pd.read_csv(args.wind_truth)
    hd = pd.read_csv(args.hitl_data)

    wt["t_s"] = wt["wall_time_usec"] / 1e6
    hd["t_s"] = hd["timestamp"]  # 已经是秒级 unix time

    t0, t1 = hd["t_s"].min(), hd["t_s"].max()
    print(f"[Pi5] hitl_data 会话窗口: {t0:.3f} .. {t1:.3f} (持续 {t1 - t0:.1f}s, {len(hd)} 帧)")

    wt_win = wt[(wt["t_s"] >= t0) & (wt["t_s"] <= t1)]
    print(f"[Sim] 对应窗口内 wind_truth 行数: {len(wt_win)}")
    if len(wt_win) == 0:
        print("!! 两边时间戳按直接对齐完全不重叠，说明零偏移假设不成立。")
        return

    print("[Sim] 对应窗口内 phase 分布:", wt_win["phase_name"].value_counts().to_dict())

    # 找出 sim 侧 packet_loss 阶段的时间区间（可能不止一段）
    is_pl = wt_win["phase_name"].eq("packet_loss")
    if not is_pl.any():
        print("窗口内没有 packet_loss 阶段，换用 gust_strong 做锚点。")
        is_pl = wt_win["phase_name"].eq("gust_strong")
        anchor_name = "gust_strong"
    else:
        anchor_name = "packet_loss"

    # 提取锚点阶段的起止时间（取最长连续段）
    idx = np.where(is_pl.values)[0]
    # 分段
    splits = np.where(np.diff(idx) > 1)[0]
    segments = np.split(idx, splits + 1)
    segments.sort(key=len, reverse=True)
    seg = segments[0]
    t_anchor = wt_win["t_s"].values
    seg_t0, seg_t1 = t_anchor[seg[0]], t_anchor[seg[-1]]
    print(f"\n[锚点] 最长的 '{anchor_name}' 段: {seg_t0:.3f} .. {seg_t1:.3f} ({seg_t1 - seg_t0:.1f}s)")

    # 在 Pi5 数据里看这段时间窗口的通信/延迟指标是否有异常
    hd_in = hd[(hd["t_s"] >= seg_t0) & (hd["t_s"] <= seg_t1)]
    hd_out = hd[(hd["t_s"] < seg_t0) | (hd["t_s"] > seg_t1)]

    print(f"\n[Pi5] 锚点窗口内帧数: {len(hd_in)}  窗口外帧数: {len(hd_out)}")

    # 采样间隔（boot_time_us 差分，单位转 ms）
    def gap_stats(df: pd.DataFrame, label: str) -> None:
        if len(df) < 2:
            print(f"  {label}: 帧数不足，跳过")
            return
        dt_ms = np.diff(np.sort(df["boot_time_us"].values)) / 1e3
        print(
            f"  {label}: n={len(df)} 采样间隔 mean={dt_ms.mean():.2f}ms "
            f"p95={np.percentile(dt_ms, 95):.2f}ms max={dt_ms.max():.2f}ms"
        )

    gap_stats(hd_in, f"{anchor_name}窗口内")
    gap_stats(hd_out, "窗口外(其余)")

    if "latency_e2e_ms" in hd.columns:
        print(
            f"\n  latency_e2e_ms: 窗口内均值={hd_in['latency_e2e_ms'].mean():.2f}ms  "
            f"窗口外均值={hd_out['latency_e2e_ms'].mean():.2f}ms"
        )

    print(
        "\n判读标准：若 packet_loss/gust_strong 窗口内的采样间隔或延迟明显"
        "高于窗口外（例如相差 >2x 或有肉眼可见的跳变簇），说明两边时钟基本"
        "同源、直接按 wall_time_usec==timestamp*1e6 对齐是可信的；"
        "若窗口内外没有可辨的差异，说明零偏移假设不成立，不能直接后处理"
        "拼接，需要重新采集并显式做时钟同步（NTP）。"
    )


if __name__ == "__main__":
    main()
