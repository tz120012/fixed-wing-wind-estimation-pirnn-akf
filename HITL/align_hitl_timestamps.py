#!/usr/bin/env python3
"""对齐 HITL 三路数据时间戳：JSBSim truth / 树莓派在线日志 / PX4 ULog。

对齐策略：
1. JSBSim truth ↔ raspi：使用绝对墙钟时间
   - JSBSim: wall_time_usec (us since epoch)
   - raspi:   timestamp (s since epoch)
2. raspi ↔ PX4 ULog：使用飞控开机时间
   - raspi: t_fc_send_us / boot_time_us (us since PX4 boot)
   - PX4:   ULog topic timestamp (us since PX4 boot)

输出：
- HITL/aligned/hitl_aligned_master.csv
- HITL/aligned/hitl_alignment_summary.md

说明：
- 以 raspi 日志为主表，每一行挂接最近邻 JSBSim 真值与 PX4 topic。
- PX4 ULog 通过系统命令 `ulog2csv` 导出关键 topic，无需 pyulog Python 包。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HITL_DIR = PROJECT_ROOT / "HITL"
TRUTH_DIR = HITL_DIR / "JSBSim_truth_wind"
RASPI_DIR = HITL_DIR / "logs_in_rasbpi"
ULOG_DIR = HITL_DIR / "PX4_ulog"
EXPORT_DIR = ULOG_DIR / "csv_export"
OUT_DIR = HITL_DIR / "aligned"

PX4_TOPICS: Dict[str, Dict[str, object]] = {
    "vehicle_local_position": {
        "columns": ["timestamp", "vx", "vy", "vz", "ax", "ay", "az", "heading", "dist_bottom"],
        "tolerance_us": 60_000,
    },
    "vehicle_gps_position": {
        "columns": ["timestamp", "vel_n_m_s", "vel_e_m_s", "vel_d_m_s", "lat", "lon", "alt", "time_utc_usec"],
        "tolerance_us": 300_000,
    },
    "airspeed": {
        "columns": ["timestamp", "indicated_airspeed_m_s", "true_airspeed_m_s", "confidence"],
        "tolerance_us": 600_000,
    },
    "vehicle_attitude": {
        "columns": ["timestamp", "q[0]", "q[1]", "q[2]", "q[3]"],
        "tolerance_us": 30_000,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Align HITL timestamps across JSBSim / raspi / PX4")
    p.add_argument("--truth", type=str, default="", help="Override truth CSV path")
    p.add_argument("--ulog", type=str, default="", help="Override PX4 ULog path")
    p.add_argument("--outdir", type=str, default=str(OUT_DIR), help="Output directory")
    p.add_argument("--force-export", action="store_true", help="Force re-export PX4 ULog CSVs")
    p.add_argument("--truth-tol-ms", type=float, default=50.0, help="JSBSim↔raspi nearest tolerance in ms")
    return p.parse_args()


def _resolve_truth(path_arg: str) -> Path:
    if path_arg:
        p = Path(path_arg)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    files = sorted(TRUTH_DIR.glob("wind_truth_*.csv"))
    if not files:
        raise FileNotFoundError(f"未找到真值 CSV: {TRUTH_DIR}")
    return files[-1]


def _resolve_ulog(path_arg: str) -> Path:
    if path_arg:
        p = Path(path_arg)
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()
    files = sorted(ULOG_DIR.glob("*.ulg"))
    if not files:
        raise FileNotFoundError(f"未找到 PX4 ULog: {ULOG_DIR}")
    return files[0]


def export_px4_topics(ulog_path: Path, export_dir: Path, force: bool = False) -> Dict[str, Path]:
    export_dir.mkdir(parents=True, exist_ok=True)
    ulog2csv = shutil.which("ulog2csv")
    if not ulog2csv:
        raise RuntimeError("未找到 ulog2csv，请先安装 pyulog 命令行工具")

    stem = ulog_path.stem
    expected = {
        topic: export_dir / f"{stem}_{topic}_0.csv"
        for topic in PX4_TOPICS
    }

    need_export = force or any(not p.exists() for p in expected.values())
    if need_export:
        cmd = [
            ulog2csv,
            str(ulog_path),
            "-m", ",".join(PX4_TOPICS.keys()),
            "-o", str(export_dir),
        ]
        subprocess.run(cmd, check=True)

    missing = [topic for topic, path in expected.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少导出的 PX4 topic CSV: {missing}")
    return expected


def load_truth(truth_path: Path) -> pd.DataFrame:
    truth = pd.read_csv(truth_path).sort_values("wall_time_usec").reset_index(drop=True)
    truth["jsbsim_wall_sec"] = truth["wall_time_usec"] / 1e6
    truth = truth.rename(columns={
        "wall_time_usec": "jsbsim_wall_time_usec",
        "sim_time_s": "jsbsim_sim_time_s",
        "phase_name": "jsbsim_phase_name",
    })
    return truth


def load_raspi() -> pd.DataFrame:
    raspi_files = sorted(RASPI_DIR.glob("hitl_data_*.csv"))
    if not raspi_files:
        raise FileNotFoundError(f"未找到 raspi CSV: {RASPI_DIR}")

    chunks: List[pd.DataFrame] = []
    for session_idx, path in enumerate(raspi_files, start=1):
        df = pd.read_csv(path).sort_values("timestamp").reset_index(drop=True)
        if "phase" in df.columns:
            df = df.rename(columns={"phase": "raspi_phase"})
        backend = "PyTorch CPU" if df["inference_ms"].mean() > 10 else "ONNX CPU"
        df["session_file"] = path.name
        df["session_idx"] = session_idx
        df["backend"] = backend
        chunks.append(df)

    raspi = pd.concat(chunks, ignore_index=True)
    raspi = raspi.sort_values("timestamp").reset_index(drop=True)
    raspi["raspi_wall_sec"] = raspi["timestamp"]
    raspi["px4_boot_us"] = raspi["t_fc_send_us"]
    raspi["px4_boot_sec"] = raspi["px4_boot_us"] / 1e6
    raspi["exp_wall_sec"] = raspi["raspi_wall_sec"] - raspi["raspi_wall_sec"].iloc[0]
    return raspi


def align_truth_to_raspi(raspi: pd.DataFrame, truth: pd.DataFrame, truth_tol_ms: float) -> pd.DataFrame:
    truth_cols = [
        "jsbsim_wall_sec", "jsbsim_wall_time_usec", "jsbsim_sim_time_s", "jsbsim_phase_name",
        "wind_north_ms", "wind_east_ms", "wind_down_ms",
        "total_wind_north_ms", "total_wind_east_ms", "total_wind_down_ms",
        "alt_agl_m", "airspeed_kt", "groundspeed_kt",
    ]
    aligned = pd.merge_asof(
        raspi.sort_values("raspi_wall_sec"),
        truth[truth_cols].sort_values("jsbsim_wall_sec"),
        left_on="raspi_wall_sec",
        right_on="jsbsim_wall_sec",
        direction="nearest",
        tolerance=truth_tol_ms / 1000.0,
    )
    aligned["align_truth_dt_ms"] = (aligned["raspi_wall_sec"] - aligned["jsbsim_wall_sec"]) * 1000.0
    aligned["phase_name"] = aligned["jsbsim_phase_name"]
    aligned["phase_source"] = np.where(
        aligned["phase_name"].notna(),
        "jsbsim_truth_wind",
        np.nan,
    )
    if "raspi_phase" in aligned.columns:
        aligned["phase_match"] = np.where(
            aligned["jsbsim_phase_name"].notna() & aligned["raspi_phase"].notna(),
            aligned["jsbsim_phase_name"] == aligned["raspi_phase"],
            np.nan,
        )
    else:
        aligned["phase_match"] = np.nan
    return aligned


def align_px4_topic(master: pd.DataFrame, topic_name: str, topic_path: Path, columns: List[str], tolerance_us: int) -> pd.DataFrame:
    topic = pd.read_csv(topic_path, usecols=columns).sort_values("timestamp").reset_index(drop=True)
    topic = topic.rename(columns={"timestamp": f"px4_{topic_name}_timestamp_us"})
    rename_map = {
        c: f"px4_{topic_name}_{c}" for c in topic.columns if c != f"px4_{topic_name}_timestamp_us"
    }
    topic = topic.rename(columns=rename_map)

    merged = pd.merge_asof(
        master.sort_values("px4_boot_us"),
        topic,
        left_on="px4_boot_us",
        right_on=f"px4_{topic_name}_timestamp_us",
        direction="nearest",
        tolerance=tolerance_us,
    )
    merged[f"align_{topic_name}_dt_ms"] = (
        merged["px4_boot_us"] - merged[f"px4_{topic_name}_timestamp_us"]
    ) / 1000.0
    return merged


def build_summary(master: pd.DataFrame, truth_path: Path, ulog_path: Path, out_path: Path) -> None:
    def _fmt_stats(series: pd.Series) -> str:
        s = series.dropna().astype(float)
        if len(s) == 0:
            return "无匹配"
        q = np.percentile(np.abs(s), [50, 95, 99])
        return (
            f"count={len(s)}, mean={s.mean():.3f} ms, "
            f"median_abs={q[0]:.3f} ms, p95_abs={q[1]:.3f} ms, p99_abs={q[2]:.3f} ms"
        )

    lines = []
    lines.append("# HITL 时间戳对齐摘要\n")
    lines.append(f"- **JSBSim truth**: `{truth_path.name}`")
    lines.append(f"- **PX4 ULog**: `{ulog_path.name}`")
    lines.append(f"- **raspi rows**: {len(master):,}")
    lines.append(f"- **sessions**: {master['session_file'].nunique()}  ({', '.join(sorted(master['backend'].unique()))})\n")

    lines.append("## 总体覆盖率\n")
    lines.append(f"- **truth 匹配率**: {master['jsbsim_sim_time_s'].notna().mean() * 100:.2f}%")
    lines.append(f"- **truth phase 覆盖率**: {master['phase_name'].notna().mean() * 100:.2f}%")
    if 'raspi_phase' in master.columns and master['phase_match'].dropna().shape[0] > 0:
        lines.append(f"- **raspi phase 一致率（仅审计）**: {master['phase_match'].dropna().mean() * 100:.2f}%")
    for topic in PX4_TOPICS:
        col = f"px4_{topic}_timestamp_us"
        lines.append(f"- **{topic} 匹配率**: {master[col].notna().mean() * 100:.2f}%")
    lines.append("")

    lines.append("## 时间差统计\n")
    lines.append(f"- **truth vs raspi**: {_fmt_stats(master['align_truth_dt_ms'])}")
    for topic in PX4_TOPICS:
        lines.append(f"- **raspi vs {topic}**: {_fmt_stats(master[f'align_{topic}_dt_ms'])}")
    lines.append("\n## Session 范围\n")
    for session_file, sub in master.groupby("session_file", sort=False):
        lines.append(
            f"- **{session_file}** ({sub['backend'].iloc[0]}): "
            f"wall={sub['raspi_wall_sec'].min():.3f}~{sub['raspi_wall_sec'].max():.3f} s, "
            f"boot={int(sub['px4_boot_us'].min())}~{int(sub['px4_boot_us'].max())} us, "
            f"sim={sub['jsbsim_sim_time_s'].min():.3f}~{sub['jsbsim_sim_time_s'].max():.3f} s"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    truth_path = _resolve_truth(args.truth)
    ulog_path = _resolve_ulog(args.ulog)
    out_dir = Path(args.outdir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] truth: {truth_path}")
    print(f"[2/5] ulog:  {ulog_path}")
    topic_paths = export_px4_topics(ulog_path, EXPORT_DIR, force=args.force_export)
    print(f"[3/5] exported PX4 topics to: {EXPORT_DIR}")

    truth = load_truth(truth_path)
    raspi = load_raspi()
    master = align_truth_to_raspi(raspi, truth, truth_tol_ms=args.truth_tol_ms)

    for topic_name, cfg in PX4_TOPICS.items():
        master = align_px4_topic(
            master,
            topic_name=topic_name,
            topic_path=topic_paths[topic_name],
            columns=list(cfg["columns"]),
            tolerance_us=int(cfg["tolerance_us"]),
        )
        print(f"[4/5] aligned PX4 topic: {topic_name}")

    master = master.sort_values(["session_idx", "runtime_s", "px4_boot_us"]).reset_index(drop=True)
    csv_path = out_dir / "hitl_aligned_master.csv"
    md_path = out_dir / "hitl_alignment_summary.md"
    master.to_csv(csv_path, index=False)
    build_summary(master, truth_path, ulog_path, md_path)

    print(f"[5/5] saved master CSV: {csv_path}")
    print(f"[5/5] saved summary   : {md_path}")


if __name__ == "__main__":
    main()
