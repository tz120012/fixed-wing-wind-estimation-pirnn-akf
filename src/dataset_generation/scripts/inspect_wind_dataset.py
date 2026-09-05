#!/usr/bin/env python3
"""
inspect_wind_dataset.py

对 `datasets-*.json` / `flight_*.json` 做风场验收与可视化：
1. 单文件模式：输出关键统计、风险提示，并保存 4 联图 PNG
2. 目录模式：递归扫描数据文件，输出汇总 CSV，并可额外保存若干高风险样本图

说明：
- 日志中的 `wind_north/east/down` 是逐时刻真值风（常值风 + gust）；
- 湍流未逐样本写回风真值列，因此本脚本用姿态/地速/控制抖动侧面观察湍流是否过猛；
- 评价规则偏向“小固定翼/泡沫机”的经验筛查，不等价于严格飞行力学判定。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SEVERITY_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}
DEFAULT_PATTERNS = ("datasets-*.json", "flight_*.json")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_metadata_path(data_path: Path) -> Optional[Path]:
    if data_path.name.endswith("_metadata.json"):
        return data_path
    candidate = data_path.with_name(f"{data_path.stem}_metadata.json")
    return candidate if candidate.exists() else None


def list_data_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        if input_path.name.endswith("_metadata.json"):
            raise ValueError("请输入数据 JSON，而不是 _metadata.json")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"路径不存在: {input_path}")

    files: List[Path] = []
    for pattern in DEFAULT_PATTERNS:
        for path in input_path.rglob(pattern):
            if path.name.endswith("_metadata.json"):
                continue
            files.append(path)
    return sorted(set(files))


def finite_array(records: Sequence[dict], key: str) -> np.ndarray:
    values = []
    for row in records:
        value = row.get(key, np.nan)
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=float)


def finite_values(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def safe_percentile(arr: np.ndarray, q: float) -> float:
    vals = finite_values(arr)
    if vals.size == 0:
        return float("nan")
    return float(np.percentile(vals, q))


def safe_mean(arr: np.ndarray) -> float:
    vals = finite_values(arr)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals))


def safe_min(arr: np.ndarray) -> float:
    vals = finite_values(arr)
    if vals.size == 0:
        return float("nan")
    return float(np.min(vals))


def safe_max(arr: np.ndarray) -> float:
    vals = finite_values(arr)
    if vals.size == 0:
        return float("nan")
    return float(np.max(vals))


def safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or abs(den) < 1e-9:
        return float("nan")
    return num / den


def fraction_over_threshold(arr: np.ndarray, threshold: float, absolute: bool = False) -> float:
    vals = finite_values(np.abs(arr) if absolute else arr)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals > threshold))


def fraction_between(arr: np.ndarray, low: float, high: float) -> float:
    vals = finite_values(arr)
    if vals.size == 0:
        return float("nan")
    return float(np.mean((vals < low) | (vals > high)))


def summarize_gust_window(metadata: dict) -> Optional[Tuple[float, float]]:
    gust = metadata.get("gust") if isinstance(metadata, dict) else None
    if not isinstance(gust, dict):
        return None
    start = float(gust.get("start_time", 0.0) or 0.0)
    duration = float(gust.get("duration", 0.0) or 0.0)
    if duration <= 0:
        return None
    return start, start + duration


def classify_risk(metrics: dict) -> Tuple[str, List[str], List[str]]:
    warns: List[str] = []
    fails: List[str] = []

    peak_ratio = metrics.get("peak_wind_ratio", float("nan"))
    base_ratio = metrics.get("base_wind_ratio", float("nan"))
    vertical_ratio = metrics.get("vertical_wind_ratio", float("nan"))
    airspeed_drop_ratio = metrics.get("airspeed_p05_over_median", float("nan"))
    min_gs_ratio = metrics.get("min_groundspeed_over_median_airspeed", float("nan"))
    roll_p95 = metrics.get("roll_abs_p95_deg", float("nan"))
    pitch_p95 = metrics.get("pitch_abs_p95_deg", float("nan"))
    ctrl_sat = metrics.get("ctrl_sat_max_fraction", float("nan"))
    sample_rate = metrics.get("effective_rate_hz", float("nan"))
    coverage_ratio = metrics.get("coverage_ratio", float("nan"))

    if math.isfinite(peak_ratio):
        if peak_ratio > 0.50:
            fails.append(f"峰值合成风/空速比过高 ({peak_ratio:.2f})")
        elif peak_ratio > 0.35:
            warns.append(f"峰值合成风/空速比偏高 ({peak_ratio:.2f})")
    if math.isfinite(base_ratio) and base_ratio > 0.30:
        warns.append(f"常值风/空速比偏高 ({base_ratio:.2f})")

    if math.isfinite(vertical_ratio):
        if vertical_ratio > 0.35:
            fails.append(f"垂直风占水平风比例过高 ({vertical_ratio:.2f})")
        elif vertical_ratio > 0.20:
            warns.append(f"垂直风占水平风比例偏高 ({vertical_ratio:.2f})")

    if math.isfinite(airspeed_drop_ratio):
        if airspeed_drop_ratio < 0.50:
            fails.append(f"空速低谷过深 (P05/中位={airspeed_drop_ratio:.2f})")
        elif airspeed_drop_ratio < 0.65:
            warns.append(f"空速有明显下探 (P05/中位={airspeed_drop_ratio:.2f})")

    if math.isfinite(min_gs_ratio):
        if min_gs_ratio < 0.15:
            fails.append(f"最小地速过低 (minGS/中位空速={min_gs_ratio:.2f})")
        elif min_gs_ratio < 0.25:
            warns.append(f"地速明显偏低 (minGS/中位空速={min_gs_ratio:.2f})")

    if math.isfinite(roll_p95):
        if roll_p95 > 55:
            fails.append(f"横滚响应过猛 (|roll| P95={roll_p95:.1f}°)")
        elif roll_p95 > 45:
            warns.append(f"横滚响应偏猛 (|roll| P95={roll_p95:.1f}°)")

    if math.isfinite(pitch_p95):
        if pitch_p95 > 22:
            fails.append(f"俯仰响应过猛 (|pitch| P95={pitch_p95:.1f}°)")
        elif pitch_p95 > 15:
            warns.append(f"俯仰响应偏猛 (|pitch| P95={pitch_p95:.1f}°)")

    if math.isfinite(ctrl_sat):
        if ctrl_sat > 0.20:
            fails.append(f"控制量长时间接近饱和 ({ctrl_sat:.1%})")
        elif ctrl_sat > 0.08:
            warns.append(f"控制量接近饱和较频繁 ({ctrl_sat:.1%})")

    if math.isfinite(sample_rate) and sample_rate < 35:
        warns.append(f"有效采样率偏低 ({sample_rate:.1f} Hz)")

    if math.isfinite(coverage_ratio) and coverage_ratio < 0.85:
        warns.append(f"时间覆盖不足 ({coverage_ratio:.1%})")

    verdict = "FAIL" if fails else ("WARN" if warns else "PASS")
    return verdict, warns, fails


def compute_metrics(records: Sequence[dict], metadata: Optional[dict], data_path: Path) -> dict:
    if not records:
        raise ValueError(f"数据为空: {data_path}")

    meta = metadata or {}
    t = finite_array(records, "timestamp")
    airspeed = finite_array(records, "airspeed_m_s")
    groundspeed = finite_array(records, "groundspeed_m_s")
    wn = finite_array(records, "wind_north")
    we = finite_array(records, "wind_east")
    wd = finite_array(records, "wind_down")
    roll = finite_array(records, "roll_deg")
    pitch = finite_array(records, "pitch_deg")
    yaw = finite_array(records, "yaw_deg")
    roll_ctrl = finite_array(records, "roll_ctrl")
    pitch_ctrl = finite_array(records, "pitch_ctrl")
    yaw_ctrl = finite_array(records, "yaw_ctrl")
    throttle = finite_array(records, "throttle_ctrl")

    dt = np.diff(finite_values(t))
    coverage_sec = max(0.0, safe_max(t) - safe_min(t)) if finite_values(t).size >= 2 else 0.0
    sample_count = len(records)
    requested_duration = float(meta.get("duration", np.nan) or np.nan)
    effective_rate = sample_count / requested_duration if math.isfinite(requested_duration) and requested_duration > 1e-9 else sample_count / max(coverage_sec, 1e-9)

    wind_mag = np.sqrt(wn ** 2 + we ** 2 + wd ** 2)
    horizontal_wind_mag = np.sqrt(wn ** 2 + we ** 2)

    base_wn = float(meta.get("wind_north", safe_mean(wn)) or 0.0)
    base_we = float(meta.get("wind_east", safe_mean(we)) or 0.0)
    base_wd = float(meta.get("wind_down", safe_mean(wd)) or 0.0)
    base_horiz_mag = math.sqrt(base_wn ** 2 + base_we ** 2)
    base_wind_mag = math.sqrt(base_wn ** 2 + base_we ** 2 + base_wd ** 2)

    gust = meta.get("gust") if isinstance(meta.get("gust"), dict) else {}
    gust_mag = float(gust.get("magnitude", 0.0) or 0.0)
    gust_window = summarize_gust_window(meta)

    median_airspeed = safe_percentile(airspeed, 50)
    p05_airspeed = safe_percentile(airspeed, 5)
    min_groundspeed = safe_min(groundspeed)
    roll_abs_p95 = safe_percentile(np.abs(roll), 95)
    pitch_abs_p95 = safe_percentile(np.abs(pitch), 95)

    roll_sat = fraction_over_threshold(roll_ctrl, 0.95, absolute=True)
    pitch_sat = fraction_over_threshold(pitch_ctrl, 0.95, absolute=True)
    yaw_sat = fraction_over_threshold(yaw_ctrl, 0.95, absolute=True)
    throttle_sat = fraction_between(throttle, 0.05, 0.95)
    ctrl_sat_values = [x for x in (roll_sat, pitch_sat, yaw_sat, throttle_sat) if math.isfinite(x)]
    ctrl_sat_max = max(ctrl_sat_values) if ctrl_sat_values else float("nan")

    metrics = {
        "path": str(data_path),
        "file": data_path.name,
        "dataset_type": str(meta.get("dataset_type", "")),
        "config_type": str(meta.get("config_type", "")),
        "maneuver_type": str(meta.get("maneuver_type", "")),
        "sample_count": sample_count,
        "duration_sec": requested_duration,
        "coverage_sec": coverage_sec,
        "coverage_ratio": safe_ratio(coverage_sec, requested_duration),
        "effective_rate_hz": effective_rate,
        "median_dt_sec": safe_percentile(dt, 50),
        "median_airspeed_m_s": median_airspeed,
        "p05_airspeed_m_s": p05_airspeed,
        "min_airspeed_m_s": safe_min(airspeed),
        "mean_groundspeed_m_s": safe_mean(groundspeed),
        "min_groundspeed_m_s": min_groundspeed,
        "base_wind_mag_m_s": base_wind_mag,
        "base_horizontal_wind_m_s": base_horiz_mag,
        "base_wind_down_m_s": base_wd,
        "gust_mag_m_s": gust_mag,
        "max_logged_wind_mag_m_s": safe_max(wind_mag),
        "max_logged_horizontal_wind_m_s": safe_max(horizontal_wind_mag),
        "base_wind_ratio": safe_ratio(base_wind_mag, median_airspeed),
        "peak_wind_ratio": safe_ratio(base_wind_mag + gust_mag, median_airspeed),
        "vertical_wind_ratio": safe_ratio(abs(base_wd), max(base_horiz_mag, 1e-9)),
        "airspeed_p05_over_median": safe_ratio(p05_airspeed, median_airspeed),
        "min_groundspeed_over_median_airspeed": safe_ratio(min_groundspeed, median_airspeed),
        "roll_abs_p95_deg": roll_abs_p95,
        "pitch_abs_p95_deg": pitch_abs_p95,
        "yaw_span_deg": safe_max(yaw) - safe_min(yaw),
        "roll_ctrl_sat_fraction": roll_sat,
        "pitch_ctrl_sat_fraction": pitch_sat,
        "yaw_ctrl_sat_fraction": yaw_sat,
        "throttle_ctrl_sat_fraction": throttle_sat,
        "ctrl_sat_max_fraction": ctrl_sat_max,
        "gust_start_sec": gust_window[0] if gust_window else float("nan"),
        "gust_end_sec": gust_window[1] if gust_window else float("nan"),
        "has_gust_metadata": bool(gust_window),
    }

    verdict, warns, fails = classify_risk(metrics)
    metrics["verdict"] = verdict
    metrics["warnings"] = warns
    metrics["failures"] = fails
    metrics["flags"] = fails + warns
    return metrics


def fmt_num(value: float, digits: int = 3) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def print_single_summary(metrics: dict) -> None:
    print(f"\n=== 风场验收: {metrics['file']} ===")
    print(f"结论: {metrics['verdict']}")
    print(
        "基础统计: "
        f"samples={metrics['sample_count']}, "
        f"coverage={fmt_num(metrics['coverage_sec'], 1)}s, "
        f"rate={fmt_num(metrics['effective_rate_hz'], 1)}Hz, "
        f"config={metrics.get('config_type') or '-'}, "
        f"maneuver={metrics.get('maneuver_type') or '-'}"
    )
    print(
        "风场强度: "
        f"base={fmt_num(metrics['base_wind_mag_m_s'], 2)}m/s, "
        f"gust={fmt_num(metrics['gust_mag_m_s'], 2)}m/s, "
        f"base_ratio={fmt_num(metrics['base_wind_ratio'], 2)}, "
        f"peak_ratio={fmt_num(metrics['peak_wind_ratio'], 2)}, "
        f"vertical_ratio={fmt_num(metrics['vertical_wind_ratio'], 2)}"
    )
    print(
        "飞行响应: "
        f"median_airspeed={fmt_num(metrics['median_airspeed_m_s'], 2)}m/s, "
        f"p05/median={fmt_num(metrics['airspeed_p05_over_median'], 2)}, "
        f"minGS/medianV={fmt_num(metrics['min_groundspeed_over_median_airspeed'], 2)}, "
        f"|roll|P95={fmt_num(metrics['roll_abs_p95_deg'], 1)}°, "
        f"|pitch|P95={fmt_num(metrics['pitch_abs_p95_deg'], 1)}°, "
        f"ctrl_sat_max={fmt_num(metrics['ctrl_sat_max_fraction'] * 100.0, 1)}%"
    )

    if metrics["flags"]:
        print("风险提示:")
        for item in metrics["flags"]:
            prefix = "  -"
            print(f"{prefix} {item}")
    else:
        print("风险提示: 无明显异常")


def render_plot(records: Sequence[dict], metadata: Optional[dict], metrics: dict, output_png: Path) -> None:
    meta = metadata or {}
    t = finite_array(records, "timestamp")
    wn = finite_array(records, "wind_north")
    we = finite_array(records, "wind_east")
    wd = finite_array(records, "wind_down")
    airspeed = finite_array(records, "airspeed_m_s")
    groundspeed = finite_array(records, "groundspeed_m_s")
    roll = finite_array(records, "roll_deg")
    pitch = finite_array(records, "pitch_deg")
    yaw = finite_array(records, "yaw_deg")
    roll_ctrl = finite_array(records, "roll_ctrl")
    pitch_ctrl = finite_array(records, "pitch_ctrl")
    yaw_ctrl = finite_array(records, "yaw_ctrl")
    throttle = finite_array(records, "throttle_ctrl")
    wind_mag = np.sqrt(wn ** 2 + we ** 2 + wd ** 2)
    gust_window = summarize_gust_window(meta)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    ax1, ax2, ax3, ax4 = axes

    ax1.plot(t, wn, label="wind_north", linewidth=1.5)
    ax1.plot(t, we, label="wind_east", linewidth=1.5)
    ax1.plot(t, wd, label="wind_down", linewidth=1.5)
    ax1.plot(t, wind_mag, label="|wind|", linewidth=2.0, linestyle="--", alpha=0.8)
    ax1.set_ylabel("Wind (m/s)")
    ax1.legend(loc="upper right", ncol=4, fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.plot(t, airspeed, label="airspeed", linewidth=1.8)
    ax2.plot(t, groundspeed, label="groundspeed", linewidth=1.5)
    ax2.set_ylabel("Speed (m/s)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.25)

    ax3.plot(t, roll, label="roll", linewidth=1.3)
    ax3.plot(t, pitch, label="pitch", linewidth=1.3)
    ax3.plot(t, yaw, label="yaw", linewidth=1.0, alpha=0.5)
    ax3.set_ylabel("Attitude (deg)")
    ax3.legend(loc="upper right", ncol=3)
    ax3.grid(True, alpha=0.25)

    ax4.plot(t, roll_ctrl, label="roll_ctrl", linewidth=1.2)
    ax4.plot(t, pitch_ctrl, label="pitch_ctrl", linewidth=1.2)
    ax4.plot(t, yaw_ctrl, label="yaw_ctrl", linewidth=1.2)
    ax4.plot(t, throttle, label="throttle_ctrl", linewidth=1.2)
    ax4.axhline(0.95, color="r", linestyle=":", alpha=0.4)
    ax4.axhline(-0.95, color="r", linestyle=":", alpha=0.4)
    ax4.axhline(0.05, color="orange", linestyle=":", alpha=0.35)
    ax4.set_ylabel("Control")
    ax4.set_xlabel("Time (s)")
    ax4.legend(loc="upper right", ncol=4, fontsize=9)
    ax4.grid(True, alpha=0.25)

    if gust_window is not None:
        for ax in axes:
            ax.axvspan(gust_window[0], gust_window[1], color="gold", alpha=0.18)

    title = (
        f"{metrics['file']} | {metrics['verdict']} | "
        f"base/air={fmt_num(metrics['base_wind_ratio'], 2)}, "
        f"peak/air={fmt_num(metrics['peak_wind_ratio'], 2)}, "
        f"|roll|P95={fmt_num(metrics['roll_abs_p95_deg'], 1)}°, "
        f"ctrl_sat={fmt_num(metrics['ctrl_sat_max_fraction'] * 100.0, 1)}%"
    )
    fig.suptitle(title, fontsize=13)

    if metrics["flags"]:
        note = "\n".join(f"- {x}" for x in metrics["flags"][:6])
        fig.text(0.015, 0.01, note, fontsize=9, va="bottom", ha="left")

    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def write_summary_csv(rows: Sequence[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file",
        "path",
        "dataset_type",
        "config_type",
        "maneuver_type",
        "sample_count",
        "duration_sec",
        "coverage_sec",
        "coverage_ratio",
        "effective_rate_hz",
        "median_airspeed_m_s",
        "p05_airspeed_m_s",
        "min_airspeed_m_s",
        "mean_groundspeed_m_s",
        "min_groundspeed_m_s",
        "base_wind_mag_m_s",
        "gust_mag_m_s",
        "max_logged_wind_mag_m_s",
        "base_wind_ratio",
        "peak_wind_ratio",
        "vertical_wind_ratio",
        "airspeed_p05_over_median",
        "min_groundspeed_over_median_airspeed",
        "roll_abs_p95_deg",
        "pitch_abs_p95_deg",
        "roll_ctrl_sat_fraction",
        "pitch_ctrl_sat_fraction",
        "yaw_ctrl_sat_fraction",
        "throttle_ctrl_sat_fraction",
        "ctrl_sat_max_fraction",
        "verdict",
        "flags",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["flags"] = " | ".join(row.get("flags", []))
            writer.writerow({key: serializable.get(key, "") for key in fieldnames})


def sort_metrics(rows: Sequence[dict]) -> List[dict]:
    return sorted(
        rows,
        key=lambda item: (
            SEVERITY_ORDER.get(item.get("verdict", "PASS"), 0),
            item.get("peak_wind_ratio", 0.0) if math.isfinite(item.get("peak_wind_ratio", float("nan"))) else -1.0,
            item.get("ctrl_sat_max_fraction", 0.0) if math.isfinite(item.get("ctrl_sat_max_fraction", float("nan"))) else -1.0,
        ),
        reverse=True,
    )


def default_output_dir(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.parent / "wind_inspection"
    return input_path / "wind_inspection"


def analyze_single_file(data_path: Path, output_dir: Path, no_plot: bool) -> dict:
    records = load_json(data_path)
    metadata_path = infer_metadata_path(data_path)
    metadata = load_json(metadata_path) if metadata_path else None
    metrics = compute_metrics(records, metadata, data_path)
    print_single_summary(metrics)
    if not no_plot:
        png_path = output_dir / f"{data_path.stem}_inspection.png"
        render_plot(records, metadata, metrics, png_path)
        print(f"图像已保存: {png_path}")
    return metrics


def analyze_directory(input_dir: Path, output_dir: Path, no_plot: bool, plot_limit: int) -> List[dict]:
    data_files = list_data_files(input_dir)
    if not data_files:
        raise FileNotFoundError(f"未找到数据文件: {input_dir}")

    rows: List[dict] = []
    payload_cache: Dict[str, Tuple[Sequence[dict], Optional[dict], dict]] = {}
    for data_path in data_files:
        records = load_json(data_path)
        metadata_path = infer_metadata_path(data_path)
        metadata = load_json(metadata_path) if metadata_path else None
        metrics = compute_metrics(records, metadata, data_path)
        rows.append(metrics)
        payload_cache[str(data_path)] = (records, metadata, metrics)

    rows = sort_metrics(rows)
    csv_path = output_dir / "inspection_summary.csv"
    write_summary_csv(rows, csv_path)

    counts = {key: sum(1 for row in rows if row["verdict"] == key) for key in ("PASS", "WARN", "FAIL")}
    print(
        f"\n=== 目录验收完成: {input_dir} ===\n"
        f"总文件数: {len(rows)}\n"
        f"PASS={counts['PASS']} | WARN={counts['WARN']} | FAIL={counts['FAIL']}\n"
        f"汇总 CSV: {csv_path}"
    )

    print("\nTop 风险样本:")
    for item in rows[: min(10, len(rows))]:
        flag_text = item["flags"][0] if item["flags"] else "无"
        print(
            f"- {item['file']}: {item['verdict']} | peak/air={fmt_num(item['peak_wind_ratio'], 2)} | "
            f"|roll|P95={fmt_num(item['roll_abs_p95_deg'], 1)}° | ctrl_sat={fmt_num(item['ctrl_sat_max_fraction'] * 100.0, 1)}% | {flag_text}"
        )

    if not no_plot and plot_limit > 0:
        plot_dir = output_dir / "plots"
        for item in rows[: min(plot_limit, len(rows))]:
            records, metadata, metrics = payload_cache[item["path"]]
            render_plot(records, metadata, metrics, plot_dir / f"{Path(item['path']).stem}_inspection.png")
        print(f"额外保存高风险样本图: {plot_dir}")

    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="风场验收与可视化工具")
    parser.add_argument(
        "input_path",
        type=str,
        help="单个数据 JSON 文件，或包含多个数据文件的目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录；默认单文件/目录下自动创建 wind_inspection",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="只做统计与风险判断，不保存 PNG 图",
    )
    parser.add_argument(
        "--plot-limit",
        type=int,
        default=0,
        help="目录模式额外保存前 N 个高风险样本图，默认 0",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir(input_path)

    if input_path.is_file():
        analyze_single_file(input_path, output_dir, no_plot=args.no_plot)
    else:
        analyze_directory(input_path, output_dir, no_plot=args.no_plot, plot_limit=max(0, args.plot_limit))


if __name__ == "__main__":
    main()
