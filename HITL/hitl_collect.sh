#!/usr/bin/env bash
# =============================================================================
# HITL 实验后处理一键脚本
# 归档三路数据 → 时间戳对齐 → 分阶段 RMSE / 时延统计
#
# 三路数据来源：
#   1) 树莓派 : hitl_data_*.csv         → HITL/logs_in_rasbpi/   （需从 Pi 拷回）
#   2) WSL2   : wind_truth_*.csv        → HITL/JSBSim_truth_wind/ （jsbsim_bridge 产出，本脚本自动拷贝）
#   3) 飞控   : *.ulg (ULog)            → HITL/PX4_ulog/          （从 QGC/SD 卡拷回）
#
# 用法：
#   bash HITL/hitl_collect.sh                     # 用默认路径
#   PI_CSV_SRC=airsim@raspi:~/wind-estimation/HITL/logs_in_rasbpi/ bash HITL/hitl_collect.sh
#
# 可用环境变量覆盖：
#   PX4_ROOT        PX4-Autopilot 根目录（默认 /path/to/PX4-Autopilot）
#   TRUTH_SRC_DIR   风真值源目录（默认 $PX4_ROOT/Tools/jsbsim_bridge）
#   PI_CSV_SRC      树莓派 CSV 源（scp 地址或本地目录；留空表示已手动放好，仅校验）
#   PY              Python 解释器（默认项目 .venv）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PX4_ROOT="${PX4_ROOT:-$HOME/PX4-Autopilot}"
TRUTH_SRC_DIR="${TRUTH_SRC_DIR:-$PX4_ROOT/Tools/jsbsim_bridge}"
PI_CSV_SRC="${PI_CSV_SRC:-}"
PY="${PY:-$PROJ_ROOT/.venv/bin/python3}"

TRUTH_DIR="$PROJ_ROOT/HITL/JSBSim_truth_wind"
RASPI_DIR="$PROJ_ROOT/HITL/logs_in_rasbpi"
ULOG_DIR="$PROJ_ROOT/HITL/PX4_ulog"
ALIGNED_DIR="$PROJ_ROOT/HITL/aligned"

mkdir -p "$TRUTH_DIR" "$RASPI_DIR" "$ULOG_DIR" "$ALIGNED_DIR"

echo "=================================================================="
echo " HITL 后处理  |  项目根: $PROJ_ROOT"
echo "=================================================================="

# ---------- [1/5] 归档 JSBSim 风真值 ----------
echo "[1/5] 归档 JSBSim 风真值 (wind_truth_*.csv)"
latest_truth="$(ls -t "$TRUTH_SRC_DIR"/wind_truth_*.csv 2>/dev/null | head -n1 || true)"
if [[ -n "$latest_truth" ]]; then
    cp -v "$latest_truth" "$TRUTH_DIR/"
else
    echo "  ! 未在 $TRUTH_SRC_DIR 找到 wind_truth_*.csv"
    echo "    若 jsbsim_bridge 输出到别处，请设 TRUTH_SRC_DIR，或手动拷入 $TRUTH_DIR"
fi

# ---------- [2/5] 归档树莓派 CSV ----------
echo "[2/5] 归档树莓派在线日志 (hitl_data_*.csv)"
if [[ -n "$PI_CSV_SRC" ]]; then
    if [[ "$PI_CSV_SRC" == *:* ]]; then
        echo "  scp 从 $PI_CSV_SRC 拉取..."
        scp "$PI_CSV_SRC"/hitl_data_*.csv "$RASPI_DIR/" || echo "  ! scp 失败，请手动拷贝"
        scp "$PI_CSV_SRC"/online_deployment_*.log "$RASPI_DIR/" 2>/dev/null || true
    else
        cp -v "$PI_CSV_SRC"/hitl_data_*.csv "$RASPI_DIR/" 2>/dev/null || echo "  ! 源目录无 CSV"
    fi
fi

# ---------- [3/5] 校验三路数据齐备 ----------
echo "[3/5] 校验三路数据"
n_truth=$(ls "$TRUTH_DIR"/wind_truth_*.csv 2>/dev/null | wc -l)
n_raspi=$(ls "$RASPI_DIR"/hitl_data_*.csv 2>/dev/null | wc -l)
n_ulog=$(ls "$ULOG_DIR"/*.ulg 2>/dev/null | wc -l)
echo "  风真值 CSV : $n_truth 个   ($TRUTH_DIR)"
echo "  树莓派 CSV : $n_raspi 个   ($RASPI_DIR)"
echo "  PX4 ULog   : $n_ulog 个   ($ULOG_DIR)"
missing=0
[[ "$n_truth" -eq 0 ]] && { echo "  ✗ 缺风真值 CSV"; missing=1; }
[[ "$n_raspi" -eq 0 ]] && { echo "  ✗ 缺树莓派 CSV（请从 Pi 拷回，或用 PI_CSV_SRC）"; missing=1; }
[[ "$n_ulog"  -eq 0 ]] && { echo "  ✗ 缺 PX4 ULog（请从 QGC/SD 卡导出 .ulg 到 $ULOG_DIR）"; missing=1; }
if [[ "$missing" -eq 1 ]]; then
    echo "  → 数据不齐，无法算硬件 RMSE。补齐后重跑本脚本。"
    exit 1
fi

# ---------- [4/5] 三路时间戳对齐 ----------
echo "[4/5] 时间戳对齐 → HITL/aligned/hitl_aligned_master.csv"
if ! command -v ulog2csv >/dev/null 2>&1; then
    echo "  ! 未找到 ulog2csv（pyulog）。安装：$PY -m pip install pyulog"
fi
"$PY" "$PROJ_ROOT/HITL/align_hitl_timestamps.py"

# ---------- [5/5] 分布内硬件 RMSE / 时延统计 ----------
echo "[5/5] 分布内硬件 RMSE / 时延统计（单段稳态风）"
"$PY" "$PROJ_ROOT/scripts/hitl_indist_rmse.py"

# 多段实验（wind_config_phases.txt）才需要分阶段统计；单段可忽略下面这步。
if [[ "${RUN_PHASE_ANALYSIS:-0}" == "1" ]]; then
    echo "[5b] 分阶段统计（RUN_PHASE_ANALYSIS=1）"
    "$PY" "$PROJ_ROOT/scripts/hitl_phase_analysis.py" || echo "  ! 分阶段统计失败（单段实验 phase=unknown 时正常，可忽略）"
fi

echo "=================================================================="
echo " 完成。产物："
echo "   对齐主表 : HITL/aligned/hitl_aligned_master.csv"
echo "   对齐摘要 : HITL/aligned/hitl_alignment_summary.md"
echo "   分布内RMSE: paper/tables/table_hitl_indist_rmse.md   ← 回填论文表6第四列"
echo "=================================================================="
