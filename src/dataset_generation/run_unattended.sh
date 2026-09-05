#!/usr/bin/env bash
# 无人值守数据采集监督器：
# 1) 先跑完整论文版 160 轮采集
# 2) 自动生成缺失报告
# 3) 按缺失整轮 / 缺失单段自动补采，直到收齐或达到补采上限
#
# 用法：
#   ./run_unattended.sh [nohup|tmux|foreground] [seed] [--speed 倍数]
#   ./run_unattended.sh foreground --seed 26 --speed 2 --max-recover-passes 5
#
# 说明：
# - foreground: 前台运行监督器
# - nohup: 后台运行监督器（断 SSH 不中断）
# - tmux: 在 tmux 会话中启动监督器（默认 detached）
# - 监督器会自动执行：bulk collect -> report -> targeted recover -> report

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
cd "$SCRIPT_DIR"

DATA_DIR="$SCRIPT_DIR/data"
LOG_DIR="$SCRIPT_DIR/logs"
REPORT_PATH="$LOG_DIR/missing_data_report.txt"
SUPERVISOR_LOG="$LOG_DIR/unattended_supervisor.log"
TOTAL_SEGMENTS="800"
DATASET_LOG="$LOG_DIR/multi_segment_160runs.log"
PIDFILE="$LOG_DIR/collect.pid"
STATE_FILE="$LOG_DIR/unattended_state.env"
TMUX_SESSION="dataset_collect"

mkdir -p "$DATA_DIR" "$LOG_DIR"

declare -a CHILD_CMD=()
CURRENT_CHILD_PID=""
LOCK_ACQUIRED=0

MODE="foreground"
SEED="26"
SPEED="1"
ROUND_TIMEOUT="1800"
MAX_RECOVER_PASSES="5"
if [[ -x "$WORKSPACE_VENV_PYTHON" ]]; then
  PYTHON_BIN="$WORKSPACE_VENV_PYTHON"
else
  PYTHON_BIN="python3"
fi
PX4_ROOT_OVERRIDE=""
NO_SKIP=0
DRY_RUN=0
INTERNAL_WORKER=0

usage() {
  cat <<EOF
用法:
  $(basename "$0") [nohup|tmux|foreground] [seed] [选项]

选项:
  --seed N                  随机种子（默认: 26）
  --speed X                 仿真加速倍数（默认: 1）
  --round-timeout SEC       单轮超时秒数（默认: 1800）
  --max-recover-passes N    自动补采轮数上限（默认: 5）
  --px4-root PATH           指定 PX4 根目录
  --python BIN              指定 Python 可执行文件（默认: 优先 `.venv/bin/python`，否则 python3）
  --no-skip                 首轮采集时不跳过已存在数据
  --dry-run                 仅打印将执行的命令，不真正启动采集
  -h, --help                显示帮助

示例:
  $(basename "$0") foreground
  $(basename "$0") foreground --speed 2
  $(basename "$0") nohup 26 --speed 3 --max-recover-passes 6
  $(basename "$0") tmux --seed 26 --speed 4
EOF
}

timestamp() {
  date '+%F %T'
}

log() {
  local msg="$*"
  local line
  line="[$(timestamp)] $msg"

  local stdout_target=""
  local log_target=""
  stdout_target="$(readlink -f /proc/$$/fd/1 2>/dev/null || true)"
  log_target="$(readlink -f "$SUPERVISOR_LOG" 2>/dev/null || true)"

  if [[ -n "$stdout_target" && -n "$log_target" && "$stdout_target" == "$log_target" ]]; then
    printf '%s\n' "$line" >> "$SUPERVISOR_LOG"
  else
    printf '%s\n' "$line" | tee -a "$SUPERVISOR_LOG"
  fi
}

die() {
  log "错误: $*"
  exit 1
}

is_positive_number() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

is_positive_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

cleanup() {
  local exit_code=$?

  if [[ -n "$CURRENT_CHILD_PID" ]]; then
    if kill -0 "$CURRENT_CHILD_PID" 2>/dev/null; then
      kill "$CURRENT_CHILD_PID" 2>/dev/null || true
      wait "$CURRENT_CHILD_PID" 2>/dev/null || true
    fi
    CURRENT_CHILD_PID=""
  fi

  if [[ "$LOCK_ACQUIRED" -eq 1 && -f "$PIDFILE" ]]; then
    local pid_in_file
    pid_in_file="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [[ "$pid_in_file" == "$$" ]]; then
      rm -f "$PIDFILE"
    fi
  fi

  if [[ "$LOCK_ACQUIRED" -eq 1 ]]; then
    local current_status=""
    if [[ -f "$STATE_FILE" ]]; then
      current_status="$(grep '^STATUS=' "$STATE_FILE" | tail -n 1 | cut -d'=' -f2- || true)"
    fi

    case "$current_status" in
      completed|partial)
        printf 'EXIT_CODE=%q\n' "$exit_code" >> "$STATE_FILE"
        printf 'UPDATED_AT=%q\n' "$(timestamp)" >> "$STATE_FILE"
        ;;
      *)
        printf 'STATUS=%q\n' "stopped" > "$STATE_FILE"
        printf 'EXIT_CODE=%q\n' "$exit_code" >> "$STATE_FILE"
        printf 'UPDATED_AT=%q\n' "$(timestamp)" >> "$STATE_FILE"
        ;;
    esac
  fi
}

trap cleanup EXIT INT TERM

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      foreground|nohup|tmux)
        MODE="$1"
        shift
        ;;
      __worker)
        INTERNAL_WORKER=1
        shift
        ;;
      --seed)
        [[ $# -ge 2 ]] || die "--seed 需要参数"
        SEED="$2"
        shift 2
        ;;
      --speed)
        [[ $# -ge 2 ]] || die "--speed 需要参数"
        SPEED="$2"
        shift 2
        ;;
      --round-timeout)
        [[ $# -ge 2 ]] || die "--round-timeout 需要参数"
        ROUND_TIMEOUT="$2"
        shift 2
        ;;
      --max-recover-passes)
        [[ $# -ge 2 ]] || die "--max-recover-passes 需要参数"
        MAX_RECOVER_PASSES="$2"
        shift 2
        ;;
      --px4-root)
        [[ $# -ge 2 ]] || die "--px4-root 需要参数"
        PX4_ROOT_OVERRIDE="$2"
        shift 2
        ;;
      --python)
        [[ $# -ge 2 ]] || die "--python 需要参数"
        PYTHON_BIN="$2"
        shift 2
        ;;
      --no-skip)
        NO_SKIP=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        if [[ "$SEED" == "26" && "$1" =~ ^[0-9]+$ ]]; then
          SEED="$1"
          shift
        else
          die "未知参数: $1"
        fi
        ;;
    esac
  done

  is_positive_integer "$SEED" || die "seed 必须是正整数，当前: $SEED"
  is_positive_number "$SPEED" || die "speed 必须是正数，当前: $SPEED"
  is_positive_integer "$ROUND_TIMEOUT" || die "round-timeout 必须是正整数，当前: $ROUND_TIMEOUT"
  is_positive_integer "$MAX_RECOVER_PASSES" || die "max-recover-passes 必须是非负整数，当前: $MAX_RECOVER_PASSES"
}

detect_px4_root() {
  if [[ -n "$PX4_ROOT_OVERRIDE" ]]; then
    echo "$PX4_ROOT_OVERRIDE"
    return
  fi

  if [[ -n "${PX4_ROOT:-}" ]]; then
    echo "$PX4_ROOT"
    return
  fi

  local candidates=(
    "$HOME/wind_datasets/PX4-Autopilot-v133"
    "$HOME/wind_datasets/PX4-Autopilot"
    "$HOME/PX4-Autopilot-v133"
    "$HOME/PX4-Autopilot"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -d "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done

  echo "$HOME/PX4-Autopilot"
}

preflight() {
  export PX4_ROOT
  PX4_ROOT="$(detect_px4_root)"
  export PX4_SIM_SPEED_FACTOR="$SPEED"

  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "未找到 Python 可执行文件: $PYTHON_BIN"
  [[ -f "$SCRIPT_DIR/scripts/generate_dataset.py" ]] || die "缺少 scripts/generate_dataset.py"
  [[ -d "$PX4_ROOT" ]] || die "PX4_ROOT 不存在: $PX4_ROOT"

  if [[ "$DRY_RUN" -eq 0 ]]; then
    command -v tee >/dev/null 2>&1 || die "系统缺少 tee 命令"
    (cd "$SCRIPT_DIR/scripts" && "$PYTHON_BIN" -c 'from flight_controller import FlightController' >/dev/null 2>&1) \
      || die "当前 Python 环境无法加载飞控依赖链（mavsdk / grpcio 版本可能不兼容）。请先执行: python3 -m pip install --user --upgrade grpcio mavsdk numpy"
  fi
}

acquire_lock() {
  if [[ -f "$PIDFILE" ]]; then
    local old_pid
    old_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
      die "已有无人值守采集监督器在运行 (PID=$old_pid)，请先停止旧任务"
    fi
    rm -f "$PIDFILE"
  fi

  echo "$$" > "$PIDFILE"
  LOCK_ACQUIRED=1
}

write_state() {
  local status="$1"
  shift || true
  {
    printf 'STATUS=%q\n' "$status"
    printf 'PID=%q\n' "$$"
    printf 'MODE=%q\n' "$MODE"
    printf 'SEED=%q\n' "$SEED"
    printf 'SPEED=%q\n' "$SPEED"
    printf 'ROUND_TIMEOUT=%q\n' "$ROUND_TIMEOUT"
    printf 'MAX_RECOVER_PASSES=%q\n' "$MAX_RECOVER_PASSES"
    printf 'PX4_ROOT=%q\n' "$PX4_ROOT"
    printf 'DATA_DIR=%q\n' "$DATA_DIR"
    printf 'UPDATED_AT=%q\n' "$(timestamp)"
    for item in "$@"; do
      printf '%s\n' "$item"
    done
  } > "$STATE_FILE"
}

run_python() {
  local label="$1"
  shift
  local -a cmd=("$PYTHON_BIN" -u "$SCRIPT_DIR/scripts/generate_dataset.py" "$@")

  log "开始执行 [$label]: ${cmd[*]}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "[dry-run] 跳过真实执行 [$label]"
    return 0
  fi

  set +e
  local _fifo; _fifo=$(mktemp -u /tmp/run_unattended_XXXXXX.fifo)
  mkfifo "$_fifo"
  tee -a "$SUPERVISOR_LOG" < "$_fifo" &
  local _tee_pid=$!
  "${cmd[@]}" > "$_fifo" 2>&1 &
  CURRENT_CHILD_PID=$!
  rm -f "$_fifo"
  wait "$CURRENT_CHILD_PID"
  local rc=$?
  CURRENT_CHILD_PID=""
  set -e

  if [[ $rc -eq 0 ]]; then
    log "[$label] 执行完成"
  else
    log "[$label] 执行失败，退出码=$rc"
  fi
  return $rc
}

refresh_missing_report() {
  run_python "生成缺失报告" \
    --mode report \
    --output-dir "$DATA_DIR"
}

get_missing_count() {
  if [[ "$DRY_RUN" -eq 1 && ! -f "$REPORT_PATH" ]]; then
    echo 0
    return
  fi

  if [[ ! -f "$REPORT_PATH" ]]; then
    echo "$TOTAL_SEGMENTS"
    return
  fi

  local count
  count="$(grep -Eo '总计: [0-9]+/[0-9]+ 段缺失' "$REPORT_PATH" | tail -n 1 | sed -E 's/.*总计: ([0-9]+)\/[0-9]+.*/\1/' || true)"
  if [[ -z "$count" ]]; then
    echo "$TOTAL_SEGMENTS"
  else
    echo "$count"
  fi
}

load_missing_segments() {
  MISSING_RUNS=()
  MISSING_SEGS=()

  [[ -f "$REPORT_PATH" ]] || return 0

  while IFS= read -r line; do
    if [[ "$line" =~ Run[[:space:]]+([0-9]+)[[:space:]]+\|[[:space:]]+Seg[[:space:]]+([0-9]+)[[:space:]]+\| ]]; then
      MISSING_RUNS+=("${BASH_REMATCH[1]}")
      MISSING_SEGS+=("${BASH_REMATCH[2]}")
    fi
  done < "$REPORT_PATH"
}

recover_missing_segments() {
  declare -A run_counts=()
  declare -A run_segments=()

  if [[ ${#MISSING_RUNS[@]} -eq 0 ]]; then
    log "缺失报告为空，跳过补采"
    return 0
  fi

  local idx run seg
  for idx in "${!MISSING_RUNS[@]}"; do
    run="${MISSING_RUNS[$idx]}"
    seg="${MISSING_SEGS[$idx]}"
    run_counts[$run]=$(( ${run_counts[$run]:-0} + 1 ))
    run_segments[$run]="${run_segments[$run]:-} $seg"
  done

  local sorted_runs=()
  while IFS= read -r run; do
    [[ -n "$run" ]] && sorted_runs+=("$run")
  done < <(printf '%s\n' "${!run_counts[@]}" | sort -n)

  for run in "${sorted_runs[@]}"; do
    local count="${run_counts[$run]}"
    if [[ "$count" -ge 5 ]]; then
      log "自动补采整轮: run=$run (5 段缺失)"
      run_python "补采整轮 run=$run" \
        --mode recover \
        --run "$run" \
        --seed "$SEED" \
        --output-dir "$DATA_DIR" || true
      continue
    fi

    local seg_list=()
    while IFS= read -r seg; do
      [[ -n "$seg" ]] && seg_list+=("$seg")
    done < <(for seg in ${run_segments[$run]}; do echo "$seg"; done | sort -n | uniq)

    for seg in "${seg_list[@]}"; do
      log "自动补采单段: run=$run segment=$seg"
      run_python "补采单段 run=$run seg=$seg" \
        --mode recover \
        --run "$run" \
        --segment "$seg" \
        --seed "$SEED" \
        --output-dir "$DATA_DIR" || true
    done
  done
}

run_supervisor() {
  preflight
  acquire_lock

  : > "$SUPERVISOR_LOG"
  write_state "starting"

  log "==========================================================="
  log "无人值守监督器启动"
  if [[ "$INTERNAL_WORKER" -eq 1 ]]; then
    log "模式: ${MODE} (worker)"
  else
    log "模式: ${MODE}"
  fi
  log "种子: $SEED"
  log "加速: ${SPEED}x"
  log "单轮超时: ${ROUND_TIMEOUT}s"
  log "补采上限: ${MAX_RECOVER_PASSES} 轮"
  log "PX4_ROOT: $PX4_ROOT"
  log "数据目录: $DATA_DIR"
  log "监督器日志: $SUPERVISOR_LOG"
  log "数据集日志: $DATASET_LOG"
  log "Python: $PYTHON_BIN"
  log "==========================================================="

  write_state "bulk_collecting"
  local -a bulk_args=(
    --mode multi_segment_160
    --seed "$SEED"
    --output-dir "$DATA_DIR"
    --log "$DATASET_LOG"
    --round-timeout "$ROUND_TIMEOUT"
  )
  if [[ "$NO_SKIP" -eq 1 ]]; then
    bulk_args+=(--no-skip)
  fi

  if ! run_python "160轮批量采集" "${bulk_args[@]}"; then
    log "批量采集返回非零退出码，将先尝试生成缺失报告判断是否还能继续"
  fi

  write_state "reporting"
  if ! refresh_missing_report; then
    write_state "failed" "ERROR=missing_report_failed"
    log "错误: 缺失报告生成失败，停止无人值守补采；请先修复 Python 环境或查看上方错误日志"
    return 3
  fi
  local missing_count
  missing_count="$(get_missing_count)"
  log "首轮采集后缺失段数: $missing_count/$TOTAL_SEGMENTS"

  if [[ "$missing_count" -eq 0 ]]; then
    write_state "completed" "MISSING_COUNT=0"
    log "数据集已完整收齐，无需补采"
    return 0
  fi

  local pass=1
  local previous_missing="$missing_count"
  while [[ "$pass" -le "$MAX_RECOVER_PASSES" && "$missing_count" -gt 0 ]]; do
    write_state "recovering" "RECOVER_PASS=$pass" "MISSING_COUNT=$missing_count"
    log "开始自动补采 pass ${pass}/${MAX_RECOVER_PASSES}"

    load_missing_segments
    recover_missing_segments

    write_state "reporting" "RECOVER_PASS=$pass"
    refresh_missing_report || true
    missing_count="$(get_missing_count)"
    log "补采 pass $pass 后缺失段数: $missing_count/$TOTAL_SEGMENTS"

    if [[ "$missing_count" -eq 0 ]]; then
      write_state "completed" "MISSING_COUNT=0" "RECOVER_PASS=$pass"
      log "所有缺失段已自动补齐"
      return 0
    fi

    if [[ "$missing_count" -ge "$previous_missing" ]]; then
      log "警告: 本轮补采后缺失段未减少（$previous_missing -> $missing_count）"
    fi

    previous_missing="$missing_count"
    pass=$((pass + 1))
  done

  if [[ "$missing_count" -gt 0 ]]; then
    write_state "partial" "MISSING_COUNT=$missing_count"
    log "仍有缺失段未补齐，请查看 $REPORT_PATH"
    return 2
  fi

  write_state "completed" "MISSING_COUNT=0"
  return 0
}

launch_nohup() {
  local launcher="$SCRIPT_DIR/run_unattended.sh"
  local -a cmd=(
    bash "$launcher" __worker
    --seed "$SEED"
    --speed "$SPEED"
    --round-timeout "$ROUND_TIMEOUT"
    --max-recover-passes "$MAX_RECOVER_PASSES"
    --python "$PYTHON_BIN"
  )
  [[ -n "$PX4_ROOT_OVERRIDE" ]] && cmd+=(--px4-root "$PX4_ROOT_OVERRIDE")
  [[ "$NO_SKIP" -eq 1 ]] && cmd+=(--no-skip)
  [[ "$DRY_RUN" -eq 1 ]] && cmd+=(--dry-run)

  printf 'STATUS=%q\n' "launching_nohup" > "$STATE_FILE"
  printf 'LAUNCHER_PID=%q\n' "$$" >> "$STATE_FILE"
  printf 'UPDATED_AT=%q\n' "$(timestamp)" >> "$STATE_FILE"

  nohup "${cmd[@]}" >> "$SUPERVISOR_LOG" 2>&1 &
  local bg_pid=$!
  printf 'WORKER_PID=%q\n' "$bg_pid" >> "$STATE_FILE"

  echo "后台监督器已启动"
  echo "PID: $bg_pid"
  echo "日志: $SUPERVISOR_LOG"
  echo "数据日志: $DATASET_LOG"
  echo "查看进度: tail -f $SUPERVISOR_LOG"
}

launch_tmux() {
  command -v tmux >/dev/null 2>&1 || die "tmux 未安装，无法使用 tmux 模式"

  local launcher="$SCRIPT_DIR/run_unattended.sh"
  local -a cmd=(
    bash "$launcher" __worker
    --seed "$SEED"
    --speed "$SPEED"
    --round-timeout "$ROUND_TIMEOUT"
    --max-recover-passes "$MAX_RECOVER_PASSES"
    --python "$PYTHON_BIN"
  )
  [[ -n "$PX4_ROOT_OVERRIDE" ]] && cmd+=(--px4-root "$PX4_ROOT_OVERRIDE")
  [[ "$NO_SKIP" -eq 1 ]] && cmd+=(--no-skip)
  [[ "$DRY_RUN" -eq 1 ]] && cmd+=(--dry-run)

  local cmd_str
  printf -v cmd_str '%q ' "${cmd[@]}"

  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    die "tmux 会话 '$TMUX_SESSION' 已存在，请先手动结束或更换会话名"
  fi

  tmux new-session -d -s "$TMUX_SESSION" "cd $(printf '%q' "$SCRIPT_DIR") && $cmd_str"
  echo "tmux 监督器已启动: session=$TMUX_SESSION"
  echo "附加会话: tmux attach -t $TMUX_SESSION"
  echo "日志: $SUPERVISOR_LOG"
}

main() {
  parse_args "$@"

  if [[ "$INTERNAL_WORKER" -eq 1 ]]; then
    MODE="foreground"
    run_supervisor
    return
  fi

  case "$MODE" in
    foreground)
      run_supervisor
      ;;
    nohup)
      preflight
      launch_nohup
      ;;
    tmux)
      preflight
      launch_tmux
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
