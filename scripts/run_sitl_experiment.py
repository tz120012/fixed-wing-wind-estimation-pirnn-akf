#!/usr/bin/env python3
"""
run_sitl_experiment.py  ─  SITL 全自动闭环实验编排器 v1.0
===========================================================
一键执行全部 6 组 SITL 实验（3 种风场 × 2 种估计模式），
自动管理 PX4 SITL 生命周期、任务上传、数据采集与结果分析。

实验矩阵
--------
  风场阶段   | estimator_mode | 输出 CSV
  -----------|----------------|-----------------------------
  steady     | baseline       | SITL/results/steady_baseline_*.csv
  steady     | pirnn_akf      | SITL/results/steady_pirnn_akf_*.csv
  gust_light | baseline       | SITL/results/gust_light_baseline_*.csv
  gust_light | pirnn_akf      | SITL/results/gust_light_pirnn_akf_*.csv
  gust_strong| baseline       | SITL/results/gust_strong_baseline_*.csv
  gust_strong| pirnn_akf      | SITL/results/gust_strong_pirnn_akf_*.csv

运行命令
--------
  cd .
  source .venv/bin/activate
  python3 scripts/run_sitl_experiment.py

  # 只跑指定风场和模式（调试用）：
  python3 scripts/run_sitl_experiment.py --phase steady --mode baseline
  python3 scripts/run_sitl_experiment.py --phase steady --mode pirnn_akf

依赖
----
  - PX4-Autopilot 已编译（make px4_sitl jsbsim_rascal 可用）
  - pymavlink、mavsdk 已安装（.venv 中）
  - SITL/wind_configs/*.txt 风场配置文件已存在
  - config/config_sitl.yaml 模型路径配置已存在
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── 代理清理（防止 gRPC/mavsdk 走 WSL 代理）──────────────────────────────
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ[_v] = ""
os.environ["grpc_proxy"] = ""
os.environ["no_grpc_proxy"] = "*"
os.environ["no_proxy"] = os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1,0.0.0.0"

from pymavlink import mavutil

_ROOT = Path(__file__).resolve().parent.parent
_SRC  = _ROOT / "src"
_SITL = _ROOT / "SITL"
_WIND_CONFIGS = _SITL / "wind_configs"
_RESULTS      = _SITL / "results"
_PX4_ROOT = Path(os.environ.get("PX4_ROOT", str(Path.home() / "PX4-Autopilot"))).expanduser()
_WIND_CFG_DST = _PX4_ROOT / "Tools" / "jsbsim_bridge" / "wind_config.txt"
_CONFIG_SITL  = _ROOT / "config" / "config_sitl.yaml"
_VENV_PYTHON  = _ROOT / ".venv" / "bin" / "python3"

WIND_PHASES = ["steady", "gust_light", "gust_strong"]
MODES       = ["baseline", "pirnn_akf"]

# 每组实验时长（秒）：稳定阶段 5 分钟，给模型足够的 warmup + 数据
EXPERIMENT_DURATION_S = 300   # 5 分钟
# PX4 SITL 启动等待时间（秒）
PX4_STARTUP_WAIT_S    = 45
# 飞机解锁+起飞等待（秒）
TAKEOFF_WAIT_S        = 30
# MAVLink 连接字符串
MAVLINK_CONN          = "udpin:0.0.0.0:14550"


# ─────────────────────────────────────────────────────────────────────────────
# PX4 SITL 管理
# ─────────────────────────────────────────────────────────────────────────────

class PX4SITLManager:
    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def set_wind_config(self, phase: str):
        src = _WIND_CONFIGS / f"{phase}.txt"
        if not src.exists():
            raise FileNotFoundError(f"风场配置不存在: {src}")
        shutil.copy2(src, _WIND_CFG_DST)
        print(f"  [PX4] 风场配置已设置: {phase}")

    def clear_rootfs(self):
        rootfs = _PX4_ROOT / "build" / "px4_sitl_default" / "tmp" / "rootfs"
        for name in ["eeprom", "dataman", "mission_state"]:
            target = rootfs / name
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.is_file():
                target.unlink(missing_ok=True)
        for i in range(4):
            Path(f"/tmp/px4_lock-{i}").unlink(missing_ok=True)

    def start(self, log_file: Path | None = None):
        self.clear_rootfs()
        env = os.environ.copy()
        env["HEADLESS"] = "1"
        env["NO_PXH"]   = "1"
        for _v in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env[_v] = ""
        env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost,::1,0.0.0.0"

        log_path = log_file or (_SITL / "px4_sitl.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(log_path, "w")

        print(f"  [PX4] 启动 SITL (jsbsim_rascal)...")
        self.proc = subprocess.Popen(
            ["make", "px4_sitl", "jsbsim_rascal"],
            cwd=str(_PX4_ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        print(f"  [PX4] PID={self.proc.pid}，等待 {PX4_STARTUP_WAIT_S}s 启动...")
        time.sleep(PX4_STARTUP_WAIT_S)

        if self.proc.poll() is not None:
            raise RuntimeError(f"PX4 SITL 启动失败 (exit={self.proc.returncode})，查看日志: {log_path}")
        print(f"  [PX4] SITL 已启动")

    def stop(self):
        if self.proc is not None:
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

        for cmd in [
            ["pkill", "-9", "-x", "px4"],
            ["pkill", "-9", "-x", "JSBSim"],
            ["pkill", "-9", "-f", "jsbsim_bridge"],
        ]:
            subprocess.run(cmd, stderr=subprocess.DEVNULL, timeout=5)

        time.sleep(3)
        self.clear_rootfs()
        print("  [PX4] SITL 已停止")


# ─────────────────────────────────────────────────────────────────────────────
# 飞机起飞与任务控制（pymavlink）
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_connection(timeout: float = 30.0) -> mavutil.mavfile | None:
    """等待 MAVLink 心跳，返回连接对象或 None。"""
    print(f"  [MAV] 等待 MAVLink 心跳 (最多 {timeout:.0f}s)...")
    try:
        conn = mavutil.mavlink_connection(MAVLINK_CONN)
        conn.wait_heartbeat(timeout=timeout)
        print(f"  [MAV] 连接成功 (system={conn.target_system})")
        return conn
    except Exception as exc:
        print(f"  [MAV] 连接失败: {exc}")
        return None


def arm_and_takeoff(conn: mavutil.mavfile, target_alt: float = 100.0) -> bool:
    """
    通过 MAVLink 解锁飞机并切换到 AUTO 模式（MISSION 模式）或 GUIDED 模式起飞。
    返回 True 表示成功。
    """
    # 请求所有数据流
    conn.mav.request_data_stream_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1,
    )
    time.sleep(1)

    # 设置模式为 GUIDED（固定翼 GUIDED = 绕圈）
    mode_id = conn.mode_mapping().get("GUIDED", None)
    if mode_id is None:
        # 固定翼可能叫 AUTO 或 LOITER
        for candidate in ["GUIDED", "AUTO", "LOITER", "FBWA"]:
            mode_id = conn.mode_mapping().get(candidate)
            if mode_id is not None:
                print(f"  [MAV] 模式映射: {candidate} = {mode_id}")
                break

    if mode_id is not None:
        conn.mav.set_mode_send(
            conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        time.sleep(1)

    # 解锁
    print("  [MAV] 发送解锁指令...")
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0,
    )
    time.sleep(2)

    # 起飞指令
    print(f"  [MAV] 发送起飞指令 (目标高度 {target_alt}m)...")
    conn.mav.command_long_send(
        conn.target_system, conn.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        15,   # 最小俯仰角
        0, 0, 0,
        float("nan"), float("nan"),
        target_alt,
    )

    print(f"  [MAV] 等待起飞 ({TAKEOFF_WAIT_S}s)...")
    time.sleep(TAKEOFF_WAIT_S)

    # 切换到 LOITER（固定翼在目标高度绕圈）
    loiter_id = conn.mode_mapping().get("LOITER")
    if loiter_id is not None:
        conn.mav.set_mode_send(
            conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            loiter_id,
        )
        print("  [MAV] 已切换到 LOITER 模式")

    return True


# ─────────────────────────────────────────────────────────────────────────────
# 单组实验执行
# ─────────────────────────────────────────────────────────────────────────────

def run_one_experiment(phase: str, mode: str, run_idx: int) -> Path | None:
    """
    执行单组 SITL 实验：设置风场 → 启动 PX4 → 起飞 → 运行估计器 → 停止。
    返回输出 CSV 路径，失败返回 None。
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = _RESULTS / f"{phase}_{mode}_{ts}.csv"
    _RESULTS.mkdir(parents=True, exist_ok=True)

    px4 = PX4SITLManager()
    px4_log = _SITL / "logs" / f"px4_{phase}_{mode}_{ts}.log"
    px4_log.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  实验 #{run_idx}: phase={phase}  mode={mode}")
    print(f"{'='*60}")

    try:
        # 1. 设置风场配置
        px4.set_wind_config(phase)

        # 2. 启动 PX4 SITL
        px4.start(log_file=px4_log)

        # 3. 等待 MAVLink 连接
        conn = wait_for_connection(timeout=30.0)
        if conn is None:
            print("  ❌ MAVLink 连接失败，跳过此组")
            return None

        # 4. 请求额外消息（NAV_CONTROLLER_OUTPUT）
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0, 62, 20000, 0, 0, 0, 0, 0,
        )
        conn.close()

        # 5. 起飞（用单独连接，避免与估计器冲突）
        conn2 = wait_for_connection(timeout=15.0)
        if conn2 is not None:
            arm_and_takeoff(conn2, target_alt=100.0)
            conn2.close()

        # 6. 运行估计器（子进程）
        estimator_args = [
            str(_VENV_PYTHON),
            str(_SRC / "7_sitl_closedloop_eval.py"),
            "--mode", mode,
            "--connection", MAVLINK_CONN,
            "--config", str(_CONFIG_SITL),
            "--output", str(out_csv),
            "--duration", str(EXPERIMENT_DURATION_S),
        ]
        print(f"\n  [EVAL] 启动估计器: {' '.join(estimator_args[-4:])}")
        eval_proc = subprocess.Popen(
            estimator_args,
            cwd=str(_SRC),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # 实时打印估计器输出
        start_t = time.time()
        while eval_proc.poll() is None:
            line = eval_proc.stdout.readline()
            if line:
                print(f"    {line.rstrip()}")
            elapsed = time.time() - start_t
            if elapsed > EXPERIMENT_DURATION_S + 30:
                print("  [EVAL] 超时，强制终止估计器")
                eval_proc.terminate()
                break

        rc = eval_proc.wait(timeout=10)
        print(f"  [EVAL] 估计器退出 (rc={rc})")

        if out_csv.exists() and out_csv.stat().st_size > 1024:
            print(f"  ✓ CSV 已保存: {out_csv}  ({out_csv.stat().st_size//1024} KB)")
            return out_csv
        else:
            print(f"  ⚠ CSV 为空或过小: {out_csv}")
            return None

    except Exception as exc:
        print(f"  ❌ 实验异常: {exc}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        px4.stop()
        time.sleep(5)   # 等待端口释放


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SITL 全自动实验编排器")
    parser.add_argument("--phase", choices=WIND_PHASES + ["all"], default="all",
                        help="指定单个风场阶段（默认 all = 全部跑）")
    parser.add_argument("--mode", choices=MODES + ["all"], default="all",
                        help="指定单个估计模式（默认 all = 全部跑）")
    args = parser.parse_args()

    phases = WIND_PHASES if args.phase == "all" else [args.phase]
    modes  = MODES       if args.mode  == "all" else [args.mode]

    results: dict[str, list[Path]] = {m: [] for m in modes}

    print("\n" + "="*60)
    print("  SITL 闭环实验编排器")
    print(f"  风场阶段: {phases}")
    print(f"  估计模式: {modes}")
    total = len(phases) * len(modes)
    print(f"  共 {total} 组实验，每组约 {EXPERIMENT_DURATION_S//60+2} 分钟")
    estimated_min = total * (EXPERIMENT_DURATION_S + PX4_STARTUP_WAIT_S + TAKEOFF_WAIT_S + 30) // 60
    print(f"  预计总时长: ~{estimated_min} 分钟")
    print("="*60)

    # 先跑 baseline，再跑 pirnn_akf（减少重启次数的顺序）
    run_idx = 1
    for mode in sorted(modes):
        for phase in phases:
            csv_path = run_one_experiment(phase, mode, run_idx)
            if csv_path is not None:
                results[mode].append(csv_path)
            run_idx += 1
            time.sleep(5)

    # 分析
    print("\n" + "="*60)
    print("  实验完成，开始生成分析图表")
    print("="*60)

    baseline_csvs = results.get("baseline", [])
    pirnn_csvs    = results.get("pirnn_akf", [])

    if baseline_csvs or pirnn_csvs:
        out_prefix = str(_RESULTS / "comparison")
        analyze_args = [str(_VENV_PYTHON), str(_ROOT / "scripts" / "analyze_sitl_closedloop.py")]
        if baseline_csvs:
            analyze_args += ["--baseline"] + [str(p) for p in baseline_csvs]
        if pirnn_csvs:
            analyze_args += ["--pirnn"] + [str(p) for p in pirnn_csvs]
        analyze_args += ["--output", out_prefix]
        subprocess.run(analyze_args, check=False)
        print(f"\n  输出图表: {out_prefix}_*.png")
        print(f"  数值摘要: {out_prefix}_summary.txt")
    else:
        print("  ⚠ 没有成功的实验结果，跳过分析")

    print("\n全部实验完成！")


if __name__ == "__main__":
    main()
