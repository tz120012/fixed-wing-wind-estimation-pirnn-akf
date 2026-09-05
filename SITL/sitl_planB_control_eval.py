#!/usr/bin/env python3
"""
sitl_planB_control_eval.py  —  方案B：SITL 闭环控制对比实验
=============================================================

实验设计（两阶段）
------------------
  Phase 1 – Baseline (60s):
    PX4 SITL 飞行，**不注入** WIND_COV，仅用 EKF2 自估计风速做控制
  Phase 2 – PIRNN-AKF Injection (60s):
    启动 8_dataset_replay_eval.py（后台进程），通过 MAVLink UDP
    连续向 PX4 注入 WIND_COV（每帧 ~2 ms 推理延迟，50 Hz 节奏）

测量指标
--------
  - 空速跟踪误差 |airspeed - 12.0 m/s|（MAVLink VFR_HUD）
  - 油门通道输出标准差（MAVLink SERVO_OUTPUT_RAW channel 3）
  - 升降舵输出标准差（MAVLink SERVO_OUTPUT_RAW channel 2）
  - PX4 报告风速（MAVLink WIND_COV）
  - 推理注入率统计（仅 Phase 2）

运行方式
--------
  python3 sitl_planB_control_eval.py [--wind steady] [--phase-dur 60]
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread, Event
from typing import Optional

# ── 路径 ──────────────────────────────────────────────────────────────────────
SITL_DIR  = Path(__file__).resolve().parent
ROOT      = SITL_DIR.parent
SRC_DIR   = ROOT / "src"
DSGEN_DIR = SRC_DIR / "dataset_generation"
SCRIPTS_DIR = DSGEN_DIR / "scripts"
LOG_DIR   = SITL_DIR / "logs"
RESULTS   = SITL_DIR / "results"
_PX4_ROOT = Path.home() / "wind_datasets" / "PX4-Autopilot"

# gRPC / 代理隔离（必须在 grpc/mavsdk import 之前）
import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
for _pv in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
    os.environ[_pv] = ""
os.environ["grpc_proxy"] = ""
os.environ["no_grpc_proxy"] = "*"
os.environ["no_proxy"] = os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1,0.0.0.0"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(DSGEN_DIR))    # lib.infra / lib.config 从这里访问

LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── 全局 ──────────────────────────────────────────────────────────────────────
CRUISE_SPEED  = 12.0
ORBIT_RADIUS  = 150.0
ORBIT_ALT     = 70.0
MAV_UDP       = "udpin:0.0.0.0:14550"   # 评估脚本连接地址
REPLAY_SCRIPT = str(SRC_DIR / "8_dataset_replay_eval.py")
VENV_PYTHON   = str(ROOT / ".venv" / "bin" / "python3")

# 确定具有完整依赖的 Python 解释器（mavsdk + torch + pymavlink）
import shutil as _shutil
_REPLAY_PYTHON = "/usr/bin/python3.8"
if not _shutil.which("python3.8"):
    # 回退到系统 python3（可能缺 mavsdk，但 replay 不需要）
    _REPLAY_PYTHON = sys.executable


# ─────────────────────────────────────────────────────────────────────────────
# MAVLink 监听器（独立线程）
# ─────────────────────────────────────────────────────────────────────────────
class MavlinkMetricRecorder:
    """后台线程持续读取 MAVLink 消息，按相位记录控制指标。"""

    TARGET_AIRSPEED = CRUISE_SPEED

    def __init__(self, connection_str: str = "udpin:0.0.0.0:14555"):
        self.conn_str = connection_str
        self._stop   = Event()
        self._phase  = "idle"   # "baseline" | "injection" | "idle"
        self._baseline_rows: list[dict] = []
        self._injection_rows: list[dict] = []
        self._thread: Optional[Thread] = None

    # ── 控制 ──────────────────────────────────────────────────────────────────
    def start(self):
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def set_phase(self, phase: str):
        print(f"[METRIC] → 切换到阶段: {phase}")
        self._phase = phase

    # ── 主循环 ────────────────────────────────────────────────────────────────
    def _run(self):
        from pymavlink import mavutil

        print(f"[METRIC] 连接 MAVLink: {self.conn_str}")
        conn = mavutil.mavlink_connection(self.conn_str, input=False, source_system=255)
        # 请求 SERVO_OUTPUT_RAW + VFR_HUD + WIND_COV
        for msg_id, rate in [(36, 20), (74, 20), (231, 5)]:   # IDs: VFR_HUD, VFR_HUD, WIND
            try:
                conn.mav.request_data_stream_send(
                    1, 1,
                    mavutil.mavlink.MAV_DATA_STREAM_ALL,
                    20, 1
                )
            except Exception:
                pass

        last_servo: dict = {}
        last_airspeed: float = 0.0
        last_wind_cov: dict = {}

        while not self._stop.is_set():
            msg = conn.recv_match(blocking=True, timeout=0.05)
            if msg is None:
                continue
            mtype = msg.get_type()
            t_now = time.time()

            if mtype == "VFR_HUD":
                last_airspeed = float(msg.airspeed)
            elif mtype == "SERVO_OUTPUT_RAW":
                last_servo = {
                    "thr": float(msg.servo3_raw),    # throttle
                    "ele": float(msg.servo2_raw),    # elevator
                    "ail": float(msg.servo1_raw),    # aileron
                    "rud": float(msg.servo4_raw),    # rudder
                }
            elif mtype == "WIND_COV":
                last_wind_cov = {
                    "wn": float(msg.wind_x),
                    "we": float(msg.wind_y),
                }

            # 每 20 Hz 记录一行
            if self._phase in ("baseline", "injection") and last_servo:
                row = {
                    "time": t_now,
                    "airspeed": last_airspeed,
                    "as_err": abs(last_airspeed - self.TARGET_AIRSPEED),
                    **last_servo,
                    **{f"wnd_{k}": v for k, v in last_wind_cov.items()},
                }
                if self._phase == "baseline":
                    self._baseline_rows.append(row)
                else:
                    self._injection_rows.append(row)

        conn.close()
        print("[METRIC] 记录线程已退出")

    # ── 取结果 ────────────────────────────────────────────────────────────────
    def get_rows(self, phase: str) -> list[dict]:
        return self._baseline_rows if phase == "baseline" else self._injection_rows

    def save_csv(self, phase: str, path: Path):
        rows = self.get_rows(phase)
        if not rows:
            print(f"[METRIC] !! {phase} 无数据，不保存")
            return
        fields = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"[METRIC] {phase} 已保存: {path} ({len(rows)} 行)")


# ─────────────────────────────────────────────────────────────────────────────
# 主实验流程
# ─────────────────────────────────────────────────────────────────────────────
async def run_planB(wind_phase: str, phase_dur: float):
    # 动态加载 SITL 基础设施
    from lib.infra import Px4SitlProcess, MavsdkServerProcess, clear_px4_lock_files
    from lib.infra.px4 import clear_px4_rootfs_state
    from lib.config.runtime import load_runtime_config
    from flight_controller import FlightController
    import importlib.util

    # 加载风场配置
    spec = importlib.util.spec_from_file_location("sitl_launch", SITL_DIR / "sitl_launch_and_eval.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    set_wind_config = m.set_wind_config

    runtime = load_runtime_config()
    ts = time.strftime("%Y%m%d_%H%M%S")

    # ── 1. 清理旧进程 ──────────────────────────────────────────────────────────
    print("[PLAN-B] 清理残留进程...")
    for cmd in [["pkill","-9","-x","px4"],["pkill","-9","-x","JSBSim"],
                ["pkill","-9","-f","jsbsim_bridge"],["pkill","-9","-f","mavsdk_server"],
                ["pkill","-9","-f","8_dataset_replay_eval"]]:
        subprocess.run(cmd, stderr=subprocess.DEVNULL)
    await asyncio.sleep(3)
    clear_px4_lock_files()
    clear_px4_rootfs_state(_PX4_ROOT)

    # ── 2. 写风场 ──────────────────────────────────────────────────────────────
    if not set_wind_config(wind_phase):
        raise RuntimeError(f"风场配置失败: {wind_phase}")

    # ── 3. 启动 PX4 SITL ───────────────────────────────────────────────────────
    px4_log = LOG_DIR / f"px4_planB_{wind_phase}_{ts}.log"
    print(f"[PLAN-B] 启动 PX4 SITL → {px4_log}")
    px4_ctx = Px4SitlProcess(px4_root=_PX4_ROOT, runtime=runtime, redirect_log_to=px4_log)
    await px4_ctx.start()

    mavsdk_ctx = MavsdkServerProcess(runtime=runtime, log_dir=LOG_DIR)
    await mavsdk_ctx.start()
    await asyncio.sleep(3)

    fc = FlightController(mavsdk_server_address="127.0.0.1")
    await fc.connect()
    print("[PLAN-B] FlightController 已连接")

    # ── 4. 设置参数 ────────────────────────────────────────────────────────────
    for name, val in [("COM_RCL_EXCEPT", 7), ("NAV_RCL_ACT", 0),
                      ("EKF2_GPS_CHECK", 0), ("COM_OBL_ACT", -1),
                      ("ASPD_DO_CHECKS", 0)]:
        try: await fc.drone.param.set_param_int(name, val)
        except: pass
    for name, val in [("FW_AIRSPD_TRIM", 12.0), ("FW_AIRSPD_MIN", 10.0),
                      ("FW_AIRSPD_MAX", 20.0), ("COM_POS_FS_EPH", 100.0),
                      ("COM_POS_FS_EPV", 100.0), ("COM_VEL_FS_EVH", 10.0),
                      ("COM_LKDOWN_TKO", 0.0)]:
        try: await fc.drone.param.set_param_float(name, val)
        except: pass
    print("[PLAN-B] PX4 参数已设置")

    # ── 5. 等待 EKF 预热 + 起飞 ───────────────────────────────────────────────
    print("[PLAN-B] 等待 EKF 预热 (15s)...")
    await asyncio.sleep(15)

    for attempt in range(1, 4):
        print(f"[PLAN-B] 起飞到 {ORBIT_ALT}m（尝试 {attempt}/3）...")
        try:
            await fc.arm_and_takeoff(
                altitude=ORBIT_ALT,
                takeoff_timeout_s=runtime.timeouts.takeoff_total_s,
                airspeed_ready_timeout_s=runtime.timeouts.airspeed_ready_s,
                airspeed_stable_s=runtime.timeouts.airspeed_stable_s,
                position_stream_max_failures=runtime.timeouts.position_stream_max_failures,
            )
            print("[PLAN-B] ✓ 起飞成功")
            break
        except Exception as e:
            print(f"[PLAN-B] 起飞失败: {e}")
            if attempt >= 3:
                await px4_ctx.stop()
                await mavsdk_ctx.stop()
                raise RuntimeError("起飞失败 3 次，放弃")
            await asyncio.sleep(5)

    await asyncio.sleep(20)   # 等飞机稳定

    # ── 6. 启动盘旋（后台任务）──────────────────────────────────────────────────
    async def _orbit_forever():
        dirs = ["cw", "ccw"]
        i = 0
        while True:
            try:
                await fc.fly_orbit(
                    radius=ORBIT_RADIUS, altitude=ORBIT_ALT,
                    direction=dirs[i % 2], duration=240.0, speed=CRUISE_SPEED,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[FLIGHT] fly_orbit 异常: {e}")
            i += 1

    orbit_task = asyncio.create_task(_orbit_forever())
    print("[PLAN-B] OFFBOARD 盘旋已启动")
    await asyncio.sleep(10)  # 让盘旋先稳定 10s

    # ── 7. 启动 MAVLink 指标录制（独立线程，监听 14555 即 PX4 广播）──────────────
    recorder = MavlinkMetricRecorder(connection_str="udpin:0.0.0.0:14555")
    recorder.start()
    await asyncio.sleep(2)  # 等录制线程就绪

    # ── 8. Phase 1 – Baseline（无注入）────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"[PLAN-B] Phase 1: Baseline（无 WIND_COV 注入）{phase_dur}s")
    print(f"{'='*56}")
    recorder.set_phase("baseline")
    await asyncio.sleep(phase_dur)
    recorder.set_phase("idle")

    baseline_csv = RESULTS / f"planB_baseline_{wind_phase}_{ts}.csv"
    recorder.save_csv("baseline", baseline_csv)

    # ── 9. 过渡缓冲 5 秒 ─────────────────────────────────────────────────────
    print("[PLAN-B] 缓冲 5s 后开始注入...")
    await asyncio.sleep(5)

    # ── 10. Phase 2 – 启动 dataset_replay 注入（子进程）──────────────────────
    print(f"\n{'='*56}")
    print(f"[PLAN-B] Phase 2: PIRNN-AKF WIND_COV 注入 {phase_dur}s")
    print(f"{'='*56}")

    replay_log = LOG_DIR / f"planB_replay_{ts}.log"
    replay_proc = await asyncio.create_subprocess_exec(
        _REPLAY_PYTHON, REPLAY_SCRIPT,
        "--dataset", "test_ood",
        "--inject-wind-cov",
        "--connection", "udpout:127.0.0.1:14550",
        "--rate", "50",
        "--max-samples", str(int(phase_dur * 50 * 2)),  # 足够多
        "--output-dir", str(RESULTS),
        stdout=open(replay_log, "w"),
        stderr=subprocess.STDOUT,
    )
    print(f"[PLAN-B] dataset_replay 进程已启动 (PID={replay_proc.pid})，日志: {replay_log}")
    await asyncio.sleep(2)   # 等推理就绪

    recorder.set_phase("injection")
    await asyncio.sleep(phase_dur)
    recorder.set_phase("idle")

    # 停止 replay 进程
    try:
        replay_proc.terminate()
        await asyncio.wait_for(replay_proc.wait(), timeout=5)
    except Exception:
        replay_proc.kill()
    print(f"[PLAN-B] dataset_replay 进程已停止")

    injection_csv = RESULTS / f"planB_injection_{wind_phase}_{ts}.csv"
    recorder.save_csv("injection", injection_csv)

    # ── 11. 清理 ──────────────────────────────────────────────────────────────
    recorder.stop()
    orbit_task.cancel()
    try:
        await orbit_task
    except asyncio.CancelledError:
        pass

    await fc.disconnect()
    await mavsdk_ctx.stop()
    await px4_ctx.stop()

    print(f"\n[PLAN-B] ✓ 实验完成")
    print(f"  Baseline CSV : {baseline_csv}")
    print(f"  Injection CSV: {injection_csv}")
    return baseline_csv, injection_csv


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wind",      default="steady",
                        help="风场配置（steady / dryden）")
    parser.add_argument("--phase-dur", type=float, default=60.0,
                        help="每阶段秒数（默认 60s）")
    args = parser.parse_args()

    asyncio.run(run_planB(wind_phase=args.wind, phase_dur=args.phase_dur))


if __name__ == "__main__":
    main()
