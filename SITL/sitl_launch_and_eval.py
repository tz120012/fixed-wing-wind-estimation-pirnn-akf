#!/usr/bin/env python3.8
"""
sitl_launch_and_eval.py
=======================
借用 generate_dataset.py 的成熟 PX4 SITL 启动 + 起飞基础设施，
在飞机稳定巡航后立即执行闭环评估脚本。

用法
----
  python3.8 sitl_launch_and_eval.py --mode baseline --wind steady \
      --output ../SITL/results/sitl_baseline_YYYYMMDD.csv

  python3.8 sitl_launch_and_eval.py --mode pirnn_akf --wind steady \
      --output ../SITL/results/sitl_pirnn_akf_YYYYMMDD.csv
"""

import os
import sys

# ── gRPC / 代理隔离（必须在 grpc/mavsdk import 之前）──────────────────────────
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
for _pv in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
    os.environ[_pv] = ""
os.environ["grpc_proxy"] = ""
os.environ["no_grpc_proxy"] = "*"
os.environ["no_proxy"] = os.environ["NO_PROXY"] = "127.0.0.1,localhost,::1,0.0.0.0"

import argparse
import asyncio
import shutil
import subprocess
import time
from pathlib import Path

# ── 路径 ─────────────────────────────────────────────────────────────────────
SITL_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = SITL_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
DSGEN_DIR    = SRC_DIR / "dataset_generation"
SCRIPTS_DIR  = DSGEN_DIR / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(DSGEN_DIR))

_PX4_ROOT = Path.home() / "wind_datasets" / "PX4-Autopilot"
WIND_CFG_DST = _PX4_ROOT / "Tools" / "jsbsim_bridge" / "wind_config.txt"
WIND_CFG_SRC = SITL_DIR / "wind_configs"

LOG_DIR = SITL_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def set_wind_config(phase: str) -> bool:
    src = WIND_CFG_SRC / f"{phase}.txt"
    if not src.exists():
        print(f"[LAUNCH] 风场配置不存在: {src}")
        return False
    WIND_CFG_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, WIND_CFG_DST)
    print(f"[LAUNCH] 风场已设置: {phase}")
    return True


async def _upload_and_start_mission(fc, altitude: float = 70.0, speed: float = 12.0):
    """
    上传一个简单的 4 航点来回 Mission（North→South→North→South…），
    使 PX4 在 MISSION/AUTO 模式下追踪直线轨迹并报告真实 XTE。
    航点沿当前位置北向 ±600m 设置，间距 1200m。
    """
    from mavsdk.mission import MissionItem, MissionPlan
    import math

    # 1. 获取当前位置
    async def _get_pos():
        async for pos in fc.drone.telemetry.position():
            return pos
    pos = await asyncio.wait_for(_get_pos(), timeout=10.0)
    lat0, lon0 = pos.latitude_deg, pos.longitude_deg
    print(f"[MISSION] 当前位置: lat={lat0:.6f}°, lon={lon0:.6f}°, alt={pos.relative_altitude_m:.1f}m")

    # 2. 计算航点（WGS84 近似：1° lat ≈ 111320 m）
    delta_lat = 600.0 / 111320.0   # 600m 北/南偏移
    delta_lon = 600.0 / (111320.0 * math.cos(math.radians(lat0)))  # 600m 东/西偏移

    def make_wp(lat, lon, spd=speed, fly_through=True):
        return MissionItem(
            latitude_deg=lat,
            longitude_deg=lon,
            relative_altitude_m=altitude,
            speed_m_s=spd,
            is_fly_through=fly_through,
            gimbal_pitch_deg=float("nan"),
            gimbal_yaw_deg=float("nan"),
            camera_action=MissionItem.CameraAction.NONE,
            loiter_time_s=0.0,
            camera_photo_interval_s=0.0,
            acceptance_radius_m=25.0,
            yaw_deg=float("nan"),
            camera_photo_distance_m=0.0,
            vehicle_action=MissionItem.VehicleAction.NONE,
        )

    # 北→南→北→南 4 个航点（确保来回均有直线段）
    waypoints = [
        make_wp(lat0 + delta_lat,  lon0),           # P1: 正北 600m
        make_wp(lat0 - delta_lat,  lon0),           # P2: 正南 600m
        make_wp(lat0 + delta_lat,  lon0),           # P3: 回北
        make_wp(lat0 - delta_lat,  lon0),           # P4: 再南（loiter）
    ]
    mission_plan = MissionPlan(waypoints)

    # 3. 上传任务
    print(f"[MISSION] 上传 {len(waypoints)} 个航点 @ 速度={speed} m/s, 高度={altitude} m")
    await fc.drone.mission.upload_mission(mission_plan)

    # 4. 设置任务结束后返回（HOLD）
    await fc.drone.mission.set_return_to_launch_after_mission(False)

    # 5. 先置位到第 0 个任务项，等待 PX4 准备好
    await fc.drone.mission.set_current_mission_item(0)
    await asyncio.sleep(3)

    # 6. 启动任务（含重试）
    for attempt in range(1, 4):
        try:
            await fc.drone.mission.start_mission()
            print("[MISSION] ✓ Mission 已启动")
            return
        except Exception as e:
            print(f"[MISSION] start_mission 失败（尝试 {attempt}/3）: {e}")
            await asyncio.sleep(3)

    # 7. 最终备选：直接用 MAVLink 命令切 MISSION 模式
    print("[MISSION] 备选：直接发 MAV_CMD_DO_SET_MODE 切 MISSION 模式")
    try:
        import pymavlink.mavutil as _mav
        conn = _mav.mavlink_connection("udpin:0.0.0.0:14550")
        conn.wait_heartbeat(timeout=5)
        conn.mav.command_long_send(
            conn.target_system, conn.target_component,
            _mav.mavlink.MAV_CMD_DO_SET_MODE,
            0,
            _mav.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            4,  # PX4 AUTO.MISSION = custom mode 4
            0, 0, 0, 0, 0,
        )
        await asyncio.sleep(2)
        conn.close()
        print("[MISSION] ✓ 已发送 MISSION 模式切换命令")
    except Exception as e2:
        print(f"[MISSION] 备选方案也失败: {e2}，继续（PX4 将维持 HOLD）")


async def run_experiment(mode: str, wind_phase: str, output_csv: str,
                         eval_duration: float, venv_python: str):
    """
    全自动：清理环境 → 写风场 → 启动 PX4 → 起飞 → 执行评估 → 关闭 PX4
    """
    from lib.infra import Px4SitlProcess, MavsdkServerProcess, clear_px4_lock_files
    from lib.infra.px4 import clear_px4_rootfs_state
    from lib.config.runtime import load_runtime_config

    from flight_controller import FlightController

    runtime = load_runtime_config()

    # 1. 清理旧进程
    print("[LAUNCH] 清理残留进程...")
    for cmd in [["pkill","-9","-x","px4"],["pkill","-9","-x","JSBSim"],
                ["pkill","-9","-f","jsbsim_bridge"],["pkill","-9","-f","mavsdk_server"]]:
        subprocess.run(cmd, stderr=subprocess.DEVNULL)
    await asyncio.sleep(3)
    clear_px4_lock_files()
    clear_px4_rootfs_state(_PX4_ROOT)

    # 2. 写风场配置
    if not set_wind_config(wind_phase):
        raise RuntimeError(f"风场配置失败: {wind_phase}")

    # 3. 启动 PX4 SITL
    px4_log = LOG_DIR / f"px4_{mode}_{wind_phase}_{int(time.time())}.log"
    print(f"[LAUNCH] 启动 PX4 SITL → 日志: {px4_log}")

    px4_ctx = Px4SitlProcess(
        px4_root=_PX4_ROOT,
        runtime=runtime,
        redirect_log_to=px4_log,
    )
    await px4_ctx.start()
    print("[LAUNCH] PX4 SITL 已启动")

    # 4. 启动 mavsdk_server
    mavsdk_log = LOG_DIR / f"mavsdk_{mode}_{int(time.time())}.log"
    mavsdk_ctx = MavsdkServerProcess(runtime=runtime, log_dir=LOG_DIR)
    await mavsdk_ctx.start()
    print("[LAUNCH] mavsdk_server 已启动")
    await asyncio.sleep(3)

    # 5. FlightController 连接
    fc = FlightController(mavsdk_server_address="127.0.0.1")
    await fc.connect()
    print("[LAUNCH] FlightController 已连接")

    # 6. 设置关键参数
    params_int = [
        ("COM_RCL_EXCEPT", 7),
        ("NAV_RCL_ACT", 0),
        ("EKF2_GPS_CHECK", 0),
        ("COM_OBL_ACT", -1),
        ("ASPD_DO_CHECKS", 0),
    ]
    params_float = [
        ("COM_POS_FS_EPH", 100.0),
        ("COM_POS_FS_EPV", 100.0),
        ("COM_VEL_FS_EVH", 10.0),
        ("COM_LKDOWN_TKO", 0.0),
        ("FW_AIRSPD_TRIM",  14.0),   # 温和上调靠拢训练巡航 (实测均值 15.99); MIN 保持10保证起飞爬升
        ("FW_AIRSPD_MIN",   10.0),
        ("FW_AIRSPD_MAX",   20.0),
    ]
    print("[LAUNCH] 设置 PX4 参数...")
    for name, val in params_int:
        try:
            await fc.drone.param.set_param_int(name, val)
            print(f"  ✓ {name}={val}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    for name, val in params_float:
        try:
            await fc.drone.param.set_param_float(name, val)
            print(f"  ✓ {name}={val}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # 7. 等待 EKF 预热
    print("[LAUNCH] 等待 EKF 预热 (15s)...")
    await asyncio.sleep(15)

    # 8. 起飞（含重试）
    MAX_TAKEOFF_RETRIES = 3
    takeoff_ok = False
    last_err = None
    for attempt in range(1, MAX_TAKEOFF_RETRIES + 1):
        print(f"[LAUNCH] 起飞到 70m（尝试 {attempt}/{MAX_TAKEOFF_RETRIES}）...")
        try:
            await fc.arm_and_takeoff(
                altitude=70.0,
                takeoff_timeout_s=runtime.timeouts.takeoff_total_s,
                airspeed_ready_timeout_s=runtime.timeouts.airspeed_ready_s,
                airspeed_stable_s=runtime.timeouts.airspeed_stable_s,
                position_stream_max_failures=runtime.timeouts.position_stream_max_failures,
            )
            print("[LAUNCH] ✓ 起飞成功！")
            takeoff_ok = True
            break
        except Exception as e:
            last_err = e
            print(f"[LAUNCH] ✗ 起飞失败: {e}")
            if attempt >= MAX_TAKEOFF_RETRIES:
                break
            print(f"[LAUNCH] 重启 PX4 SITL 后重试...")
            try:
                await fc.disconnect()
            except Exception:
                pass
            await mavsdk_ctx.stop()
            await px4_ctx.stop()
            await asyncio.sleep(5)
            # 重新启动
            clear_px4_rootfs_state(_PX4_ROOT)
            clear_px4_lock_files()
            px4_ctx = Px4SitlProcess(
                px4_root=_PX4_ROOT,
                runtime=runtime,
                redirect_log_to=px4_log,
            )
            await px4_ctx.start()
            mavsdk_ctx = MavsdkServerProcess(runtime=runtime, log_dir=LOG_DIR)
            await mavsdk_ctx.start()
            await asyncio.sleep(3)
            fc = FlightController(mavsdk_server_address="127.0.0.1")
            await fc.connect()
            # 重设参数
            for name, val in params_int:
                try:
                    await fc.drone.param.set_param_int(name, val)
                except Exception:
                    pass
            for name, val in params_float:
                try:
                    await fc.drone.param.set_param_float(name, val)
                except Exception:
                    pass
            await asyncio.sleep(15)

    if not takeoff_ok:
        await px4_ctx.stop()
        await mavsdk_ctx.stop()
        raise RuntimeError(f"起飞失败（{MAX_TAKEOFF_RETRIES} 次后放弃）: {last_err}")

    # 9. 稳定飞行 20 秒（给 PX4 TECS 和 EKF2 足够预热时间）
    print("[LAUNCH] 等待 20s 让飞机稳定在 70m...")
    await asyncio.sleep(20)

    # 10. 切换到 OFFBOARD 盘旋（fly_orbit，与训练数据工况完全一致）
    CRUISE_SPEED = 14.0   # 温和上调靠拢训练巡航空速 (实测均值 15.99 m/s)
    ORBIT_RADIUS = 150.0  # 盘旋半径 [m]，Rascal110 最小转弯半径 ~80m

    async def _orbit_forever():
        """交替顺逆时针盘旋，直到被取消。"""
        directions = ["cw", "ccw"]
        idx = 0
        seg = 180.0  # 每段盘旋 3 分钟
        print(f"[FLIGHT] 开始 OFFBOARD 盘旋 @ speed={CRUISE_SPEED} m/s, r={ORBIT_RADIUS}m")
        while True:
            d = directions[idx % 2]
            try:
                await fc.fly_orbit(
                    radius=ORBIT_RADIUS,
                    altitude=70.0,
                    direction=d,
                    duration=seg,
                    speed=CRUISE_SPEED,
                )
            except asyncio.CancelledError:
                print("[FLIGHT] 盘旋任务已取消")
                raise
            except Exception as fe:
                print(f"[FLIGHT] fly_orbit 异常: {fe}")
            idx += 1

    flight_task = asyncio.create_task(_orbit_forever())
    print("[LAUNCH] OFFBOARD 盘旋任务已启动")
    await asyncio.sleep(5)  # 等待 offboard 建立

    # 11. 启动评估脚本（子进程，使用 venv Python）
    eval_script = SRC_DIR / "7_sitl_closedloop_eval.py"
    config_sitl = PROJECT_ROOT / "config" / "config_sitl.yaml"
    if not config_sitl.exists():
        config_sitl = PROJECT_ROOT / "config" / "config.yaml"

    if mode == "online_6c":
        # 直接跑改写后的 45 维在线部署 6c（真实 MAVLink→特征映射端到端验证）。
        # 6c 自身把估计值与 JSBSim 真风(WIND 消息)一并写入 hitl_data_*.csv。
        online_script = SRC_DIR / "6c_online_deployment_pigru-akf_rsbpi.py"
        cmd = [
            venv_python, str(online_script),
            "--config", str(config_sitl),
            "--phase", "steady",
            "--duration", str(int(eval_duration)),
        ]
    elif mode == "pirnn_akf_replay":
        # 方案B：数据集重播注入模式（规避 HIGHRES_IMU + OOD 问题）
        replay_script = SRC_DIR / "8_dataset_replay_eval.py"
        cmd = [
            venv_python, str(replay_script),
            "--dataset", "test_ood",
            "--inject-wind-cov",
            "--connection", "udpout:127.0.0.1:14550",
            "--rate", "50",
            "--max-samples", str(int(eval_duration * 50 * 3)),
            "--output-dir", str(SITL_DIR / "results"),
        ]
    else:
        cmd = [
            venv_python, str(eval_script),
            "--mode", mode,
            "--connection", "udpin:0.0.0.0:14550",
            "--output", output_csv,
            "--duration", str(int(eval_duration)),
            "--target-airspeed", str(CRUISE_SPEED),
            "--config", str(config_sitl),
        ]
    print(f"[LAUNCH] 启动评估: {' '.join(cmd)}")
    eval_proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(SRC_DIR))
    print(f"[LAUNCH] 评估进程 PID={eval_proc.pid}，时长={eval_duration}s")

    # 12. 等待评估完成
    ret = await eval_proc.wait()
    print(f"[LAUNCH] 评估完成，退出码={ret}")

    # 停止飞行任务
    flight_task.cancel()
    try:
        await flight_task
    except asyncio.CancelledError:
        pass

    # 12. 清理
    print("[LAUNCH] 清理 PX4 SITL...")
    await fc.disconnect()
    await mavsdk_ctx.stop()
    await px4_ctx.stop()
    print("[LAUNCH] 全部完成")
    return ret


def main():
    parser = argparse.ArgumentParser(description="SITL 全自动实验（PX4启动 + 起飞 + 评估）")
    parser.add_argument("--mode", required=True,
                        choices=["baseline", "pirnn_akf", "pirnn_akf_replay", "online_6c"])
    parser.add_argument("--wind", default="steady",
                        choices=["steady", "steady_noturb", "steady_indist",
                                 "gust_light", "gust_strong"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--duration", type=float, default=360.0,
                        help="评估时长 [s]，默认 360s")
    parser.add_argument("--venv-python", default=None,
                        help="评估脚本使用的 Python 路径（默认自动查找 venv）")
    args = parser.parse_args()

    # 自动生成输出路径
    if args.output is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        results_dir = SITL_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        mode_label = args.mode
        args.output = str(results_dir / f"sitl_{mode_label}_{ts}.csv")

    # 查找 venv Python
    venv_py = args.venv_python
    if venv_py is None:
        candidates = [
            str(PROJECT_ROOT / ".venv" / "bin" / "python3"),
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            "python3",
        ]
        for c in candidates:
            if Path(c).exists() or shutil.which(c):
                venv_py = c
                break
    print(f"[LAUNCH] 评估脚本 Python: {venv_py}")
    print(f"[LAUNCH] 模式: {args.mode}  风场: {args.wind}")
    print(f"[LAUNCH] 输出: {args.output}")

    asyncio.run(run_experiment(
        mode=args.mode,
        wind_phase=args.wind,
        output_csv=args.output,
        eval_duration=args.duration,
        venv_python=venv_py,
    ))


if __name__ == "__main__":
    main()
