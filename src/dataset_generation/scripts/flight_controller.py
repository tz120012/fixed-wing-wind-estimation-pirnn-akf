"""
flight_controller.py
负责控制无人机执行各种飞行机动（MAVSDK + PX4 SITL，端口 14540）
"""

import asyncio
import math
import os
import sys
from pathlib import Path

from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, PositionNedYaw, AccelerationNed, OffboardError
import numpy as np

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

try:
    from lib.errors import TakeoffFailed, GpsTimeout, ArmTimeout, FlightControllerError
except Exception:
    class FlightControllerError(Exception):
        pass

    class TakeoffFailed(FlightControllerError):
        pass

    class GpsTimeout(FlightControllerError):
        pass

    class ArmTimeout(FlightControllerError):
        pass


class FlightController:
    def __init__(self, mavsdk_server_address=None, ned_altitude_sign=-1.0):
        """
        mavsdk_server_address: 若已显式启动 mavsdk_server（如 127.0.0.1），则传入以避免 SDK 自启失败导致 Connection refused。
        """
        self._mavsdk_server_address = mavsdk_server_address
        self._ned_altitude_sign = float(ned_altitude_sign)
        if self._ned_altitude_sign not in {-1.0, 0.0, 1.0}:
            raise ValueError(
                "ned_altitude_sign must be -1 (standard), +1, or 0 (magnitude)"
            )
        self.drone = System(mavsdk_server_address=mavsdk_server_address) if mavsdk_server_address else System()
        self.is_connected = False
        self._manual_heartbeat_task = None
        self._connection_lost = False  # 连接丢失标志

    async def check_connection_health(self):
        """检查 MAVSDK 连接是否健康。返回 True 表示健康，False 表示连接已丢失。"""
        if self._connection_lost:
            return False
        try:
            # 尝试获取一次心跳，超时 2 秒
            async def _get_health():
                async for health in self.drone.telemetry.health_all_ok():
                    return True
                return False
            await asyncio.wait_for(_get_health(), timeout=2.0)
            return True
        except (asyncio.TimeoutError, Exception) as e:
            if "Connection refused" in str(e) or "UNAVAILABLE" in str(e):
                self._connection_lost = True
                print(f"[FlightController] 连接健康检查失败: 连接已丢失 - {e}")
                return False
            # 其他错误视为暂时性问题
            return True

    async def _run_manual_control_heartbeat(self):
        """周期发送中性摇杆量，使 PX4 认为有 MAVLink 手动控制，避免触发 No manual control stick input 导致 RTL。"""
        while True:
            try:
                # x=roll, y=pitch, z=throttle, r=yaw；中性：0,0,0.5,0
                await self.drone.manual_control.set_manual_control_input(0.0, 0.0, 0.5, 0.0)
            except Exception:
                pass
            await asyncio.sleep(0.1)

    async def disconnect(self):
        """断开连接并释放端口，便于下一轮重新 bind。必须在每轮结束后调用，避免下一轮 Address in use。"""
        if self._manual_heartbeat_task and not self._manual_heartbeat_task.done():
            self._manual_heartbeat_task.cancel()
            try:
                await self._manual_heartbeat_task
            except asyncio.CancelledError:
                pass
            self._manual_heartbeat_task = None
        self.is_connected = False

    async def connect(self, system_address="udpin://0.0.0.0:14540", timeout_s=90, retries=10, retry_interval_s=2):
        """连接到 PX4 SITL。显式 mavsdk_server 时无需 system_address。遇 Connection refused 时重试+退避。"""
        last_err = None
        for attempt in range(retries):
            try:
                if self._mavsdk_server_address:
                    await self.drone.connect()
                else:
                    await self.drone.connect(system_address=system_address)
                break
            except Exception as e:
                last_err = e
                if "Connection refused" in str(e) or "50051" in str(e) or "connect" in str(e).lower():
                    if attempt < retries - 1:
                        print(f"[FlightController] gRPC/连接被拒，{retry_interval_s}s 后重试 ({attempt + 1}/{retries})...")
                        await asyncio.sleep(retry_interval_s)
                        continue
                raise

        print("[FlightController] 等待连接...")
        async def _wait_connected():
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    return
        try:
            await asyncio.wait_for(_wait_connected(), timeout=timeout_s)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"连接 PX4 超时（{timeout_s}s）。"
                "通常因 JSBSim bridge 未在 TCP 4560 就绪，PX4 卡在 Waiting for simulator。"
                "请检查: 1) make px4_sitl jsbsim_rascal 是否成功 2) build/.../tmp/rootfs/jsbsim_bridge.log 是否有报错"
            ) from None
        print("[FlightController] 已连接!")
        self.is_connected = True
        self._manual_heartbeat_task = asyncio.create_task(self._run_manual_control_heartbeat())

    async def _wait_until_armable(self, timeout_s=60):
        """
        等待 PX4 预检通过，避免 COMMAND_DENIED。
        注意：跳过陀螺仪校准检查（SITL环境下不影响飞行）。
        """
        deadline = asyncio.get_event_loop().time() + timeout_s
        last_health_status = None
        while asyncio.get_event_loop().time() < deadline:
            try:
                async def _get_health_detail():
                    async for health in self.drone.telemetry.health():
                        return health
                    return None
                
                health = await asyncio.wait_for(_get_health_detail(), timeout=5.0)
                last_health_status = health
                
                if health:
                    # 自定义预检逻辑：跳过陀螺仪校准，只检查关键项
                    critical_checks = [
                        health.is_accelerometer_calibration_ok,  # 加速度计
                        health.is_magnetometer_calibration_ok,   # 磁力计
                        health.is_local_position_ok,             # 本地位置(EKF)
                        health.is_global_position_ok,            # 全球位置(GPS)
                        health.is_home_position_ok,              # Home位置
                        health.is_armable,                       # 可解锁状态
                    ]
                    
                    # 只要关键项全部通过即可（忽略陀螺仪）
                    if all(critical_checks):
                        return True
                
                # 记录最后的健康状态，用于超时后报告
                last_health_status = health
                    
            except (asyncio.TimeoutError, Exception):
                pass
            await asyncio.sleep(2)
        
        # 返回最后的健康状态，供调用者输出详细信息
        return False, last_health_status

    async def _wait_local_position_valid(self, timeout_s=120):
        """等待 EKF 收敛、local_position_valid 为 True，这是 AUTO_TAKEOFF 模式所需的最低条件。"""
        deadline = asyncio.get_event_loop().time() + timeout_s
        last_health = None
        check_count = 0
        
        while asyncio.get_event_loop().time() < deadline:
            try:
                async def _one_health():
                    async for h in self.drone.telemetry.health():
                        return h
                health = await asyncio.wait_for(_one_health(), timeout=5.0)
                last_health = health
                check_count += 1
                
                if health.is_local_position_ok:
                    return True
                    
                # 每10次检查（约20秒）输出一次状态
                if check_count % 10 == 0:
                    print(f"  [EKF诊断 {check_count*2}s] 本地位置: {'✓' if health.is_local_position_ok else '✗'}, "
                          f"全球位置: {'✓' if health.is_global_position_ok else '✗'}, "
                          f"Home: {'✓' if health.is_home_position_ok else '✗'}")
                    
            except (asyncio.TimeoutError, Exception):
                pass
            await asyncio.sleep(2)
        
        # 超时后输出最终状态
        if last_health:
            print(f"  [EKF最终状态] 本地位置: {'✓' if last_health.is_local_position_ok else '✗ 未收敛'}, "
                  f"全球位置: {'✓' if last_health.is_global_position_ok else '✗ 未收敛'}, "
                  f"Home: {'✓' if last_health.is_home_position_ok else '✗ 未设置'}")
            print(f"  [可能原因] eeprom残留参数、GPS信号弱、仿真器未就绪、EKF2_MULTI_IMU=1等异常配置")
        
        return False

    async def _wait_telemetry_valid(self, max_checks=5, check_interval_s=5):
        """
        等待关键遥测数据全部有效：空速、地速、姿态、控制输出。
        JSBSim 各传感器插件初始化有延迟，起飞前必须确认遥测流已就绪，
        否则采集数据会出现 NaN 或全零，导致数据集不可用。

        最多检查 max_checks 次（每次间隔 check_interval_s 秒），
        全部通过返回 True，否则返回 False。
        """
        def _is_valid_float(v):
            """非 None、非 NaN、非 Inf"""
            return v is not None and not math.isnan(v) and not math.isinf(v)

        def _extract_groundspeed_m_s(fw_metrics, pv):
            """兼容不同 MAVSDK 版本：优先读取 fixedwing_metrics，自缺字段时回退到 NED 水平速度。"""
            groundspeed = getattr(fw_metrics, "groundspeed_m_s", None) if fw_metrics is not None else None
            if _is_valid_float(groundspeed):
                return float(groundspeed)

            if pv is None:
                return None

            velocity = getattr(pv, "velocity", None)
            if velocity is None:
                return None

            north = getattr(velocity, "north_m_s", None)
            east = getattr(velocity, "east_m_s", None)
            if not (_is_valid_float(north) and _is_valid_float(east)):
                return None
            return math.hypot(north, east)

        for attempt in range(1, max_checks + 1):
            issues = []

            # --- 1) 空速 & 地速 (fixedwing_metrics + position_velocity_ned) ---
            fw = None
            pv = None
            try:
                async def _get_fw():
                    async for m in self.drone.telemetry.fixedwing_metrics():
                        return m
                fw = await asyncio.wait_for(_get_fw(), timeout=5.0)
                if fw is None:
                    issues.append("fixedwing_metrics=None")
                else:
                    # 注意：延迟风场注入时，飞机在地面静止，空速可能为 0 或 nan，这是正常的
                    # 只检查空速是否为负数（明显异常），允许 nan 和 0
                    if _is_valid_float(fw.airspeed_m_s) and fw.airspeed_m_s < 0:
                        issues.append(f"airspeed={fw.airspeed_m_s}")
            except (asyncio.TimeoutError, Exception) as e:
                issues.append(f"fixedwing_metrics超时({e})")

            try:
                async def _get_pv():
                    async for item in self.drone.telemetry.position_velocity_ned():
                        return item
                pv = await asyncio.wait_for(_get_pv(), timeout=5.0)
                groundspeed = _extract_groundspeed_m_s(fw, pv)
                if not _is_valid_float(groundspeed):
                    issues.append(f"groundspeed={groundspeed}")
            except (asyncio.TimeoutError, Exception) as e:
                issues.append(f"position_velocity_ned超时({e})")

            # --- 2) 姿态 (attitude_euler) ---
            try:
                async def _get_att():
                    async for a in self.drone.telemetry.attitude_euler():
                        return a
                att = await asyncio.wait_for(_get_att(), timeout=5.0)
                if att is None:
                    issues.append("attitude=None")
                else:
                    for name, val in [("roll", att.roll_deg), ("pitch", att.pitch_deg), ("yaw", att.yaw_deg)]:
                        if not _is_valid_float(val):
                            issues.append(f"{name}_deg={val}")
            except (asyncio.TimeoutError, Exception) as e:
                issues.append(f"attitude超时({e})")

            # --- 3) 执行器控制 (actuator_control_target) ---
            try:
                async def _get_act():
                    async for a in self.drone.telemetry.actuator_control_target():
                        return a
                act = await asyncio.wait_for(_get_act(), timeout=5.0)
                if act is None:
                    issues.append("actuator=None")
                else:
                    # group 列表中每个元素应为有效浮点（至少 roll/pitch/yaw/throttle 前4个）
                    controls = act.controls if hasattr(act, 'controls') else (act.group if hasattr(act, 'group') else None)
                    if controls is None or len(controls) < 4:
                        issues.append(f"actuator.controls长度不足({controls})")
                    else:
                        for idx, label in enumerate(["roll_ctrl", "pitch_ctrl", "yaw_ctrl", "throttle_ctrl"]):
                            if not _is_valid_float(controls[idx]):
                                issues.append(f"{label}={controls[idx]}")
            except (asyncio.TimeoutError, Exception) as e:
                issues.append(f"actuator超时({e})")

            # --- 判定 ---
            if not issues:
                print(f"  [遥测预检 {attempt}/{max_checks}] 全部通过 ✓")
                return True
            else:
                print(f"  [遥测预检 {attempt}/{max_checks}] 异常项: {', '.join(issues)}")
                if attempt < max_checks:
                    await asyncio.sleep(check_interval_s)

        # 全部尝试用完，仍有异常
        return False

    async def _wait_airspeed_selector_ready(
        self, total_timeout_s: float = 30.0, stable_window_s: float = 3.0,
    ) -> bool:
        """等待 PX4 airspeed_selector 输出有限 airspeed 并保持稳定 ``stable_window_s``。

        起飞前的关键预检：若 airspeed 持续为 NaN，TECS 会输出 0 油门导致
        飞机无法离地，最终 JSBSim 因攻角越界触发 FGTable 断言崩溃（详见 doc/修改.md）。
        """
        def _is_finite(v):
            return v is not None and not math.isnan(v) and not math.isinf(v)

        deadline = asyncio.get_event_loop().time() + total_timeout_s
        ok_since = None  # type: ignore[var-annotated]
        last_log = 0.0
        last_value = None
        while asyncio.get_event_loop().time() < deadline:
            now = asyncio.get_event_loop().time()
            try:
                async def _one():
                    async for m in self.drone.telemetry.fixedwing_metrics():
                        return m
                fw = await asyncio.wait_for(_one(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                fw = None

            airspeed = getattr(fw, "airspeed_m_s", None) if fw is not None else None
            last_value = airspeed
            if _is_finite(airspeed):
                if ok_since is None:
                    ok_since = now
                if (now - ok_since) >= stable_window_s:
                    print(
                        f"[FlightController] airspeed_selector 已就绪（airspeed={airspeed:.2f}m/s，"
                        f"持续 {now - ok_since:.1f}s 稳定）"
                    )
                    return True
            else:
                ok_since = None

            if (now - last_log) >= 5.0:
                print(
                    f"[FlightController] 等待 airspeed_selector...（当前 airspeed={airspeed}, "
                    f"剩余 {deadline - now:.0f}s）"
                )
                last_log = now
            await asyncio.sleep(0.5)

        print(
            f"[FlightController] 警告: airspeed_selector 在 {total_timeout_s:.0f}s 内未稳定输出"
            f"（最后一次 airspeed={last_value}）"
        )
        return False

    async def arm_and_takeoff(
        self,
        altitude,
        *,
        takeoff_timeout_s: float = 120.0,
        airspeed_ready_timeout_s: float = 30.0,
        airspeed_stable_s: float = 3.0,
        position_stream_max_failures: int = 30,
    ):
        """解锁并起飞。先等待 EKF 收敛（local_position_valid），再 arm + takeoff。

        关键安全约束：
          - airspeed_selector 必须在解锁前输出有限 airspeed 并稳定，否则 TECS 给 0 油门；
          - 监控阶段连续 ``position_stream_max_failures`` 次 NED 流失败即升级异常；
          - 整段 ``takeoff_timeout_s`` 内未达目标高度（±tolerance）抛 ``TakeoffFailed``；
        所有阈值由 ``runtime.yaml::timeouts`` 注入，不再硬编码。
        """
        print(f"[FlightController] 解锁并起飞到 {altitude:.1f}m")

        # 第一步：等待 local_position_valid（AUTO_TAKEOFF 必需条件）
        print("[FlightController] 等待 EKF 收敛（local_position_valid）...")
        if not await self._wait_local_position_valid(timeout_s=150):
            print("[FlightController] 警告: local_position 150s 内未 valid，仍尝试继续")
        else:
            print("[FlightController] EKF local_position 已 valid")

        # 第二步：等待关键遥测数据有效（空速、地速、姿态、控制）
        print("[FlightController] 等待遥测数据有效（空速/地速/姿态/控制）...")
        if not await self._wait_telemetry_valid(max_checks=5, check_interval_s=5):
            raise FlightControllerError(
                "遥测预检失败：空速/地速/姿态/控制数据在 5 次检查后仍异常，"
                "无法保证数据质量，拒绝解锁。本轮需重新运行。"
            )
        print("[FlightController] 遥测数据已有效 ✓")

        # 注意：地面静止时 airspeed=nan 是 PX4 设计行为（差压传感器没风→nan/0），
        # 因此**不在解锁前**强制要求有限 airspeed。改为在起飞监控循环里：飞机离地
        # 一定高度（airborne_alt_threshold_m）后开始监控，连续 nan 超过
        # airspeed_inflight_grace_s 才 abort。

        # 第三步：等待完整预检（跳过陀螺仪）
        print("[FlightController] 等待预检通过（EKF/Home/传感器，跳过陀螺仪）...")
        result = await self._wait_until_armable(timeout_s=30)
                
        # 处理返回值：可能是布尔值（旧版本）或元组（新版本）
        if isinstance(result, tuple):
            is_armable, health_status = result
        else:
            is_armable = result
            health_status = None
                    
        if not is_armable:
            print("[FlightController] 警告: 预检未在 30s 内通过，仍尝试解锁")
            if health_status:
                print("[FlightController] 预检详情:")
                
                # 陀螺仪校准已跳过检查
                gyro_status = "⊘ 已跳过" if not health_status.is_gyrometer_calibration_ok else "✓"
                print(f"  - 陀螺仪校准: {gyro_status} (SITL环境不检查)")
                
                print(f"  - 加速度计校准: {'✓' if health_status.is_accelerometer_calibration_ok else '✗ 失败'}")
                print(f"  - 磁力计校准: {'✓' if health_status.is_magnetometer_calibration_ok else '✗ 失败'}")
                print(f"  - 本地位置(EKF): {'✓' if health_status.is_local_position_ok else '✗ 失败'}")
                print(f"  - 全球位置(GPS): {'✓' if health_status.is_global_position_ok else '✗ 失败'}")
                print(f"  - Home位置: {'✓' if health_status.is_home_position_ok else '✗ 失败'}")
                print(f"  - 可解锁状态: {'✓' if health_status.is_armable else '✗ 失败'}")
        else:
            print("[FlightController] 预检已通过（已跳过陀螺仪校准检查）")

        # ARM 重试逻辑
        arm_error = None
        for attempt in range(6):
            try:
                await self.drone.action.arm()
                print("[FlightController] 解锁成功")
                arm_error = None
                break
            except Exception as e:
                arm_error = e
                if "COMMAND_DENIED" in str(e) and attempt < 5:
                    print(f"[FlightController] 解锁被拒，5s 后重试 ({attempt + 1}/6)")
                    await asyncio.sleep(5)
                else:
                    break
        if arm_error is not None and self._ned_altitude_sign == 0.0:
            # Malolo is used only in SITL.  Its legacy airframe occasionally
            # keeps is_armable false despite all explicit health fields being
            # valid.  Use PX4's documented force-arm magic value as a final
            # simulator-only fallback; never take this path on hardware.
            try:
                from pymavlink import mavutil

                link = mavutil.mavlink_connection(
                    "udpout:127.0.0.1:14540",
                    source_system=255,
                )
                link.mav.command_long_send(
                    1,
                    1,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0,
                    1.0,
                    21196.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                link.close()
                await asyncio.sleep(2.0)
                armed_stream = self.drone.telemetry.armed()
                if await asyncio.wait_for(armed_stream.__anext__(), timeout=3.0):
                    print("[FlightController] Malolo SITL 强制解锁成功")
                    arm_error = None
                else:
                    print("[FlightController] Malolo SITL 强制解锁未被飞控接受")
            except Exception as force_error:
                print(
                    "[FlightController] Malolo SITL 强制解锁失败: "
                    f"{force_error}"
                )
        if arm_error is not None:
            raise ArmTimeout(
                f"解锁在 6 次尝试后失败: {arm_error}",
                context={"attempts": 6, "error": str(arm_error)},
            ) from arm_error

        # TAKEOFF 重试逻辑：增大间隔到 10s，给 EKF 更多时间收敛
        await self.drone.action.set_takeoff_altitude(altitude)
        for attempt in range(8):
            try:
                await self.drone.action.takeoff()
                print(f"[FlightController] 起飞指令已发送，等待爬升到 {altitude:.1f}m...")
                break
            except Exception as e:
                if "COMMAND_DENIED" in str(e) and attempt < 7:
                    print(f"[FlightController] 起飞被拒（可能 local_position 尚未 valid），10s 后重试 ({attempt + 1}/8)")
                    await asyncio.sleep(10)
                else:
                    raise

        TAKEOFF_TOLERANCE_M = 3.0
        PROGRESS_INTERVAL_S = 5
        ALTITUDE_DIVE_THRESHOLD_M = 15.0
        ALTITUDE_DIVE_WINDOW_S = 20.0
        # airspeed 在空检查门限：飞机离地超过此高度才开始检查 airspeed_selector，
        # 因为 PX4 设计上地面静止时 airspeed=nan/0 是合法的。
        AIRSPEED_AIRBORNE_THRESHOLD_M = 15.0
        airspeed_nan_since = None  # type: ignore[var-annotated]
        start_time = asyncio.get_event_loop().time()
        last_report = start_time
        peak_alt = None
        last_alt = None
        consecutive_pv_failures = 0
        last_pv_error: str = ""
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > takeoff_timeout_s:
                raise TakeoffFailed(
                    f"起飞超时：{takeoff_timeout_s:.0f}s 内未达目标高度 "
                    f"{altitude:.1f}m（peak={peak_alt}m, last={last_alt}m, "
                    f"NED 流连续失败={consecutive_pv_failures}, last_err='{last_pv_error}'）。"
                    "疑似 PX4 TECS 异常 / EKF 漂移 / SITL 仿真停滞。",
                    context={
                        "target_alt_m": altitude,
                        "peak_alt_m": peak_alt,
                        "last_alt_m": last_alt,
                        "elapsed_s": elapsed,
                        "consecutive_pv_failures": consecutive_pv_failures,
                    },
                )
            try:
                pv = await self._get_position_velocity_ned()
                current_alt = (
                    abs(pv.position.down_m)
                    if self._ned_altitude_sign == 0.0
                    else self._ned_altitude_sign * pv.position.down_m
                )
                consecutive_pv_failures = 0
                last_pv_error = ""
            except Exception as e:
                consecutive_pv_failures += 1
                last_pv_error = f"{type(e).__name__}: {e}"
                if consecutive_pv_failures >= position_stream_max_failures:
                    raise TakeoffFailed(
                        f"NED 位置流连续失败 {consecutive_pv_failures} 次（约 "
                        f"{consecutive_pv_failures}s）：{last_pv_error}。"
                        "mavsdk_server / gRPC 连接已退化，无法可靠监控起飞。",
                        context={
                            "consecutive_failures": consecutive_pv_failures,
                            "max_failures": position_stream_max_failures,
                            "last_error": last_pv_error,
                        },
                    ) from e
                await asyncio.sleep(1.0)
                continue

            now = asyncio.get_event_loop().time()
            last_alt = current_alt
            if peak_alt is None or current_alt > peak_alt:
                peak_alt = current_alt
            if now - last_report >= PROGRESS_INTERVAL_S:
                print(
                    f"[FlightController] 爬升中... 当前高度={current_alt:.1f}m / "
                    f"目标={altitude:.1f}m ({elapsed:.0f}s)"
                )
                last_report = now

            if elapsed > ALTITUDE_DIVE_WINDOW_S and peak_alt is not None:
                drop = peak_alt - current_alt
                if drop > ALTITUDE_DIVE_THRESHOLD_M and current_alt < 0:
                    raise TakeoffFailed(
                        f"起飞异常：高度从峰值 {peak_alt:.1f}m 下跌至 "
                        f"{current_alt:.1f}m (下跌={drop:.1f}m)，"
                        "疑似 EKF local position 漂移或 PX4 failsafe 导致坐标系原点偏移，需重试。",
                        context={
                            "peak_alt_m": peak_alt,
                            "current_alt_m": current_alt,
                            "drop_m": drop,
                        },
                    )

            # airspeed_selector in-flight 检查：飞机离地后如果空速持续 nan
            # 超过 grace 窗口，说明 PX4 airspeed_selector 没收敛，DataLogger
            # 拿到的将是垃圾数据，提前 abort 让上层重启 SITL。
            if current_alt is not None and current_alt >= AIRSPEED_AIRBORNE_THRESHOLD_M:
                airspeed_now = await self._read_airspeed_safe()
                if airspeed_now is None or math.isnan(airspeed_now) or math.isinf(airspeed_now):
                    if airspeed_nan_since is None:
                        airspeed_nan_since = now
                        print(
                            f"[FlightController] 注意: 高度 {current_alt:.1f}m 已离地，"
                            f"airspeed={airspeed_now}（开始计时，超过 "
                            f"{airspeed_ready_timeout_s:.0f}s 仍 nan 将 abort）"
                        )
                    elif (now - airspeed_nan_since) >= airspeed_ready_timeout_s:
                        raise TakeoffFailed(
                            f"airspeed_selector 在空中持续 {airspeed_ready_timeout_s:.0f}s 未输出有限值"
                            f"（飞机已升至 {current_alt:.1f}m），DataLogger 将拿到垃圾数据，主动 abort。",
                            context={
                                "alt_m": current_alt,
                                "nan_duration_s": now - airspeed_nan_since,
                                "airspeed": airspeed_now,
                            },
                        )
                else:
                    if airspeed_nan_since is not None:
                        print(
                            f"[FlightController] airspeed_selector 已在空中收敛"
                            f"（airspeed={airspeed_now:.2f}m/s）"
                        )
                    airspeed_nan_since = None

            if current_alt >= altitude - TAKEOFF_TOLERANCE_M:
                print(f"[FlightController] 已到达 {altitude:.1f}m（实际={current_alt:.1f}m）")
                return
            await asyncio.sleep(1.0)

    async def _read_airspeed_safe(self):
        """单次读取 fixedwing_metrics.airspeed_m_s，失败返回 ``None``（不抛异常）。"""
        try:
            async def _one():
                async for m in self.drone.telemetry.fixedwing_metrics():
                    return m
            fw = await asyncio.wait_for(_one(), timeout=2.0)
            return getattr(fw, "airspeed_m_s", None) if fw is not None else None
        except (asyncio.TimeoutError, Exception):
            return None

    async def fly_straight_line(self, heading, altitude, speed, duration):
        """直线飞行。heading 度, altitude m, speed m/s, duration s。"""
        print(f"[FlightController] 直线飞行: 航向={heading:.1f}°, 速度={speed:.1f}m/s, 时长={duration:.1f}s")

        vn = speed * np.cos(np.deg2rad(heading))
        ve = speed * np.sin(np.deg2rad(heading))
        vd = 0.0

        try:
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, vd, heading))
            await self.drone.offboard.start()
        except OffboardError as e:
            print(f"[FlightController] 启动 offboard 失败: {e}")
            return

        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < duration:
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, vd, heading))
            await asyncio.sleep(0.1)
        print("[FlightController] 直线飞行完成")

   # ------------------------------------------------------------------ #
    #  NED 位置/速度遥测工具
    # ------------------------------------------------------------------ #

    async def _get_position_velocity_ned(self):
        """获取当前 NED 位置和速度（单次快照）。"""
        async def _one():
            async for pv in self.drone.telemetry.position_velocity_ned():
                return pv
        return await asyncio.wait_for(_one(), timeout=5.0)

    # ------------------------------------------------------------------ #
    #  闭环盘旋（前馈 + 位置反馈）
    # ------------------------------------------------------------------ #

    async def fly_orbit(self, radius, altitude, direction="cw", duration=180, speed=None):
        """
        闭环盘旋飞行：通过 set_position_velocity_acceleration_ned 同时发送
        位置、切线速度和向心加速度给 PX4，触发 NPFG navigatePathTangent
        实现风感知路径跟踪。

        关键：PX4 固定翼 offboard 纯速度模式只控制航向、忽略速度大小，
        必须同时发送位置才能触发 NPFG 风补偿路径跟踪。
        """
        print(f"[FlightController] 盘旋飞行(PVA闭环): 半径={radius:.1f}m, 方向={direction}, 时长={duration:.1f}s")

        orbit_speed = float(speed) if speed is not None else 15.0
        omega = orbit_speed / radius                        # rad/s (正值)

        # ---- 读取初始位置/速度，计算圆心 ----
        pv = await self._get_position_velocity_ned()
        n0, e0 = pv.position.north_m, pv.position.east_m
        d0 = pv.position.down_m                             # 当前高度 (NED down)
        vn0, ve0 = pv.velocity.north_m_s, pv.velocity.east_m_s
        spd0 = np.hypot(vn0, ve0)
        if spd0 < 2.0:                                      # 速度太小，用默认东向
            vn0, ve0, spd0 = 0.0, orbit_speed, orbit_speed

        # NED 俯视图中 "右侧垂直向量" = (-ve, vn) / speed
        # CW: 圆心在速度方向右侧;  CCW: 左侧
        if direction == "cw":
            center_n = n0 + (-ve0) * radius / spd0
            center_e = e0 + vn0 * radius / spd0
            omega_s = omega                                 # 正 → θ 递增 → NED CW
        else:
            center_n = n0 + ve0 * radius / spd0
            center_e = e0 + (-vn0) * radius / spd0
            omega_s = -omega                                # 负 → NED CCW

        theta0 = np.arctan2(e0 - center_e, n0 - center_n)  # 起始极角
        target_down = float(d0)                             # 保持当前高度
        print(f"[FlightController]   圆心 N={center_n:.1f} E={center_e:.1f}, θ₀={np.rad2deg(theta0):.1f}°")

        # ---- 初始 setpoint（offboard.start() 前必须先发一次） ----
        vn_init = float(-radius * omega_s * np.sin(theta0))
        ve_init = float(radius * omega_s * np.cos(theta0))
        yaw_init = float(np.rad2deg(np.arctan2(ve_init, vn_init)) % 360)
        # 向心加速度 (指向圆心) = -ω² × (pos - center)
        an_init = float(-radius * omega_s**2 * np.cos(theta0))
        ae_init = float(-radius * omega_s**2 * np.sin(theta0))
        try:
            await self.drone.offboard.set_position_velocity_acceleration_ned(
                PositionNedYaw(float(n0), float(e0), target_down, yaw_init),
                VelocityNedYaw(vn_init, ve_init, 0.0, yaw_init),
                AccelerationNed(an_init, ae_init, 0.0))
            await self.drone.offboard.start()
        except OffboardError as e:
            if "ALREADY" not in str(e).upper():
                print(f"[FlightController] 启动 offboard 失败: {e}")
                return

        # ---- 后台流式订阅位置遥测 ----
        _latest = [pv]

        async def _pos_stream():
            try:
                async for p in self.drone.telemetry.position_velocity_ned():
                    _latest[0] = p
            except asyncio.CancelledError:
                pass

        pos_task = asyncio.create_task(_pos_stream())

        # ---- 主控制循环 ----
        dt = 0.1
        t = 0.0
        try:
            while t < duration:
                theta = theta0 + omega_s * t

                # 期望位置 (圆轨道点)
                n_des = float(center_n + radius * np.cos(theta))
                e_des = float(center_e + radius * np.sin(theta))

                # 切线速度 (前馈)
                vn_ff = float(-radius * omega_s * np.sin(theta))
                ve_ff = float(radius * omega_s * np.cos(theta))

                # 向心加速度 (指向圆心, a = -ω²·r_vec)
                an = float(-radius * omega_s**2 * np.cos(theta))
                ae = float(-radius * omega_s**2 * np.sin(theta))

                yaw = float(np.rad2deg(np.arctan2(ve_ff, vn_ff)) % 360)

                try:
                    await self.drone.offboard.set_position_velocity_acceleration_ned(
                        PositionNedYaw(n_des, e_des, target_down, yaw),
                        VelocityNedYaw(vn_ff, ve_ff, 0.0, yaw),
                        AccelerationNed(an, ae, 0.0))
                except Exception as exc:
                    # gRPC 连接断开：mavsdk_server 崩溃或 PX4 退出
                    if "Connection refused" in str(exc) or "UNAVAILABLE" in str(exc):
                        print(f"[FlightController] 致命错误: MAVSDK 连接断开 - {exc}")
                        raise RuntimeError("MAVSDK 连接丢失，可能 PX4 已崩溃") from exc
                    # 其他错误：记录但继续（可能是瞬时网络抖动）
                    print(f"[FlightController] 盘旋中设置 PVA 失败: {exc}")

                await asyncio.sleep(dt)
                t += dt
        finally:
            pos_task.cancel()
            try:
                await pos_task
            except asyncio.CancelledError:
                pass

        print("[FlightController] 盘旋飞行完成")

    # ------------------------------------------------------------------ #
    #  8 字机动
    # ------------------------------------------------------------------ #

    async def fly_figure_eight(self, lobe_radius, orientation, altitude=100, duration=300):
        """
        8 字机动：两段闭环盘旋 (CW → CCW)，根据当前速度方向自动衔接。
        orientation: 日志记录用，实际 8 字方向由飞入时速度决定。
        """
        print(f"[FlightController] 8字机动(闭环): 半径={lobe_radius:.1f}m, 方向={orientation:.1f}°, 高度={altitude:.1f}m")
        await self.fly_orbit(lobe_radius, altitude, "cw", duration=duration / 2)
        await self.fly_orbit(lobe_radius, altitude, "ccw", duration=duration / 2)
        print("[FlightController] 8字机动完成")

    async def fly_climb_descent(
        self, h_start, h_end, climb_rate, heading, duration=120,
        altitude_reached_event: asyncio.Event = None,
    ):
        """爬升/下降机动，带高度监控和安全保护。

        - 到达目标高度后自动切为平飞，持续至 duration 结束
        - 设置最低安全高度 MIN_ALT_M，防止飞入地面
        - altitude_reached_event：可选 asyncio.Event；到达目标高度时自动 set()，
          供外部在此之后才启动 DataLogger，避免爬升阶段的劣质数据进入训练集。
        """
        MIN_ALT_M = 40.0
        print(f"[FlightController] 爬升/下降: {h_start:.1f}m → {h_end:.1f}m, 爬升率={climb_rate:.1f}m/s")

        # 安全：如果目标高度低于安全线，钳位到安全线
        if h_end < MIN_ALT_M:
            print(f"[FlightController] 警告: 目标高度 {h_end:.1f}m < 安全线 {MIN_ALT_M}m，钳位")
            h_end = MIN_ALT_M

        horizontal_speed = 15.0
        vn = horizontal_speed * np.cos(np.deg2rad(heading))
        ve = horizontal_speed * np.sin(np.deg2rad(heading))
        vd_climb = -climb_rate   # 爬升/下降阶段的垂直速度
        vd_level = 0.0           # 平飞阶段

        descending = (climb_rate < 0)  # True = 正在下降

        try:
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, vd_climb, heading))
            await self.drone.offboard.start()
        except Exception:
            pass

        reached_target = False
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < duration:
            # 读取当前高度 (NED down → 高度 = -down)
            try:
                pv = await self._get_position_velocity_ned()
                current_alt = (
                    abs(pv.position.down_m)
                    if self._ned_altitude_sign == 0.0
                    else self._ned_altitude_sign * pv.position.down_m
                )
            except Exception:
                current_alt = None

            if current_alt is not None and not reached_target:
                # 检查是否到达目标高度
                if descending and current_alt <= h_end:
                    reached_target = True
                    print(f"[FlightController] 下降目标 {h_end:.1f}m 已达到 ({current_alt:.1f}m)，改平飞")
                elif not descending and current_alt >= h_end:
                    reached_target = True
                    print(f"[FlightController] 爬升目标 {h_end:.1f}m 已达到 ({current_alt:.1f}m)，改平飞")

                # 安全兜底：高度接近安全线时强制平飞
                if current_alt <= MIN_ALT_M + 5 and descending and not reached_target:
                    reached_target = True
                    print(f"[FlightController] 安全保护: 高度 {current_alt:.1f}m 接近安全线，强制平飞")

                # 通知外部等待方（DataLogger）可以开始记录了
                if reached_target and altitude_reached_event is not None:
                    altitude_reached_event.set()
                    altitude_reached_event = None  # 只触发一次

            vd = vd_level if reached_target else vd_climb
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(vn, ve, vd, heading))
            await asyncio.sleep(0.1)

        # 超时兜底：若从未到达目标高度，也释放等待方
        if altitude_reached_event is not None:
            altitude_reached_event.set()
        print("[FlightController] 爬升/下降完成")

    async def land(self):
        """降落，添加重试机制避免 COMMAND_DENIED"""
        print("[FlightController] 开始降落")
        try:
            await self.drone.offboard.stop()
        except Exception:
            pass
        # 增加等待时间确保模式切换完成
        await asyncio.sleep(3.0)
        
        # 添加降落重试逻辑
        for attempt in range(5):
            try:
                await self.drone.action.land()
                print("[FlightController] 降落指令已发送")
                break
            except Exception as e:
                if "COMMAND_DENIED" in str(e) and attempt < 4:
                    print(f"[FlightController] 降落被拒，2s 后重试 ({attempt + 1}/5)")
                    await asyncio.sleep(2)
                else:
                    raise
        await asyncio.sleep(10)
        print("[FlightController] 已降落")
