"""
data_logger.py
记录飞行过程中的传感器数据；按统一时间戳写入逐时刻真值风。

采用统一时钟快照采样（~50 Hz），所有字段严格对齐同一时间戳。
真值风优先从 jsbsim_bridge 的 wind_truth_*.csv 读取（总风 = 背景风 + 阵风 + 高频湍流），
通过 wall_time_usec 与遥测快照精确对齐。若 wind_truth CSV 不可用，则回退到
背景风 + 解析式阵风包络的近似真值。

阵风真值与 jsbsim_bridge 的配置语义保持一致：
- `start_time` 为段内起始时刻（秒）
- `duration` 为整个阵风事件总时长（秒）
- 事件形状为：1-cos 上升 + 平顶保持 + 1-cos 下降
"""

import asyncio
import bisect
import csv as _csv_mod
import json
import math
import time
from pathlib import Path


GUST_STARTUP_RATIO = 0.25
GUST_STEADY_RATIO = 0.50
GUST_END_RATIO = 0.25
GUST_MIN_PHASE_SEC = 0.5

# 转弯状态字符串常量
TURN_STATE_NON_TURNING = "non_turning"
TURN_STATE_TURNING = "turning"
TURN_STATE_UNKNOWN = "unknown"


def _safe_round(value, ndigits=4):
    """NaN 安全的 round：若值为 NaN/Inf 则返回 None。"""
    if value is None or math.isnan(value) or math.isinf(value):
        return None
    return round(value, ndigits)


def _quat_to_euler_rad(q):
    """将 PX4 ATTITUDE_TARGET 的 q=(w,x,y,z) 转为 (roll, pitch, yaw) 弧度。

    采用 ZYX 顺序（与 PX4 / MAVSDK 一致）。返回 yaw ∈ (-π, π]。
    """
    if q is None or len(q) < 4:
        return 0.0, 0.0, 0.0
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    # roll (X)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (Y)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    # yaw (Z)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


# stale 阈值：超过此时长则认为目标量已过期，*_valid=False
_TARGET_STALE_USEC = 200_000  # 200 ms


def _resolve_gust_profile(gust_params):
    """将 gust 参数解析为与 bridge 一致的分段脉冲配置。"""
    if not gust_params:
        return None

    duration = float(gust_params.get("duration", 0.0) or 0.0)
    if duration <= 0:
        return None

    startup = max(GUST_MIN_PHASE_SEC, duration * GUST_STARTUP_RATIO)
    steady = max(GUST_MIN_PHASE_SEC, duration * GUST_STEADY_RATIO)
    end = max(GUST_MIN_PHASE_SEC, duration * GUST_END_RATIO)
    return {
        "start_time": float(gust_params.get("start_time", 0.0) or 0.0),
        "startup": startup,
        "steady": steady,
        "end": end,
        "magnitude": float(gust_params.get("magnitude", 0.0) or 0.0),
        "direction_deg": float(gust_params.get("direction", 0.0) or 0.0),
    }


def _gust_factor_at_t(t_rel, gust_params):
    """返回阵风在当前时刻的幅值比例 [0, 1]。"""
    profile = _resolve_gust_profile(gust_params)
    if profile is None:
        return 0.0

    t0 = profile["start_time"]
    startup = profile["startup"]
    steady = profile["steady"]
    end = profile["end"]

    if t_rel < t0:
        return 0.0

    t1 = t0 + startup
    t2 = t1 + steady
    t3 = t2 + end

    if t_rel < t1:
        tau = (t_rel - t0) / startup if startup > 0 else 1.0
        return 0.5 * (1.0 - math.cos(math.pi * tau))
    if t_rel < t2:
        return 1.0
    if t_rel < t3:
        tau = (t_rel - t2) / end if end > 0 else 1.0
        return 0.5 * (1.0 + math.cos(math.pi * tau))
    return 0.0


def _gust_phase_at_t(t_rel, gust_params):
    """返回阵风阶段，便于后处理识别动态样本。"""
    profile = _resolve_gust_profile(gust_params)
    if profile is None:
        return "none"

    t0 = profile["start_time"]
    startup = profile["startup"]
    steady = profile["steady"]
    end = profile["end"]

    if t_rel < t0:
        return "pre"
    if t_rel < t0 + startup:
        return "rise"
    if t_rel < t0 + startup + steady:
        return "hold"
    if t_rel < t0 + startup + steady + end:
        return "fall"
    return "post"


def _gust_delta_at_t(t_rel, gust_params):
    """返回阵风增量分量与包络系数。"""
    factor = _gust_factor_at_t(t_rel, gust_params)
    if factor <= 0.0:
        return 0.0, 0.0, 0.0, factor

    magnitude = float(gust_params.get("magnitude", 0.0) or 0.0)
    direction_deg = float(gust_params.get("direction", 0.0) or 0.0)
    rad = math.radians(direction_deg)
    dwn = magnitude * factor * math.cos(rad)
    dwe = magnitude * factor * math.sin(rad)
    return dwn, dwe, 0.0, factor


def _gust_wind_at_t(t_rel, wind_north, wind_east, wind_down, gust_params):
    """
    给定段内相对时间 t_rel(s)，返回该时刻真值风 (wind_north, wind_east, wind_down) m/s。
    若无 gust_params 或 t 在阵风外，返回恒定 (wind_north, wind_east, wind_down)。
    此函数给出背景风 + 解析阵风（不含高频湍流），作为 wind_truth 对齐失败时的回退。
    """
    dwn, dwe, dwd, _ = _gust_delta_at_t(t_rel, gust_params)
    if dwn == 0.0 and dwe == 0.0 and dwd == 0.0:
        return wind_north, wind_east, wind_down
    return wind_north + dwn, wind_east + dwe, wind_down


def _load_wind_truth_csv(csv_path):
    """
    读取 jsbsim_bridge 生成的 wind_truth_*.csv，返回按 wall_time_usec 排序的
    (times_list, wn_list, we_list, wd_list)。失败返回 ([], [], [], [])。

    CSV 列名（bridge 写出格式）：
        wall_time_usec, sim_time_s, phase_name,
        wind_north_fps, wind_east_fps, wind_down_fps,
        wind_north_ms, wind_east_ms, wind_down_ms,
        turb_north_fps, turb_east_fps, turb_down_fps,
        total_wind_north_ms, total_wind_east_ms, total_wind_down_ms,
        alt_agl_m, airspeed_kt, groundspeed_kt
    """
    path = Path(csv_path)
    if not path.exists():
        return [], [], [], []
    try:
        times, wn_list, we_list, wd_list = [], [], [], []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = _csv_mod.DictReader(f)
            for row in reader:
                try:
                    times.append(int(row["wall_time_usec"]))
                    wn_list.append(float(row["total_wind_north_ms"]))
                    we_list.append(float(row["total_wind_east_ms"]))
                    wd_list.append(float(row["total_wind_down_ms"]))
                except (KeyError, ValueError):
                    continue
        # 保证时间戳单调（bridge 应该已经是有序的，以防万一）
        if times and times != sorted(times):
            order = sorted(range(len(times)), key=lambda i: times[i])
            times  = [times[i]  for i in order]
            wn_list = [wn_list[i] for i in order]
            we_list = [we_list[i] for i in order]
            wd_list = [wd_list[i] for i in order]
        return times, wn_list, we_list, wd_list
    except Exception as e:
        print(f"[DataLogger] 读取 wind_truth CSV 失败: {e}")
        return [], [], [], []


class DataLogger:
    def __init__(
        self,
        drone,
        wind_north=0.0,
        wind_east=0.0,
        wind_down=0.0,
        gust_params=None,
        wind_truth_csv_path=None,
        turn_state=TURN_STATE_UNKNOWN,
        turn_class=-1,
        pymavlink_url="udpin:0.0.0.0:14550",
        enable_pymavlink_targets=True,
        jsbsim_telnet_port=6789,
    ):
        """
        Args:
            drone: MAVSDK System
            wind_north, wind_east, wind_down: 本轮背景风 (m/s)，用于解析式阵风回退
            gust_params: 若本段为阵风，dict 含 start_time, duration, magnitude, direction；否则 None
            wind_truth_csv_path: jsbsim_bridge 生成的 wind_truth_*.csv 路径；
                若提供且有效，将用总风真值（含湍流）替换解析式风值
            turn_state: 转弯状态字符串 "non_turning" | "turning" | "unknown"
            turn_class: 转弯状态整数标签 0=非转弯 1=转弯 -1=未知
            pymavlink_url: pymavlink 监听地址，用于订阅 PX4 目标量(ATTITUDE_TARGET 等)
            enable_pymavlink_targets: 是否启用 pymavlink 目标量订阅；False 时所有 target_* 字段输出为 0
        """
        self.drone = drone
        self.wind_north = wind_north
        self.wind_east = wind_east
        self.wind_down = wind_down
        self.gust_params = gust_params
        self.wind_truth_csv_path = wind_truth_csv_path
        self.turn_state = turn_state
        self.turn_class = int(turn_class)
        self.maneuver_regime = "unknown"
        self.data_buffer = []
        self.is_logging = False
        self.start_time = None
        self.stats = {
            "skipped_missing_core": 0,
            "skipped_invalid_airspeed": 0,
            "filled_invalid_airspeed": 0,
            "pymavlink_msg_count": 0,
            "pymavlink_attitude_target_count": 0,
            "pymavlink_position_target_count": 0,
            "pymavlink_highres_imu_count": 0,
            "pymavlink_nav_ctrl_count": 0,
        }

        self._attitude = None
        self._gyro = None
        self._fw_metrics = None
        self._actuator_ctrls = None
        self._velocity_ned = None
        self._jsbsim_state = {}

        # 前向填充用最新有效空速（应对 PX4 airspeed_selector 偶发 NaN）
        self._last_valid_airspeed = None

        # ============ pymavlink 目标量缓存（阶段 1+2）============
        # 每个缓存字段都附带 wall_time_usec 时间戳，用于 _snapshot 时计算 age 和 valid
        self.pymavlink_url = pymavlink_url
        self.enable_pymavlink_targets = enable_pymavlink_targets
        self.jsbsim_telnet_port = jsbsim_telnet_port

        # 阶段 1：ATTITUDE_TARGET 提供目标姿态(rad)和目标体轴角速度(rad/s)
        self._target_attitude = None              # (roll_rad, pitch_rad, yaw_rad)
        self._target_attitude_ts = None           # wall_time_usec
        self._target_body_rates = None            # (p, q, r) rad/s
        self._target_body_rates_ts = None

        # 阶段 2：POSITION_TARGET_LOCAL_NED 目标速度（PX4 NPFG fixed-wing 下 vx/vy/vz 永远 NaN，
        #          实际由 NAV_CONTROLLER_OUTPUT 的 nav_bearing/nav_pitch 重建）
        self._target_velocity_ned = None          # (vn, ve, vd) m/s
        self._target_velocity_ned_ts = None

        # 阶段 2：NAV_CONTROLLER_OUTPUT — NPFG 制导律输出
        #   nav_bearing (deg)：NPFG 当前指令航向（课程角，North=0 顺时针）
        #   nav_pitch   (deg)：TECS 当前指令俯仰角
        #   aspd_error  (m/s)：目标空速 - 实际空速
        #   alt_error   (m)  ：目标高度 - 实际高度
        #   xtrack_error(m)  ：横向偏差
        self._nav_ctrl = None       # (nav_bearing_deg, nav_pitch_deg, aspd_error, alt_error, xtrack_error)
        self._nav_ctrl_ts = None

        # 阶段 2：HIGHRES_IMU 提供机体加速度（m/s²）
        self._highres_imu_accel = None            # (ax, ay, az) m/s² body
        self._highres_imu_ts = None

        # 阶段 2：JSBSim Telnet 实际舵面（rad / norm）
        self._actual_actuator = None              # (aileron_rad, elevator_rad, rudder_rad, throttle_norm)
        self._actual_actuator_ts = None

        # pymavlink 后台 connection 句柄（_sub_pymavlink_targets 中赋值）
        self._mav_conn = None

        # PX4 仿真时钟追踪（用于阵风标注去除 speed_factor 偏差）
        # _sim_start_boot_ms: 段开始时 PX4 的 time_boot_ms（毫秒）
        # _sim_time_s: 当前 PX4 仿真时间相对于段起点的偏移（秒）
        self._sim_start_boot_ms = None
        self._sim_time_s = None

    def set_maneuver(self, maneuver_regime):
        """实时设置机动状态标签。"""
        self.maneuver_regime = maneuver_regime

    async def start_logging(self, duration, output_file):
        """记录 duration 秒，以 ~50 Hz 快照采样，保存到 output_file（JSON）。"""
        print(f"[DataLogger] 开始记录，时长={duration:.1f}s -> {output_file}")
        if self.wind_truth_csv_path:
            print(f"[DataLogger] wind_truth 路径: {Path(self.wind_truth_csv_path).name}")
        print(f"[DataLogger] 转弯标签: turn_state={self.turn_state}, turn_class={self.turn_class}")

        self.data_buffer = []
        self.is_logging = True
        self.start_time = time.time()
        self.stats = {
            "skipped_missing_core": 0,
            "skipped_invalid_airspeed": 0,
            "filled_invalid_airspeed": 0,
        }
        self._last_valid_airspeed = None
        self._sim_start_boot_ms = None
        self._sim_time_s = None

        # 请求关键遥测流的发送速率
        for setter, hz, name in [
            (self.drone.telemetry.set_rate_fixedwing_metrics, 50, "fixedwing_metrics"),
            (self.drone.telemetry.set_rate_actuator_control_target, 50, "actuator_control_target"),
            (self.drone.telemetry.set_rate_position_velocity_ned, 50, "position_velocity_ned"),
        ]:
            try:
                await setter(hz)
            except Exception as e:
                print(f"[DataLogger] set_rate_{name} 警告: {e}")

        # 启动后台订阅任务，持续缓存最新值
        bg_tasks = [
            asyncio.create_task(self._sub_attitude()),
            asyncio.create_task(self._sub_imu()),
            asyncio.create_task(self._sub_fixedwing_metrics()),
            asyncio.create_task(self._sub_actuator()),
            asyncio.create_task(self._sub_velocity_ned()),
            asyncio.create_task(self._sub_jsbsim_telnet()),
        ]
        # pymavlink 目标量订阅（PX4 ATTITUDE_TARGET / POSITION_TARGET_LOCAL_NED / HIGHRES_IMU）
        if self.enable_pymavlink_targets:
            bg_tasks.append(asyncio.create_task(self._sub_pymavlink_targets()))

        # Phase 1: 等待核心遥测流（attitude + fw_metrics）出现
        warmup_deadline = time.time() + 8.0
        while (self._attitude is None or self._fw_metrics is None) and time.time() < warmup_deadline:
            await asyncio.sleep(0.05)
        if self._attitude is None or self._fw_metrics is None:
            print(f"[DataLogger] 警告: 核心遥测预热超时 (attitude={'OK' if self._attitude else 'None'}, "
                  f"fw_metrics={'OK' if self._fw_metrics else 'None'})，尝试继续采样")
        else:
            # Phase 2: 如果 airspeed 还是 NaN（airspeed_selector 仍在收敛），多等最多 30 秒
            airspeed_deadline = time.time() + 30.0
            airspeed_val = getattr(self._fw_metrics, "airspeed_m_s", None)
            airspeed_ok = airspeed_val is not None and not math.isnan(airspeed_val) and not math.isinf(airspeed_val)
            if not airspeed_ok:
                print(f"[DataLogger] airspeed_m_s={airspeed_val} 未就绪（airspeed_selector 仍在收敛），等待至多 30s...")
                while time.time() < airspeed_deadline:
                    await asyncio.sleep(0.2)
                    airspeed_val = getattr(self._fw_metrics, "airspeed_m_s", None)
                    if airspeed_val is not None and not math.isnan(airspeed_val) and not math.isinf(airspeed_val):
                        airspeed_ok = True
                        break
            elapsed = time.time() - self.start_time
            if not airspeed_ok:
                # airspeed_selector 卡死：继续采集只会产出 0 条有效记录（所有快照被跳过），
                # 最终触发 "采样结果无效" 并白白消耗整段飞行时间（~120s）。
                # 直接 raise 让外层段重试逻辑立即重启 SITL，节省时间。
                self.is_logging = False
                for _t in bg_tasks:
                    _t.cancel()
                await asyncio.gather(*bg_tasks, return_exceptions=True)
                raise RuntimeError(
                    f"采样结果无效: airspeed_m_s 持续为 NaN（等待 30s 仍未收敛），"
                    f"疑似 airspeed_selector 卡死，需重启 SITL。"
                )
            else:
                print(f"[DataLogger] 遥测预热完成 (耗时={elapsed:.2f}s, airspeed={airspeed_val:.2f}m/s)")

        # 主采样由 _sub_attitude_euler 的网络包到达事件触发
        try:
            # 持续运行指定的 duration（由异步睡眠完成，不再手动阻塞采样）
            await asyncio.sleep(duration)
        finally:
            self.is_logging = False
            for task in bg_tasks:
                task.cancel()
            await asyncio.gather(*bg_tasks, return_exceptions=True)

        # 用 wind_truth CSV 替换解析式真值风（含湍流）
        if self.wind_truth_csv_path:
            self._align_wind_from_truth()

        self._save_data(output_file)
        summary = self.build_summary(duration)
        print(
            f"[DataLogger] 已保存 {output_file}, 共 {summary['sample_count']} 点, "
            f"有效覆盖 {summary['coverage_sec']:.2f}s, "
            f"wind_truth 对齐 {summary.get('wind_truth_aligned_count', 0)} 行"
        )
        return summary

    def _align_wind_from_truth(self):
        """
        从 jsbsim_bridge 的 wind_truth_*.csv 中读取总风真值（含背景风 + 阵风 + 湍流），
        按 wall_time_usec 最近邻对齐到每个采样行，替换 wind_north/east/down 字段。
        对齐成功的行会加 "wind_truth_aligned": True 标记；
        超过 500ms 未找到对应行时保留解析式值。
        """
        times, wn_list, we_list, wd_list = _load_wind_truth_csv(self.wind_truth_csv_path)
        n = len(times)
        if n == 0:
            print(f"[DataLogger] wind_truth CSV 无有效数据，保留解析式真值风")
            return

        aligned_count = 0
        MAX_DT_USEC = 500_000  # 500 ms

        for entry in self.data_buffer:
            wt = entry.get("wall_time_usec")
            if wt is None:
                continue
            # bisect 找最近邻（times 已保证有序）
            pos = bisect.bisect_left(times, wt)
            best_idx = None
            best_dt = MAX_DT_USEC + 1
            for i in (pos - 1, pos):
                if 0 <= i < n:
                    dt = abs(times[i] - wt)
                    if dt < best_dt:
                        best_dt = dt
                        best_idx = i
            if best_idx is None or best_dt > MAX_DT_USEC:
                continue
            entry["wind_north"] = round(wn_list[best_idx], 6)
            entry["wind_east"] = round(we_list[best_idx], 6)
            entry["wind_down"] = round(wd_list[best_idx], 6)
            entry["wind_truth_aligned"] = True
            aligned_count += 1

        self.stats["wind_truth_aligned_count"] = aligned_count
        pct = 100.0 * aligned_count / max(len(self.data_buffer), 1)
        print(f"[DataLogger] wind_truth 对齐: {aligned_count}/{len(self.data_buffer)} 行 ({pct:.1f}%)")

    def build_summary(self, requested_duration):
        """构建采样摘要，供上层写入 metadata 并做质量校验。"""
        sample_count = len(self.data_buffer)
        first_ts = self.data_buffer[0]["timestamp"] if sample_count else None
        last_ts = self.data_buffer[-1]["timestamp"] if sample_count else None
        coverage = max(0.0, (last_ts - first_ts) if sample_count >= 2 else 0.0)
        effective_rate = sample_count / max(requested_duration, 1e-6)
        dynamic_count = sum(1 for row in self.data_buffer if row.get("gust_factor", 0.0) > 1e-3)
        transition_count = sum(
            1 for row in self.data_buffer
            if row.get("gust_phase") in {"rise", "fall"}
        )
        filled_count = self.stats.get("filled_invalid_airspeed", 0)
        filled_ratio = filled_count / max(sample_count, 1)
        aligned_count = self.stats.get("wind_truth_aligned_count", 0)
        return {
            "sample_count": sample_count,
            "requested_duration_sec": round(float(requested_duration), 4),
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
            "coverage_sec": round(coverage, 4),
            "effective_rate_hz": round(effective_rate, 3),
            "dynamic_sample_count": dynamic_count,
            "dynamic_sample_ratio": round(dynamic_count / max(sample_count, 1), 4),
            "transition_sample_count": transition_count,
            "transition_sample_ratio": round(transition_count / max(sample_count, 1), 4),
            "filled_invalid_airspeed_ratio": round(filled_ratio, 4),
            "skipped_missing_core": self.stats["skipped_missing_core"],
            "skipped_invalid_airspeed": self.stats["skipped_invalid_airspeed"],
            "filled_invalid_airspeed": self.stats.get("filled_invalid_airspeed", 0),
            "wind_truth_aligned_count": aligned_count,
            "wind_truth_aligned_ratio": round(aligned_count / max(sample_count, 1), 4),
            "wind_truth_csv": str(self.wind_truth_csv_path) if self.wind_truth_csv_path else None,
            "turn_state": self.turn_state,
            "turn_class": self.turn_class,
            "pymavlink_msg_count": self.stats.get("pymavlink_msg_count", 0),
            "pymavlink_attitude_target_count": self.stats.get("pymavlink_attitude_target_count", 0),
            "pymavlink_position_target_count": self.stats.get("pymavlink_position_target_count", 0),
            "pymavlink_highres_imu_count": self.stats.get("pymavlink_highres_imu_count", 0),
            "pymavlink_nav_ctrl_count": self.stats.get("pymavlink_nav_ctrl_count", 0),
        }

    def _snapshot(self, t):
        """快照所有缓存遥测值到一条数据记录。attitude 和 fw_metrics 是必须字段。"""
        if self._attitude is None or self._fw_metrics is None:
            self.stats["skipped_missing_core"] += 1
            return

        # 空速：优先使用 PX4 airspeed_selector 实时值；若偶发 NaN，使用最近一次有效值
        # 前向填充。绝不使用 groundspeed 作为兜底——这会让段内 airspeed=groundspeed，
        # 训练时模型学不到风对空速的影响。整段都未拿到有效空速 → 段直接失败，依靠段级重试。
        airspeed = _safe_round(self._fw_metrics.airspeed_m_s, 4)
        groundspeed_raw = getattr(self._fw_metrics, "groundspeed_m_s", None)
        if groundspeed_raw is None and self._velocity_ned is not None:
            groundspeed_raw = math.hypot(self._velocity_ned[0], self._velocity_ned[1])
        groundspeed = _safe_round(groundspeed_raw, 4)
        if airspeed is None:
            if self._last_valid_airspeed is not None:
                airspeed = self._last_valid_airspeed
                self.stats["filled_invalid_airspeed"] += 1
            else:
                self.stats["skipped_invalid_airspeed"] += 1
                return
        else:
            self._last_valid_airspeed = airspeed

        # 阵风标注使用 PX4 仿真时间（sim_t），消除 speed_factor 带来的时间偏差。
        # 当仿真时钟尚未就绪（段最初几帧）时回退到挂钟时间 t，结果几乎等价。
        sim_t = self._sim_time_s if self._sim_time_s is not None else t

        # 解析式真值风（背景风 + 阵风，不含湍流）；若 wind_truth CSV 对齐成功会被替换
        wn, we, wd = _gust_wind_at_t(
            sim_t, self.wind_north, self.wind_east, self.wind_down, self.gust_params
        )
        dwn, dwe, dwd, gust_factor = _gust_delta_at_t(sim_t, self.gust_params)
        gust_phase = _gust_phase_at_t(sim_t, self.gust_params)

        # 记录采样时的 wall clock（微秒），用于后续 wind_truth 对齐
        wall_time_usec = int(time.time() * 1e6)

        # 按文档顺序：timestamp → 空速标量 → 风三维 → 地速标量 → 地速三维 → 姿态 → 角速度 → 控制
        entry = {
            "timestamp": round(t, 4),
            "sim_time_s": round(sim_t, 4),
            "wall_time_usec": wall_time_usec,
            "airspeed_m_s": airspeed,
            "wind_north": round(wn, 6),
            "wind_east": round(we, 6),
            "wind_down": round(wd, 6),
            # 诊断字段：背景风、阵风增量、阵风相位（wind_truth 对齐后仍保留供诊断）
            "base_wind_north": round(self.wind_north, 6),
            "base_wind_east": round(self.wind_east, 6),
            "base_wind_down": round(self.wind_down, 6),
            "gust_delta_north": round(dwn, 6),
            "gust_delta_east": round(dwe, 6),
            "gust_delta_down": round(dwd, 6),
            "gust_factor": round(gust_factor, 6),
            "gust_phase": gust_phase,
            "wind_regime": "dynamic_gust" if self.gust_params else "steady",
            "wind_truth_aligned": False,
            "groundspeed_m_s": groundspeed if groundspeed is not None else 0.0,
        }

        if self._velocity_ned is not None:
            entry["velocity_north"] = round(self._velocity_ned[0], 6)
            entry["velocity_east"] = round(self._velocity_ned[1], 6)
            entry["velocity_down"] = round(self._velocity_ned[2], 6)

        entry["roll_deg"] = round(self._attitude.roll_deg, 4)
        entry["pitch_deg"] = round(self._attitude.pitch_deg, 4)
        entry["yaw_deg"] = round(self._attitude.yaw_deg, 4)

        if self._gyro is not None:
            entry["roll_rate_rad_s"] = round(self._gyro[0], 6)
            entry["pitch_rate_rad_s"] = round(self._gyro[1], 6)
            entry["yaw_rate_rad_s"] = round(self._gyro[2], 6)

        if self._actuator_ctrls is not None:
            entry["roll_ctrl"] = round(self._actuator_ctrls[0], 6)
            entry["pitch_ctrl"] = round(self._actuator_ctrls[1], 6)
            entry["yaw_ctrl"] = round(self._actuator_ctrls[2], 6)
            entry["throttle_ctrl"] = round(self._actuator_ctrls[3], 6)

        # ============ 阶段 1：PX4 目标量（pymavlink 来源）============
        # 目标姿态（rad → deg 与 roll_deg/pitch_deg/yaw_deg 单位一致）
        ta = self._target_attitude
        ta_age = (wall_time_usec - self._target_attitude_ts) if self._target_attitude_ts else None
        ta_valid = ta is not None and ta_age is not None and ta_age <= _TARGET_STALE_USEC
        if ta is not None:
            entry["target_roll_deg"] = round(math.degrees(ta[0]), 4)
            entry["target_pitch_deg"] = round(math.degrees(ta[1]), 4)
            entry["target_yaw_deg"] = round(math.degrees(ta[2]), 4)
        else:
            entry["target_roll_deg"] = 0.0
            entry["target_pitch_deg"] = 0.0
            entry["target_yaw_deg"] = 0.0
        entry["target_attitude_valid"] = bool(ta_valid)
        entry["target_attitude_age_ms"] = round(ta_age / 1000.0, 2) if ta_age is not None else -1.0

        # 目标体轴角速度（rad/s）
        tr = self._target_body_rates
        tr_age = (wall_time_usec - self._target_body_rates_ts) if self._target_body_rates_ts else None
        tr_valid = tr is not None and tr_age is not None and tr_age <= _TARGET_STALE_USEC
        if tr is not None:
            entry["target_roll_rate_rad_s"] = round(tr[0], 6)
            entry["target_pitch_rate_rad_s"] = round(tr[1], 6)
            entry["target_yaw_rate_rad_s"] = round(tr[2], 6)
        else:
            entry["target_roll_rate_rad_s"] = 0.0
            entry["target_pitch_rate_rad_s"] = 0.0
            entry["target_yaw_rate_rad_s"] = 0.0
        entry["target_body_rates_valid"] = bool(tr_valid)
        entry["target_body_rates_age_ms"] = round(tr_age / 1000.0, 2) if tr_age is not None else -1.0

        # ============ 阶段 2：目标速度 / IMU 加速度 / 实际舵面 ============

        # --- 目标速度（由 NAV_CONTROLLER_OUTPUT 重建） ---
        # PX4 NPFG fixed-wing 控制律不走速度设点层，POSITION_TARGET_LOCAL_NED 的
        # vx/vy/vz 永远是 NaN。真正的"目标速度"来源是 NAV_CONTROLLER_OUTPUT：
        #   nav_bearing (deg) = NPFG 当前指令航向（相当于目标地速方向）
        #   nav_pitch   (deg) = TECS 当前指令俯仰（相当于目标爬升角）
        # 重建公式（协调飞行假设）：
        #   target_vn = airspeed × cos(nav_pitch_rad) × cos(nav_bearing_rad)
        #   target_ve = airspeed × cos(nav_pitch_rad) × sin(nav_bearing_rad)
        #   target_vd = -airspeed × sin(nav_pitch_rad)
        nc = self._nav_ctrl
        nc_age = (wall_time_usec - self._nav_ctrl_ts) if self._nav_ctrl_ts else None
        nc_valid = nc is not None and nc_age is not None and nc_age <= _TARGET_STALE_USEC

        # 同时保留旧 POSITION_TARGET 路径（若未来 PX4 版本填上速度设点）
        tv = self._target_velocity_ned
        tv_age = (wall_time_usec - self._target_velocity_ned_ts) if self._target_velocity_ned_ts else None
        tv_finite = (
            tv is not None
            and not any(math.isnan(v) or math.isinf(v) for v in tv)
        )
        tv_valid = tv_finite and tv_age is not None and tv_age <= _TARGET_STALE_USEC

        if nc_valid:
            # 优先用 NAV_CONTROLLER_OUTPUT 重建（最准确的 PX4 fixed-wing 目标速度）
            nb_rad = math.radians(nc[0])   # nav_bearing
            np_rad = math.radians(nc[1])   # nav_pitch
            cos_np = math.cos(np_rad)
            target_vn = airspeed * cos_np * math.cos(nb_rad)
            target_ve = airspeed * cos_np * math.sin(nb_rad)
            target_vd = -airspeed * math.sin(np_rad)
            entry["target_velocity_north"] = round(target_vn, 6)
            entry["target_velocity_east"]  = round(target_ve, 6)
            entry["target_velocity_down"]  = round(target_vd, 6)
            entry["target_velocity_source"] = "nav_controller"
        elif tv_valid:
            # 备用：POSITION_TARGET 速度设点（仅当 NPFG 未填 NaN 时）
            entry["target_velocity_north"] = round(tv[0], 6)
            entry["target_velocity_east"]  = round(tv[1], 6)
            entry["target_velocity_down"]  = round(tv[2], 6)
            entry["target_velocity_source"] = "position_target"
        else:
            entry["target_velocity_north"] = 0.0
            entry["target_velocity_east"]  = 0.0
            entry["target_velocity_down"]  = 0.0
            entry["target_velocity_source"] = "none"
        entry["target_velocity_valid"] = bool(nc_valid or tv_valid)
        entry["target_velocity_age_ms"] = round(
            nc_age / 1000.0 if nc_age is not None else (tv_age / 1000.0 if tv_age is not None else -1.0),
            2,
        )

        # --- 辅助字段：NAV_CONTROLLER_OUTPUT 原始值 ---
        if nc is not None:
            entry["nav_bearing_deg"]  = round(nc[0], 4)
            entry["nav_pitch_deg"]    = round(nc[1], 4)
            entry["aspd_error_m_s"]   = round(nc[2], 4)
            entry["alt_error_m"]      = round(nc[3], 4)
            entry["xtrack_error_m"]   = round(nc[4], 4)
        else:
            entry["nav_bearing_deg"]  = 0.0
            entry["nav_pitch_deg"]    = 0.0
            entry["aspd_error_m_s"]   = 0.0
            entry["alt_error_m"]      = 0.0
            entry["xtrack_error_m"]   = 0.0

        # --- 空速向量 NED（由实际姿态 + airspeed 计算，可直接推风矢量）---
        # wind = velocity_ned - airspeed_vector_ned（协调飞行近似）
        pitch_rad = math.radians(self._attitude.pitch_deg)
        yaw_rad   = math.radians(self._attitude.yaw_deg)
        cos_pitch = math.cos(pitch_rad)
        entry["airspeed_vector_north"] = round(airspeed * cos_pitch * math.cos(yaw_rad), 6)
        entry["airspeed_vector_east"]  = round(airspeed * cos_pitch * math.sin(yaw_rad), 6)
        entry["airspeed_vector_down"]  = round(-airspeed * math.sin(pitch_rad), 6)

        ia = self._highres_imu_accel
        ia_age = (wall_time_usec - self._highres_imu_ts) if self._highres_imu_ts else None
        ia_valid = ia is not None and ia_age is not None and ia_age <= _TARGET_STALE_USEC
        if ia is not None:
            entry["imu_accel_body_x"] = round(ia[0], 6)
            entry["imu_accel_body_y"] = round(ia[1], 6)
            entry["imu_accel_body_z"] = round(ia[2], 6)
        else:
            entry["imu_accel_body_x"] = 0.0
            entry["imu_accel_body_y"] = 0.0
            entry["imu_accel_body_z"] = 0.0
        entry["imu_accel_valid"] = bool(ia_valid)
        entry["imu_accel_age_ms"] = round(ia_age / 1000.0, 2) if ia_age is not None else -1.0

        aa = self._actual_actuator
        aa_age = (wall_time_usec - self._actual_actuator_ts) if self._actual_actuator_ts else None
        aa_valid = aa is not None and aa_age is not None and aa_age <= _TARGET_STALE_USEC
        if aa is not None:
            entry["aileron_actual_rad"] = round(aa[0], 6)
            entry["elevator_actual_rad"] = round(aa[1], 6)
            entry["rudder_actual_rad"] = round(aa[2], 6)
            entry["throttle_actual_norm"] = round(aa[3], 6)
            entry["actuator_actual_source"] = "jsbsim_telnet"
        else:
            # 回退：JSBSim Telnet 不可用时，用 mavsdk actuator_control_target
            # 命令值作为 actual 的代理。fixed-wing 闭环下命令-实际滞后 <50ms。
            # 需要在 Rascal110-JSBSim.xml 加 socket output 才能拿真实 fcs/* 值。
            ac = self._actuator_ctrls
            if ac is not None and len(ac) >= 4:
                # PX4 actuator_control_target.controls (group 0): [roll, pitch, yaw, throttle]，
                # roll/pitch/yaw ∈ [-1, 1]，throttle ∈ [0, 1]。
                # 用 Rascal110-JSBSim.xml 里的 max deflection 缩放：
                #   aileron/rudder ±0.35 rad，elevator ±0.30 rad。
                roll_c = float(ac[0])
                pitch_c = float(ac[1])
                yaw_c = float(ac[2])
                thr_c = float(ac[3])
                entry["aileron_actual_rad"] = round(roll_c * 0.35, 6)
                entry["elevator_actual_rad"] = round(pitch_c * 0.30, 6)
                entry["rudder_actual_rad"] = round(yaw_c * 0.35, 6)
                entry["throttle_actual_norm"] = round(max(0.0, min(1.0, thr_c)), 6)
                entry["actuator_actual_source"] = "px4_command_proxy"
            else:
                entry["aileron_actual_rad"] = 0.0
                entry["elevator_actual_rad"] = 0.0
                entry["rudder_actual_rad"] = 0.0
                entry["throttle_actual_norm"] = 0.0
                entry["actuator_actual_source"] = "none"
        entry["actuator_actual_valid"] = bool(aa_valid)
        entry["actuator_actual_age_ms"] = round(aa_age / 1000.0, 2) if aa_age is not None else -1.0

        entry["maneuver_regime"] = self.maneuver_regime
        entry["turn_state"] = self.turn_state
        entry["turn_class"] = self.turn_class
        self.data_buffer.append(entry)

    # ---------- 后台遥测订阅 ----------

    async def _sub_attitude(self):
        """订阅姿态（频率较高），作为快照的主时钟触发器"""
        try:
            async for att in self.drone.telemetry.attitude_euler():
                if not self.is_logging:
                    break
                self._attitude = att
                # 由 MAVLink 事件驱动的真值记录，严格闭环
                if self.start_time is not None:
                    t = time.time() - self.start_time
                    self._snapshot(t)
        except asyncio.CancelledError:
            pass

    async def _sub_imu(self):
        try:
            async for imu in self.drone.telemetry.imu():
                if not self.is_logging:
                    break
                try:
                    gyro = imu.angular_velocity_frd
                    self._gyro = (gyro.forward_rad_s, gyro.right_rad_s, gyro.down_rad_s)
                except AttributeError:
                    # 兼容旧版 MAVSDK
                    self._gyro = (
                        imu.angular_velocity_forward_rad_s,
                        imu.angular_velocity_right_rad_s,
                        imu.angular_velocity_down_rad_s,
                    )
        except asyncio.CancelledError:
            pass

    async def _sub_fixedwing_metrics(self):
        try:
            async for metrics in self.drone.telemetry.fixedwing_metrics():
                if not self.is_logging:
                    break
                self._fw_metrics = metrics
        except asyncio.CancelledError:
            pass

    async def _sub_actuator(self):
        try:
            async for act in self.drone.telemetry.actuator_control_target():
                if not self.is_logging:
                    break
                # group 0 = 核心飞行控制组: [roll, pitch, yaw, throttle, ...]
                if act.group == 0 and len(act.controls) >= 4:
                    self._actuator_ctrls = (
                        act.controls[0],   # roll  [-1, 1]
                        act.controls[1],   # pitch [-1, 1]
                        act.controls[2],   # yaw   [-1, 1]
                        act.controls[3],   # throttle [0, 1]
                    )
        except asyncio.CancelledError:
            pass

    async def _sub_velocity_ned(self):
        """订阅 NED 地速分量（north, east, down）"""
        try:
            async for pv in self.drone.telemetry.position_velocity_ned():
                if not self.is_logging:
                    break
                vel = pv.velocity
                self._velocity_ned = (vel.north_m_s, vel.east_m_s, vel.down_m_s)
        except asyncio.CancelledError:
            pass

    async def _sub_jsbsim_telnet(self):
        """订阅 JSBSim Telnet 输出（可选诊断通道 + 阶段 2 实际舵面）。

        阶段 2 新增读取：fcs/aileron-pos-rad、fcs/elevator-pos-rad、fcs/rudder-pos-rad、
        fcs/throttle-pos-norm，写入 self._actual_actuator + self._actual_actuator_ts。

        注意：jsbsim_bridge 的端口 4560 是 mavlink TCP（与 PX4 通信），**不是** JSBSim
        Telnet。要拿到 fcs/* 实际舵面位置，需要在 Rascal110-JSBSim.xml 里加：

            <output type="SOCKET" port="6789" rate="100">
              <property> fcs/aileron-pos-rad </property>
              <property> fcs/elevator-pos-rad </property>
              <property> fcs/rudder-pos-rad </property>
              <property> fcs/throttle-pos-norm </property>
            </output>

        若该 socket 不可达，本任务静默退出，data_logger 自动回退到 mavsdk
        actuator_control_target（命令值，与实际值差 <50ms 滞后）。
        """
        host = "127.0.0.1"
        port = getattr(self, "jsbsim_telnet_port", 6789)
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=1.5,
            )
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
            print(f"[DataLogger] JSBSim Telnet ({host}:{port}) 不可达，"
                  f"actuator_actual 将回退到 actuator_control_target")
            return
        try:
            try:
                await asyncio.wait_for(
                    reader.readuntil(b"JSBSim> "), timeout=1.5,
                )
            except (asyncio.TimeoutError, asyncio.IncompleteReadError):
                # 端口连上了但不是 JSBSim Telnet（可能是 mavlink TCP 等其他服务）
                print(f"[DataLogger] {host}:{port} 不像 JSBSim Telnet，actuator_actual "
                      f"将回退到 actuator_control_target")
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
                return
            props = [
                "simulation/sim-time-sec",
                "atmosphere/wind-north-fps",
                "atmosphere/wind-east-fps",
                "atmosphere/wind-down-fps",
                "atmosphere/turb-north-fps",
                "atmosphere/turb-east-fps",
                "atmosphere/turb-down-fps",
                # 阶段 2：实际舵面位置（rad / norm），用于 L_dyn 残差损失
                "fcs/aileron-pos-rad",
                "fcs/elevator-pos-rad",
                "fcs/rudder-pos-rad",
                "fcs/throttle-pos-norm",
            ]
            cmd = "\n".join([f"get {p}" for p in props]) + "\n"
            cmd_bytes = cmd.encode("utf-8")

            while self.is_logging:
                writer.write(cmd_bytes)
                await writer.drain()
                new_state = {}
                for p in props:
                    try:
                        resp = await reader.readuntil(b"JSBSim> ")
                        text = resp.decode("utf-8")
                        if "=" in text:
                            val = float(text.split("=")[1].split()[0])
                            new_state[p] = val
                    except Exception:
                        pass

                self._jsbsim_state.update(new_state)

                # 写入实际舵面缓存（若 4 个属性都成功读到才更新，避免半条记录）
                a_act = new_state.get("fcs/aileron-pos-rad")
                e_act = new_state.get("fcs/elevator-pos-rad")
                r_act = new_state.get("fcs/rudder-pos-rad")
                t_act = new_state.get("fcs/throttle-pos-norm")
                if all(v is not None for v in (a_act, e_act, r_act, t_act)):
                    self._actual_actuator = (
                        float(a_act), float(e_act), float(r_act), float(t_act),
                    )
                    self._actual_actuator_ts = int(time.time() * 1e6)

                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
        except Exception as e:
            print(f"[DataLogger] JSBSim Telnet 背景任务异常: {e}")

    async def _sub_pymavlink_targets(self):
        """通过 pymavlink 订阅 PX4 原始 MAVLink 消息，获取目标姿态/目标速度/原始 IMU。

        监听 self.pymavlink_url（默认 udpin:0.0.0.0:14550，与 mavsdk_server 占用的 14540 不冲突）。
        采用同步 recv_match + 短超时 + asyncio.sleep(0) 让步，避免阻塞事件循环。

        关注消息：
          ATTITUDE_TARGET           → 目标姿态(quaternion)、目标体轴角速度
          POSITION_TARGET_LOCAL_NED → 目标速度 NED
          HIGHRES_IMU               → 机体加速度 xacc/yacc/zacc (m/s²)
        """
        try:
            from pymavlink import mavutil
        except ImportError:
            print("[DataLogger] pymavlink 未安装，跳过 _sub_pymavlink_targets")
            return

        try:
            self._mav_conn = mavutil.mavlink_connection(self.pymavlink_url)
            print(f"[DataLogger] pymavlink 监听 {self.pymavlink_url}")
        except Exception as e:
            print(f"[DataLogger] pymavlink 连接失败({self.pymavlink_url}): {e}")
            return

        # 主动请求 HIGHRES_IMU 流（PX4 默认不发，需要 set_message_interval）。
        # 没有 IMU 加速度，PI-GRU 失去关键输入特征。这里以 50Hz (20000us) 请求。
        # 即使该消息已在发，重复请求也是幂等的。
        try:
            self._mav_conn.wait_heartbeat(timeout=3)
            target_sys = self._mav_conn.target_system or 1
            target_comp = self._mav_conn.target_component or 1
            HIGHRES_IMU_MSG_ID = 105
            POSITION_TARGET_MSG_ID = 85
            ATTITUDE_TARGET_MSG_ID = 83
            NAV_CONTROLLER_OUTPUT_MSG_ID = 62   # nav_bearing/nav_pitch → target velocity NED
            for msg_id, hz in [
                (HIGHRES_IMU_MSG_ID, 50),
                (POSITION_TARGET_MSG_ID, 50),
                (ATTITUDE_TARGET_MSG_ID, 50),
                (NAV_CONTROLLER_OUTPUT_MSG_ID, 50),
            ]:
                interval_us = int(1_000_000 / hz)
                self._mav_conn.mav.command_long_send(
                    target_sys, target_comp,
                    511,  # MAV_CMD_SET_MESSAGE_INTERVAL
                    0,    # confirmation
                    msg_id, interval_us,
                    0, 0, 0, 0, 0,
                )
            print(f"[DataLogger] 已请求 HIGHRES_IMU/POSITION_TARGET/ATTITUDE_TARGET/NAV_CONTROLLER @ 50Hz")
        except Exception as e:
            print(f"[DataLogger] 请求 HIGHRES_IMU 流失败（不致命）: {e}")

        # 确保 stats 中存在 pymavlink 相关计数器（防御旧缓存或外部直接实例化）
        for _k in ("pymavlink_msg_count", "pymavlink_attitude_target_count",
                   "pymavlink_position_target_count", "pymavlink_highres_imu_count",
                   "pymavlink_nav_ctrl_count"):
            self.stats.setdefault(_k, 0)

        try:
            while self.is_logging:
                try:
                    msg = self._mav_conn.recv_match(blocking=False)
                except Exception:
                    msg = None

                if msg is None:
                    # 没有新消息时让出控制权，保持 ~200 Hz 轮询频率
                    await asyncio.sleep(0.005)
                    continue

                self.stats["pymavlink_msg_count"] += 1
                mtype = msg.get_type()
                now_usec = int(time.time() * 1e6)

                # 追踪 PX4 仿真时钟：time_boot_ms (ms) / time_usec (μs) 均为仿真时间
                # 在任何带时间戳的消息上更新，以保证 _sim_time_s 尽可能新鲜
                try:
                    _boot_ms = None
                    _t_boot_raw = getattr(msg, "time_boot_ms", None)
                    _t_usec_raw = getattr(msg, "time_usec", None)
                    if _t_boot_raw is not None:
                        _boot_ms = float(_t_boot_raw)
                    elif _t_usec_raw is not None:
                        _boot_ms = float(_t_usec_raw) / 1000.0
                    if _boot_ms is not None and _boot_ms > 0:
                        if self._sim_start_boot_ms is None and self.is_logging:
                            self._sim_start_boot_ms = _boot_ms
                        if self._sim_start_boot_ms is not None:
                            self._sim_time_s = max(0.0, (_boot_ms - self._sim_start_boot_ms) / 1000.0)
                except Exception:
                    pass

                if mtype == "ATTITUDE_TARGET":
                    # body_roll_rate / body_pitch_rate / body_yaw_rate (rad/s)
                    # q = (w, x, y, z) — 转换为 roll/pitch/yaw (rad)
                    try:
                        q = msg.q  # tuple of 4 floats: w, x, y, z
                        roll, pitch, yaw = _quat_to_euler_rad(q)
                        self._target_attitude = (roll, pitch, yaw)
                        self._target_attitude_ts = now_usec
                        self._target_body_rates = (
                            float(msg.body_roll_rate),
                            float(msg.body_pitch_rate),
                            float(msg.body_yaw_rate),
                        )
                        self._target_body_rates_ts = now_usec
                        self.stats["pymavlink_attitude_target_count"] += 1
                    except Exception:
                        pass

                elif mtype == "POSITION_TARGET_LOCAL_NED":
                    # vx/vy/vz: 目标速度 NED (m/s)
                    # 注意: PX4 NPFG fixed-wing 控制下 type_mask 把 velocity 位
                    # 标记为 ignore，对应字段填 NaN（MAVLink 协议）。直接 float()
                    # 会得到 NaN 写入数据集，污染 PI-GRU 训练输入。
                    # 这里检查任一分量为 NaN 即跳过本次更新（等下一帧），保留旧值。
                    try:
                        vx_, vy_, vz_ = float(msg.vx), float(msg.vy), float(msg.vz)
                        if not (math.isnan(vx_) or math.isnan(vy_) or math.isnan(vz_)
                                or math.isinf(vx_) or math.isinf(vy_) or math.isinf(vz_)):
                            self._target_velocity_ned = (vx_, vy_, vz_)
                            self._target_velocity_ned_ts = now_usec
                            self.stats["pymavlink_position_target_count"] += 1
                        else:
                            self.stats.setdefault("pymavlink_position_target_ignored_nan", 0)
                            self.stats["pymavlink_position_target_ignored_nan"] += 1
                    except Exception:
                        pass

                elif mtype == "NAV_CONTROLLER_OUTPUT":
                    # PX4 NPFG 制导律输出：nav_bearing (deg), nav_pitch (deg),
                    # aspd_error (m/s), alt_error (m), xtrack_error (m)。
                    # nav_bearing = NPFG 当前指令航向（度，North=0 顺时针）。
                    # 结合 airspeed 可精确重建 target_velocity NED：
                    #   vn = airspeed × cos(nav_pitch) × cos(nav_bearing)
                    #   ve = airspeed × cos(nav_pitch) × sin(nav_bearing)
                    #   vd = -airspeed × sin(nav_pitch)
                    try:
                        nb = float(msg.nav_bearing)
                        np_ = float(msg.nav_pitch)
                        ae = float(msg.aspd_error)
                        alt_e = float(msg.alt_error)
                        xt = float(msg.xtrack_error)
                        if not any(math.isnan(v) or math.isinf(v) for v in (nb, np_)):
                            self._nav_ctrl = (nb, np_, ae, alt_e, xt)
                            self._nav_ctrl_ts = now_usec
                            self.stats.setdefault("pymavlink_nav_ctrl_count", 0)
                            self.stats["pymavlink_nav_ctrl_count"] += 1
                    except Exception:
                        pass

                elif mtype == "HIGHRES_IMU":
                    # xacc/yacc/zacc 机体系加速度 (m/s²)
                    try:
                        ax_, ay_, az_ = float(msg.xacc), float(msg.yacc), float(msg.zacc)
                        if not (math.isnan(ax_) or math.isnan(ay_) or math.isnan(az_)
                                or math.isinf(ax_) or math.isinf(ay_) or math.isinf(az_)):
                            self._highres_imu_accel = (ax_, ay_, az_)
                            self._highres_imu_ts = now_usec
                        self.stats["pymavlink_highres_imu_count"] += 1
                    except Exception:
                        pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[DataLogger] _sub_pymavlink_targets 异常: {e}")
        finally:
            try:
                if self._mav_conn is not None:
                    self._mav_conn.close()
            except Exception:
                pass

    def _save_data(self, output_file):
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.data_buffer, f, indent=2)
