"""
7_sitl_closedloop_eval.py  ─  SITL 闭环控制效益评估脚本 v1.0
=====================================================================
用途
----
在 PX4 SITL + JSBSim 环境中，以「HITL 完全相同的五阶段风场脚本 + 定点
任务」对比两组实验，为论文 §3.5.2 提供闭环控制效益证据。

实验模式 (--mode)
-----------------
  baseline   : 不进行任何风场估计推理；PX4 使用内置 EKF2 风估计；
               仅收集控制性能指标作为基准。
  pirnn_akf  : 运行 PIRNN-AKF 完整推理链（PI-GRU + AKF）；
               以 WIND_COV MAVLink 消息将估计风速发回 PX4；
               同时记录估计精度与控制指标。

核心指标（CSV 输出列）
----------------------
  xtrack_error    : 横向航迹误差 XTE [m]        (NAV_CONTROLLER_OUTPUT)
  airspeed        : 实际空速 [m/s]               (VFR_HUD)
  airspeed_err    : 空速跟踪误差 [m/s]           (airspeed - target_airspeed)
  elevator_raw    : 升降舵 PWM 输出              (SERVO_OUTPUT_RAW.servo2_raw)
  throttle_raw    : 油门 PWM 输出               (SERVO_OUTPUT_RAW.servo3_raw)
  wind_gt_{n,e,d} : JSBSim 真值风 [m/s]         (WIND MAVLink msg)
  wind_ekf2_{n,e} : PX4 EKF2 内部风估计 [m/s]   (WIND_COV msg)
  wind_est_{n,e,d}: PIRNN-AKF 估计风速 [m/s]    (pirnn_akf 模式，否则 NaN)
  phase           : 当前扰动阶段标签

五阶段时序（与 HITL_JSBSim_WSL2_操作流程.md 完全一致）
------------------------------------------------------
  warmup       0 - 120 s    (2 min)  无风预热，等待飞机起飞稳定
  steady     120 - 420 s    (5 min)  稳态风 N=5 m/s, E=3 m/s
  gust_light 420 - 720 s    (5 min)  轻阵风 5 m/s
  gust_strong 720 - 1020 s  (5 min)  强阵风 6 m/s
  packet_loss 1020 s →      (持续)   稳态风，外部模拟通信干扰

典型运行命令
------------
  # 先跑基准（纯 EKF2）：
  python3 7_sitl_closedloop_eval.py --mode baseline \\
      --connection udpin:0.0.0.0:14550 \\
      --output ../SITL/sitl_baseline_$(date +%Y%m%d_%H%M%S).csv

  # 再跑 PIRNN-AKF：
  python3 7_sitl_closedloop_eval.py --mode pirnn_akf \\
      --connection udpin:0.0.0.0:14550 \\
      --output ../SITL/sitl_pirnn_akf_$(date +%Y%m%d_%H%M%S).csv

  # 完成后用分析脚本生成对比图：
  python3 ../scripts/analyze_sitl_closedloop.py \\
      --baseline ../SITL/sitl_baseline_*.csv \\
      --pirnn    ../SITL/sitl_pirnn_akf_*.csv \\
      --output   ../SITL/comparison.png
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from pymavlink import mavutil


# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

# ─────────────────────────────────────────────────────────────────────────────
# 五阶段时序（秒），与 HITL 风场配置保持一致
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PHASE_SCHEDULE = [
    ("warmup",      0,    120),
    ("steady",      120,  420),
    ("gust_light",  420,  720),
    ("gust_strong", 720,  1020),
    ("packet_loss", 1020, None),   # None 表示持续到实验结束
]


def _load_akf_class():
    """动态加载 5_pigru_akf_fusion.py 中的 AdaptiveKalmanFilter。"""
    akf_path = _SCRIPT_DIR / "5_pigru_akf_fusion.py"
    spec = importlib.util.spec_from_file_location("pigru_akf_fusion", akf_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {akf_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AdaptiveKalmanFilter


def _get_phase(runtime_s: float, schedule=DEFAULT_PHASE_SCHEDULE) -> str:
    """根据运行时间返回当前阶段名称。"""
    for name, start, end in schedule:
        if end is None or runtime_s < end:
            if runtime_s >= start:
                return name
    return schedule[-1][0]


# ─────────────────────────────────────────────────────────────────────────────
# 主评估类
# ─────────────────────────────────────────────────────────────────────────────

class SITLClosedLoopEvaluator:
    """
    SITL 闭环控制效益评估器。

    baseline 模式：仅收集 MAVLink 数据，记录 EKF2 风估计和控制指标。
    pirnn_akf 模式：运行 PIRNN-AKF 推理，发送 WIND_COV，记录所有指标。
    """

    # -------------------------------------------------------------------
    # 初始化
    # -------------------------------------------------------------------
    def __init__(
        self,
        mode: str,
        connection_str: str,
        output_csv: str,
        config_path: Optional[str] = None,
        target_airspeed: float = 12.0,
        eval_duration_s: Optional[float] = None,
        send_wind_cov: bool = True,
    ):
        assert mode in ("baseline", "pirnn_akf"), \
            f"mode 必须是 'baseline' 或 'pirnn_akf'，实际: {mode}"

        self.mode = mode
        self.connection_str = connection_str
        self.output_csv = output_csv
        self.target_airspeed = target_airspeed
        self.eval_duration_s = eval_duration_s   # None = 不限时长（手动 Ctrl+C）
        self.send_wind_cov_flag = send_wind_cov and (mode == "pirnn_akf")

        # ── 配置 ──
        cfg_path = Path(config_path) if config_path else _CONFIG_PATH
        with open(cfg_path, "r") as f:
            self.config = yaml.safe_load(f)

        deploy_cfg = self.config.get("deployment", {})
        self.inference_rate: float = float(deploy_cfg.get("inference_rate", 50.0))
        self.sequence_length: int = int(self.config["data"]["sequence_length"])
        self.ema_alpha_wind: float = float(deploy_cfg.get("ema_alpha_wind", 0.2))
        self.ema_alpha_params: float = float(deploy_cfg.get("ema_alpha_params", 0.1))
        self.max_wind_speed: float = float(deploy_cfg.get("max_wind_speed", 20.0))

        # ── 日志 ──
        self._setup_logging()

        # ── 运行时状态 ──
        self.connection: Optional[mavutil.mavudp] = None
        self.data_buffer: deque = deque(maxlen=self.sequence_length)

        self.prev_log_q = None
        self.prev_log_r = None
        self.prev_wind_nn: Optional[np.ndarray] = None
        self.ema_wind: Optional[np.ndarray] = None   # 手动 EMA（不依赖 6c 类）

        # AKF
        self.akf = None
        self.akf_initialized = False
        self.akf_warmup_count = 0
        self.akf_warmup_threshold = 10

        # 性能统计
        self._start_wall: float = 0.0
        self._infer_count: int = 0
        self._infer_total_ms: float = 0.0
        self._infer_max_ms: float = 0.0

        # ── 加载推理资源（仅 pirnn_akf 模式）──
        self.backend = None
        self.scaler_X = None
        self.y_mean: Optional[np.ndarray] = None
        self.y_std: Optional[np.ndarray] = None

        if self.mode == "pirnn_akf":
            self._load_backend()
            self._load_norm_params()
            AdaptiveKalmanFilter = _load_akf_class()
            dt = 1.0 / self.inference_rate
            self.akf = AdaptiveKalmanFilter(dt=dt)
            self.logger.info("PIRNN-AKF 推理资源加载完成")

        # ── CSV ──
        self._csv_file = None
        self._csv_writer = None
        self._init_csv()

        self.logger.info("=" * 65)
        self.logger.info(f"模式: {self.mode}")
        self.logger.info(f"连接: {self.connection_str}")
        self.logger.info(f"目标空速: {self.target_airspeed} m/s")
        self.logger.info(f"推理频率: {self.inference_rate} Hz")
        self.logger.info(f"序列长度: {self.sequence_length}")
        self.logger.info(f"输出 CSV: {self.output_csv}")
        if self.eval_duration_s:
            self.logger.info(f"实验时长上限: {self.eval_duration_s:.0f} s")
        self.logger.info("=" * 65)

    # -------------------------------------------------------------------
    # 日志
    # -------------------------------------------------------------------
    def _setup_logging(self):
        log_dir = _PROJECT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"sitl_closedloop_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger("sitl_eval")
        self.logger.info(f"日志文件: {log_file}")

    # -------------------------------------------------------------------
    # 推理资源（仅 pirnn_akf 模式）
    # -------------------------------------------------------------------
    def _load_backend(self):
        sys.path.insert(0, str(_SCRIPT_DIR))
        from inference_backends import create_backend  # type: ignore
        self.backend = create_backend(self.config)
        info = self.backend.load()
        self.logger.info(f"模型加载完成: {info.get('model_path', '?')}")
        self.logger.info(
            f"  params={info.get('model_info', {}).get('total_params', '?'):,} "
            f"hidden={info.get('model_info', {}).get('hidden_size', '?')} "
            f"layers={info.get('model_info', {}).get('num_layers', '?')}"
        )

    def _load_norm_params(self):
        model_save_path = self.config["training"]["model_save_path"]
        if not os.path.isabs(model_save_path):
            model_save_path = str(_PROJECT_ROOT / model_save_path)
        norm_path = os.path.join(model_save_path, "norm_params.pkl")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"归一化参数不存在: {norm_path}")
        with open(norm_path, "rb") as f:
            meta = pickle.load(f)
        self.scaler_X = meta["scaler_X"]
        scaler_y = meta["scaler_y"]
        self.y_mean = np.asarray(scaler_y.mean_, dtype=np.float32)
        self.y_std = np.asarray(scaler_y.scale_, dtype=np.float32)
        self.logger.info(
            f"归一化参数加载完成: 输入维度={meta.get('input_size','?')} "
            f"输出维度={meta.get('output_size','?')}"
        )

    # -------------------------------------------------------------------
    # CSV 初始化
    # -------------------------------------------------------------------
    def _init_csv(self):
        out_path = Path(self.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(out_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            # ── 时间 & 阶段 ──
            "wall_time_s", "runtime_s", "boot_time_us", "phase",
            # ── 真值风（JSBSim WIND 消息）──
            "wind_gt_n", "wind_gt_e", "wind_gt_d",
            # ── EKF2 风估计（PX4 WIND_COV 消息）──
            "wind_ekf2_n", "wind_ekf2_e",
            # ── PIRNN-AKF 风估计（pirnn_akf 模式；baseline 填 NaN）──
            "wind_est_n", "wind_est_e", "wind_est_d",
            # ── 控制性能指标 ──
            "xtrack_error",     # 横向航迹误差 [m]      NAV_CONTROLLER_OUTPUT
            "airspeed",         # 实际空速 [m/s]        VFR_HUD
            "airspeed_err",     # airspeed - target [m/s]
            "groundspeed",      # 地速 [m/s]            VFR_HUD
            "throttle_pct",     # 油门百分比 [0-1]      VFR_HUD
            # ── 舵面原始输出（用于计算 jitter）──
            "aileron_raw",      # servo1_raw   SERVO_OUTPUT_RAW
            "elevator_raw",     # servo2_raw
            "throttle_raw",     # servo3_raw
            "rudder_raw",       # servo4_raw
            # ── 推理性能（baseline 填 NaN）──
            "inference_ms",
            "confidence",
        ])
        self.logger.info(f"CSV 输出: {out_path}")

    # -------------------------------------------------------------------
    # MAVLink 连接
    # -------------------------------------------------------------------
    def connect(self) -> bool:
        self.logger.info(f"连接 MAVLink: {self.connection_str}")
        try:
            if self.connection_str.startswith("/dev/"):
                baud = int(self.config.get("deployment", {}).get("baudrate", 921600))
                self.connection = mavutil.mavlink_connection(
                    self.connection_str, baud=baud
                )
            else:
                self.connection = mavutil.mavlink_connection(self.connection_str)

            self.connection.wait_heartbeat(timeout=30)
            sys_id = self.connection.target_system
            self.logger.info(f"✓ 连接成功 (System ID: {sys_id})")

            # 请求所有数据流 50 Hz
            self.connection.mav.request_data_stream_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                50, 1,
            )
            time.sleep(0.5)
            # 单独请求各关键消息（SITL 部分消息需显式请求）
            for msg_id, interval_us in [
                (62,  20000),   # NAV_CONTROLLER_OUTPUT      50 Hz
                (105, 20000),   # HIGHRES_IMU                50 Hz
                (116, 20000),   # SCALED_IMU2（HIGHRES_IMU 备选）50 Hz
                (36,  20000),   # SERVO_OUTPUT_RAW           50 Hz
                (30,  20000),   # ATTITUDE                   50 Hz
                (83,  20000),   # ATTITUDE_TARGET            50 Hz
                (85,  20000),   # POSITION_TARGET_LOCAL_NED  50 Hz
                (26,  20000),   # SCALED_IMU                 50 Hz
            ]:
                self.connection.mav.command_long_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0,
                    msg_id, interval_us,
                    0, 0, 0, 0, 0,
                )
            return True
        except Exception as exc:
            self.logger.error(f"❌ 连接失败: {exc}")
            return False

    # -------------------------------------------------------------------
    # MAVLink 数据收集（单帧）
    # -------------------------------------------------------------------
    def _collect_frame(self) -> Optional[dict]:
        """
        在 1 s 超时内收集一帧所需的必选消息。
        HIGHRES_IMU 为可选（SITL 有时不发送），缺失时用零向量代替。
        返回包含各消息类型 → 消息对象的字典，超时返回 None。
        """
        required = {"GLOBAL_POSITION_INT", "ATTITUDE", "VFR_HUD"}
        msg_dict: dict = {}
        deadline = time.time() + 1.0

        while time.time() < deadline:
            msg = self.connection.recv_match(blocking=False)
            if msg is not None and msg.get_type() != "BAD_DATA":
                msg_dict[msg.get_type()] = msg
                if required.issubset(msg_dict.keys()):
                    return msg_dict
            time.sleep(0.001)

        missing = required - msg_dict.keys()
        if missing:
            self.logger.warning(f"帧收集超时，缺少: {missing}")
            return None
        return msg_dict

    # -------------------------------------------------------------------
    # 特征提取（45 维，与训练数据保持一致）
    # -------------------------------------------------------------------
    def _extract_features(self, msg_dict: dict) -> Optional[np.ndarray]:
        features = np.zeros(45, dtype=np.float32)
        try:
            import math as _math
            gps        = msg_dict.get("GLOBAL_POSITION_INT")
            att        = msg_dict.get("ATTITUDE")
            imu        = msg_dict.get("HIGHRES_IMU")
            hud        = msg_dict.get("VFR_HUD")
            servo      = msg_dict.get("SERVO_OUTPUT_RAW")
            att_target = msg_dict.get("ATTITUDE_TARGET")
            pos_target = msg_dict.get("POSITION_TARGET_LOCAL_NED")

            # HIGHRES_IMU 不可用时改用 SCALED_IMU2（单位 mG → m/s²）
            # SCALED_IMU2 字段: xacc/yacc/zacc 单位 mg（milli-G），需 /1000*9.81
            if imu is None:
                simu2 = msg_dict.get("SCALED_IMU2")
                if simu2 is not None:
                    # 创建一个轻量命名空间代替 HIGHRES_IMU 对象
                    class _IMU:
                        pass
                    imu = _IMU()
                    imu.xacc = float(getattr(simu2, 'xacc', 0)) * 0.00981  # mg→m/s²
                    imu.yacc = float(getattr(simu2, 'yacc', 0)) * 0.00981
                    imu.zacc = float(getattr(simu2, 'zacc', 0)) * 0.00981
                else:
                    # 最终备选：SCALED_IMU
                    simu = msg_dict.get("SCALED_IMU")
                    if simu is not None:
                        class _IMU:
                            pass
                        imu = _IMU()
                        imu.xacc = float(getattr(simu, 'xacc', 0)) * 0.00981
                        imu.yacc = float(getattr(simu, 'yacc', 0)) * 0.00981
                        imu.zacc = float(getattr(simu, 'zacc', 0)) * 0.00981

            # 0-2: NED 地速
            if gps is not None:
                features[0] = gps.vx / 100.0
                features[1] = gps.vy / 100.0
                features[2] = gps.vz / 100.0

            # 3-5: 机体系地速
            if gps is not None and att is not None:
                vn, ve, vd = features[0], features[1], features[2]
                cr, sr = np.cos(att.roll), np.sin(att.roll)
                cp, sp = np.cos(att.pitch), np.sin(att.pitch)
                cy, sy_ = np.cos(att.yaw), np.sin(att.yaw)
                features[3] = cp * cy * vn + cp * sy_ * ve - sp * vd
                features[4] = (sr * sp * cy - cr * sy_) * vn + \
                               (sr * sp * sy_ + cr * cy) * ve + sr * cp * vd
                features[5] = (cr * sp * cy + sr * sy_) * vn + \
                               (cr * sp * sy_ - sr * cy) * ve + cr * cp * vd

            # 6-8: IMU 加速度（机体系，HIGHRES_IMU）
            if imu is not None:
                features[6] = imu.xacc
                features[7] = imu.yacc
                features[8] = imu.zacc

            # 9-14: 姿态 + 角速度
            if att is not None:
                features[9]  = att.roll
                features[10] = att.pitch
                features[11] = att.yaw
                features[12] = att.rollspeed
                features[13] = att.pitchspeed
                features[14] = att.yawspeed

            # 15-18: 舵面控制量 + 油门（SERVO_OUTPUT_RAW，归一化）
            if servo is not None:
                features[15] = (servo.servo1_raw - 1500) / 500.0   # aileron
                features[16] = (servo.servo2_raw - 1500) / 500.0   # elevator
                features[17] = (servo.servo4_raw - 1500) / 500.0   # rudder
                features[18] = float(np.clip(
                    (servo.servo3_raw - 1000.0) / 1000.0, 0.0, 1.0))  # throttle

            # 19: 空速
            if hud is not None:
                features[19] = hud.airspeed

            # 20-25: 目标姿态 + 姿态误差（ATTITUDE_TARGET）
            if att_target is not None:
                try:
                    q = att_target.q
                    if q is not None and len(q) >= 4:
                        w, x, y, z = q[0], q[1], q[2], q[3]
                        t_r  = +2.0 * (w * x + y * z)
                        t_rc = +1.0 - 2.0 * (x * x + y * y)
                        target_roll = _math.atan2(t_r, t_rc)
                        t_p = max(-1.0, min(1.0, +2.0 * (w * y - z * x)))
                        target_pitch = _math.asin(t_p)
                        t_y1 = +2.0 * (w * z + x * y)
                        t_y2 = +1.0 - 2.0 * (y * y + z * z)
                        target_yaw = _math.atan2(t_y1, t_y2)
                        features[20] = target_roll
                        features[21] = target_pitch
                        features[22] = target_yaw
                        if att is not None:
                            features[23] = att.roll  - target_roll
                            features[24] = att.pitch - target_pitch
                            dz = att.yaw - target_yaw
                            while dz >  _math.pi: dz -= 2 * _math.pi
                            while dz < -_math.pi: dz += 2 * _math.pi
                            features[25] = dz
                except Exception:
                    pass

            # 26-31: 目标角速度 + 角速率误差
            if att_target is not None:
                try:
                    features[26] = float(getattr(att_target, 'body_roll_rate',  0.0))
                    features[27] = float(getattr(att_target, 'body_pitch_rate', 0.0))
                    features[28] = float(getattr(att_target, 'body_yaw_rate',   0.0))
                    if att is not None:
                        features[29] = att.rollspeed  - features[26]
                        features[30] = att.pitchspeed - features[27]
                        features[31] = att.yawspeed   - features[28]
                except Exception:
                    pass

            # 32-37: 目标速度 + 速度误差（POSITION_TARGET_LOCAL_NED；固定翼常为 NaN）
            if pos_target is not None:
                try:
                    tvn = float(getattr(pos_target, 'vx', float('nan')))
                    tve = float(getattr(pos_target, 'vy', float('nan')))
                    tvd = float(getattr(pos_target, 'vz', float('nan')))
                    if np.isfinite(tvn) and np.isfinite(tve) and np.isfinite(tvd):
                        features[32] = tvn
                        features[33] = tve
                        features[34] = tvd
                        features[35] = features[0] - tvn
                        features[36] = features[1] - tve
                        features[37] = features[2] - tvd
                except Exception:
                    pass

            # 38-41: 实际舵面角度（物理单位，Rascal110 最大偏角估算）
            if servo is not None:
                features[38] = (servo.servo1_raw - 1500) / 500.0 * 0.35  # aileron_act  [rad]
                features[39] = (servo.servo2_raw - 1500) / 500.0 * 0.30  # elevator_act [rad]
                features[40] = (servo.servo4_raw - 1500) / 500.0 * 0.35  # rudder_act   [rad]
                features[41] = float(np.clip(
                    (servo.servo3_raw - 1000.0) / 1000.0, 0.0, 1.0))     # throttle_act

            # 42-44: IMU 机体加速度（与 6-8 相同来源）
            if imu is not None:
                features[42] = imu.xacc
                features[43] = imu.yacc
                features[44] = imu.zacc

            return features
        except Exception as exc:
            self.logger.error(f"特征提取失败: {exc}")
            return None

    # -------------------------------------------------------------------
    # PIRNN-AKF 推理（仅 pirnn_akf 模式）
    # -------------------------------------------------------------------
    def _run_pirnn_akf(self, msg_dict: dict) -> Optional[dict]:
        """执行一次 PI-GRU + AKF 推理，返回结果字典或 None。"""
        if len(self.data_buffer) < self.sequence_length:
            return None

        t0 = time.time()
        try:
            X_seq = np.array(list(self.data_buffer), dtype=np.float32)
            X_seq_norm = self.scaler_X.transform(X_seq)

            out = self.backend.infer(
                X_seq_norm,
                prev_log_q=self.prev_log_q,
                prev_log_r=self.prev_log_r,
                ema_alpha=self.ema_alpha_params,
                clamp=True,
            )
            self.prev_log_q = out.get("log_q_scale")
            self.prev_log_r = out.get("log_r_scale")

            # 反归一化 PI-GRU 输出
            w_norm = np.asarray(out["wind_estimate"], dtype=np.float32).reshape(-1)[:3]
            wind_nn = w_norm * self.y_std[:3] + self.y_mean[:3]

            if not np.all(np.isfinite(wind_nn)):
                self.logger.warning("PI-GRU 输出含 NaN，跳过")
                return None

            q_scale = np.asarray(
                out.get("q_scale", [1.0, 1.0, 1.0]), dtype=np.float32
            ).reshape(-1)[:3]
            r_scale = np.asarray(
                out.get("r_scale", [1.0, 1.0, 1.0]), dtype=np.float32
            ).reshape(-1)[:3]
            angles = np.asarray(
                out.get("angles", [0.0, 0.0, 1.0]), dtype=np.float32
            ).reshape(-1)[:3]
            conf_raw = out.get("confidence")
            confidence = float(
                np.asarray(conf_raw).reshape(-1)[0]
                if conf_raw is not None
                else 1.0 / (1.0 + np.std(q_scale))
            )

            # ── AKF 融合 ──
            wind_fused = wind_nn.copy()
            self.akf_warmup_count += 1

            if (
                self.akf_warmup_count > self.akf_warmup_threshold
                and msg_dict is not None
            ):
                try:
                    gps = msg_dict.get("GLOBAL_POSITION_INT")
                    att = msg_dict.get("ATTITUDE")
                    hud = msg_dict.get("VFR_HUD")

                    if gps is not None and att is not None and hud is not None:
                        z = np.array(
                            [
                                gps.vx / 100.0,
                                gps.vy / 100.0,
                                gps.vz / 100.0,
                                max(float(hud.airspeed), 0.1),
                            ],
                            dtype=np.float64,
                        )
                        # 运动学风量测
                        wind_kin, _, _ = self.akf.construct_kinematic_wind_measurement(
                            vg_ned=z[:3],
                            tas=float(z[3]),
                            roll=float(att.roll),
                            pitch=float(att.pitch),
                            yaw=float(att.yaw),
                            angles=angles,
                        )
                        # 抑制 kin 尖峰
                        wind_kin_arr = np.asarray(wind_kin, dtype=np.float64)
                        gap = float(np.linalg.norm(wind_kin_arr - wind_nn))
                        gap_ratio = np.clip(gap / 3.0, 0.0, 1.0)
                        kin_outlier = gap > 4.0 or np.linalg.norm(wind_kin_arr) > self.max_wind_speed * 1.2
                        kin_trust = np.clip(0.90 - 0.70 * gap_ratio, 0.20, 0.90)
                        if kin_outlier:
                            kin_trust = min(kin_trust, 0.35)
                        wind_kin_stable = kin_trust * wind_kin_arr + (1.0 - kin_trust) * wind_nn

                        q_eff = np.clip(q_scale * 1.0, 0.1, 20.0)
                        r_eff = np.clip(r_scale * (1.0 + 1.5 * gap_ratio), 0.1, 20.0)
                        if kin_outlier:
                            r_eff *= 2.0

                        if not self.akf_initialized:
                            self.akf.reset()
                            self.akf.x = (
                                0.80 * wind_nn.astype(np.float64)
                                + 0.20 * wind_kin_stable
                            )
                            self.akf_initialized = True
                            wind_delta = np.zeros(3, dtype=np.float64)
                        else:
                            wind_delta = (
                                wind_nn - self.prev_wind_nn
                                if self.prev_wind_nn is not None
                                else np.zeros(3, dtype=np.float64)
                            )

                        self.akf.update_noise_covariance(q_eff, r_eff)
                        self.akf.predict(neural_wind_delta=wind_delta)

                        if not (
                            np.any(np.isnan(self.akf.P))
                            or np.any(np.diag(self.akf.P) > 1e4)
                        ):
                            self.akf.update(
                                z,
                                roll=float(att.roll),
                                pitch=float(att.pitch),
                                yaw=float(att.yaw),
                                confidence=confidence,
                                angles=angles,
                                nn_measurement=wind_nn,
                                maneuver_score=0.0,
                                wind_kin_override=wind_kin_stable,
                            )
                            w_akf = self.akf.get_wind_estimate()
                            if np.all(np.isfinite(w_akf)) and np.linalg.norm(w_akf) < self.max_wind_speed * 2:
                                akf_w = np.clip(0.58 + 0.10 * confidence - 0.28 * gap_ratio, 0.18, 0.72)
                                if kin_outlier:
                                    akf_w = min(akf_w, 0.28)
                                wind_fused = akf_w * w_akf + (1.0 - akf_w) * wind_nn
                        else:
                            self.akf.reset()
                            self.akf_initialized = False

                except Exception as exc:
                    self.logger.debug(f"AKF 融合异常（使用 PI-GRU 输出）: {exc}")

            # EMA 平滑
            if self.ema_wind is None:
                self.ema_wind = wind_fused.copy()
            else:
                self.ema_wind = (
                    self.ema_alpha_wind * wind_fused
                    + (1.0 - self.ema_alpha_wind) * self.ema_wind
                )
            wind_final = self.ema_wind.astype(np.float32)

            # 安全检查
            if not np.all(np.isfinite(wind_final)) or np.linalg.norm(wind_final) > self.max_wind_speed:
                return None

            self.prev_wind_nn = wind_nn.copy()
            inference_ms = (time.time() - t0) * 1000.0
            self._infer_count += 1
            self._infer_total_ms += inference_ms
            self._infer_max_ms = max(self._infer_max_ms, inference_ms)

            return {
                "wind_est": wind_final,
                "q_scale": q_scale,
                "r_scale": r_scale,
                "angles": angles,
                "confidence": confidence,
                "inference_ms": inference_ms,
            }

        except Exception as exc:
            self.logger.error(f"推理失败: {exc}")
            return None

    # -------------------------------------------------------------------
    # 发送 WIND_COV 到 PX4
    # -------------------------------------------------------------------
    def _send_wind_cov(self, wind: np.ndarray):
        try:
            self.connection.mav.wind_cov_send(
                int(time.time() * 1_000_000),
                float(wind[0]),
                float(wind[1]),
                float(wind[2]),
                0.0, 0.0, 0.0, 0.0, 0.0,
            )
        except Exception as exc:
            self.logger.warning(f"WIND_COV 发送失败: {exc}")

    # -------------------------------------------------------------------
    # 从 WIND 消息提取真值风
    # -------------------------------------------------------------------
    @staticmethod
    def _parse_wind_truth(msg_dict: dict) -> tuple[float, float, float]:
        """从 WIND MAVLink 消息解析真值风 (N, E, D)，不存在返回 NaN。"""
        w = msg_dict.get("WIND")
        if w is None:
            return float("nan"), float("nan"), float("nan")
        direction_rad = float(getattr(w, "direction", 0.0)) * np.pi / 180.0
        speed = float(getattr(w, "speed", 0.0))
        speed_z = float(getattr(w, "speed_z", 0.0))
        return (
            speed * np.cos(direction_rad),
            speed * np.sin(direction_rad),
            -speed_z,
        )

    # -------------------------------------------------------------------
    # 写入一行 CSV
    # -------------------------------------------------------------------
    def _log_row(
        self,
        runtime_s: float,
        phase: str,
        msg_dict: dict,
        pirnn_result: Optional[dict],
    ):
        if self._csv_writer is None:
            return

        # boot_time_us（与 ULog 对齐）
        boot_time_us = 0
        for key in ("HIGHRES_IMU", "GLOBAL_POSITION_INT"):
            if key in msg_dict:
                t = getattr(msg_dict[key], "time_usec", None) or \
                    getattr(msg_dict[key], "time_boot_ms", 0) * 1000
                boot_time_us = int(t)
                break

        # 真值风
        wg_n, wg_e, wg_d = self._parse_wind_truth(msg_dict)

        # EKF2 风估计
        wc = msg_dict.get("WIND_COV")
        ekf_n = float(getattr(wc, "wind_x", float("nan"))) if wc else float("nan")
        ekf_e = float(getattr(wc, "wind_y", float("nan"))) if wc else float("nan")

        # PIRNN-AKF 估计
        if pirnn_result is not None:
            we = pirnn_result["wind_est"]
            west_n, west_e, west_d = float(we[0]), float(we[1]), float(we[2])
            infer_ms = pirnn_result["inference_ms"]
            confidence = pirnn_result["confidence"]
        else:
            west_n = west_e = west_d = float("nan")
            infer_ms = float("nan")
            confidence = float("nan")

        # 控制指标
        nav = msg_dict.get("NAV_CONTROLLER_OUTPUT")
        xtrack = float(getattr(nav, "xtrack_error", float("nan"))) if nav else float("nan")

        hud = msg_dict.get("VFR_HUD")
        airspeed = float(getattr(hud, "airspeed", float("nan"))) if hud else float("nan")
        groundspeed = float(getattr(hud, "groundspeed", float("nan"))) if hud else float("nan")
        throttle_pct = float(getattr(hud, "throttle", float("nan"))) if hud else float("nan")
        airspeed_err = airspeed - self.target_airspeed if np.isfinite(airspeed) else float("nan")

        servo = msg_dict.get("SERVO_OUTPUT_RAW")
        if servo is not None:
            ail_raw = int(getattr(servo, "servo1_raw", 1500))
            elev_raw = int(getattr(servo, "servo2_raw", 1500))
            thr_raw = int(getattr(servo, "servo3_raw", 1000))
            rud_raw = int(getattr(servo, "servo4_raw", 1500))
        else:
            ail_raw = elev_raw = thr_raw = rud_raw = -1

        self._csv_writer.writerow([
            f"{time.time():.3f}",
            f"{runtime_s:.3f}",
            boot_time_us,
            phase,
            f"{wg_n:.4f}", f"{wg_e:.4f}", f"{wg_d:.4f}",
            f"{ekf_n:.4f}", f"{ekf_e:.4f}",
            f"{west_n:.4f}", f"{west_e:.4f}", f"{west_d:.4f}",
            f"{xtrack:.4f}",
            f"{airspeed:.4f}",
            f"{airspeed_err:.4f}",
            f"{groundspeed:.4f}",
            f"{throttle_pct:.2f}",
            ail_raw, elev_raw, thr_raw, rud_raw,
            f"{infer_ms:.3f}",
            f"{confidence:.4f}",
        ])
        self._csv_file.flush()

    # -------------------------------------------------------------------
    # 打印实时状态
    # -------------------------------------------------------------------
    def _print_status(
        self,
        runtime_s: float,
        phase: str,
        msg_dict: dict,
        pirnn_result: Optional[dict],
    ):
        hud = msg_dict.get("VFR_HUD")
        nav = msg_dict.get("NAV_CONTROLLER_OUTPUT")
        airspeed = float(getattr(hud, "airspeed", 0.0)) if hud else 0.0
        xtrack = float(getattr(nav, "xtrack_error", 0.0)) if nav else 0.0

        if pirnn_result is not None:
            w = pirnn_result["wind_est"]
            wind_str = f"[{w[0]:5.2f},{w[1]:5.2f},{w[2]:5.2f}]"
            lat_str = f"{pirnn_result['inference_ms']:.1f}ms"
        else:
            wind_str = "[EKF2 only]"
            lat_str = "--"

        print(
            f"\r[{phase:11s}] t={runtime_s:6.1f}s | "
            f"XTE={xtrack:6.2f}m | "
            f"AS={airspeed:5.2f}m/s (err={airspeed-self.target_airspeed:+.2f}) | "
            f"wind_est={wind_str} | lat={lat_str}",
            end="",
            flush=True,
        )

    # -------------------------------------------------------------------
    # 打印最终统计
    # -------------------------------------------------------------------
    def _print_summary(self, runtime_s: float):
        print("\n" + "=" * 65)
        print(f"  实验结束 ─ 模式: {self.mode}")
        print("=" * 65)
        print(f"  总运行时长  : {runtime_s:.1f} s")
        if self._infer_count > 0:
            print(f"  推理次数    : {self._infer_count}")
            print(f"  平均推理时延: {self._infer_total_ms / self._infer_count:.2f} ms")
            print(f"  最大推理时延: {self._infer_max_ms:.2f} ms")
            print(f"  实际推理频率: {self._infer_count / runtime_s:.1f} Hz")
        print(f"  输出 CSV    : {self.output_csv}")
        print("=" * 65)

    # -------------------------------------------------------------------
    # 主运行循环
    # -------------------------------------------------------------------
    def run(self):
        print("=" * 65)
        print(f"  SITL 闭环评估  ─  模式: {self.mode}")
        print("=" * 65)

        if not self.connect():
            self.logger.error("无法连接 MAVLink，退出")
            return

        print(f"\n开始评估... 按 Ctrl+C 停止\n")
        print(
            f"  目标空速    : {self.target_airspeed} m/s\n"
            f"  评估时长上限: {self.eval_duration_s if self.eval_duration_s else '不限'} s\n"
            f"  发送 WIND_COV: {self.send_wind_cov_flag}\n"
        )

        self._start_wall = time.time()
        loop_interval = 1.0 / self.inference_rate
        row_count = 0

        try:
            while True:
                t_loop = time.time()
                runtime_s = t_loop - self._start_wall
                phase = _get_phase(runtime_s)

                # ── 评估时长检查 ──
                if self.eval_duration_s and runtime_s >= self.eval_duration_s:
                    self.logger.info(f"达到评估时长上限 {self.eval_duration_s:.0f} s，停止")
                    break

                # ── 收集 MAVLink 帧 ──
                msg_dict = self._collect_frame()
                if msg_dict is None:
                    continue

                # ── 特征提取 & 推理 ──
                pirnn_result: Optional[dict] = None
                if self.mode == "pirnn_akf":
                    features = self._extract_features(msg_dict)
                    if features is not None:
                        self.data_buffer.append(features)
                    pirnn_result = self._run_pirnn_akf(msg_dict)

                    # 发送 WIND_COV
                    if pirnn_result is not None and self.send_wind_cov_flag:
                        self._send_wind_cov(pirnn_result["wind_est"])

                # ── 记录 CSV ──
                self._log_row(runtime_s, phase, msg_dict, pirnn_result)
                row_count += 1

                # ── 终端打印（每 10 行）──
                if row_count % 10 == 0:
                    self._print_status(runtime_s, phase, msg_dict, pirnn_result)

                # ── 频率控制 ──
                elapsed = time.time() - t_loop
                if elapsed < loop_interval:
                    time.sleep(loop_interval - elapsed)

        except KeyboardInterrupt:
            print("\n\n收到停止信号 (Ctrl+C)")

        except Exception as exc:
            self.logger.error(f"运行时异常: {exc}")
            import traceback
            traceback.print_exc()

        finally:
            runtime_s = time.time() - self._start_wall
            self._print_summary(runtime_s)

            if self._csv_file is not None:
                self._csv_file.close()
                self.logger.info(f"CSV 已保存: {self.output_csv}  ({row_count} 行)")

            if self.backend is not None:
                self.backend.close()

            if self.connection is not None:
                self.connection.close()
                self.logger.info("MAVLink 连接已关闭")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="SITL 闭环控制效益评估 (PIRNN-AKF vs. EKF2 baseline)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例
----
# Step 1: 跑基准组（PX4 内置 EKF2 风估计）
python3 7_sitl_closedloop_eval.py \\
    --mode baseline \\
    --connection udpin:0.0.0.0:14550 \\
    --output ../SITL/sitl_baseline_$(date +%Y%m%d_%H%M%S).csv

# Step 2: 跑 PIRNN-AKF 组
python3 7_sitl_closedloop_eval.py \\
    --mode pirnn_akf \\
    --connection udpin:0.0.0.0:14550 \\
    --output ../SITL/sitl_pirnn_akf_$(date +%Y%m%d_%H%M%S).csv

# Step 3: 生成对比图
python3 ../scripts/analyze_sitl_closedloop.py \\
    --baseline ../SITL/sitl_baseline_*.csv \\
    --pirnn    ../SITL/sitl_pirnn_akf_*.csv

注意
----
- PX4 SITL 必须已在运行，jsbsim_bridge 使用相同的五阶段风场脚本
- 默认连接 UDP 14550；HITL 串口模式改为 --connection /dev/ttyAMA0
- 两组实验须使用相同的 JSBSim 风场配置文件和相同的任务航线
- --duration 设置自动停止时间（秒）；不设则需手动 Ctrl+C（约 22 分钟）
""",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "pirnn_akf"],
        help="实验模式",
    )
    parser.add_argument(
        "--connection",
        default="udpin:0.0.0.0:14550",
        help="MAVLink 连接字符串，默认 udpin:0.0.0.0:14550",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 CSV 路径；默认 ../SITL/sitl_{mode}_{timestamp}.csv",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="config.yaml 路径；默认自动查找",
    )
    parser.add_argument(
        "--target-airspeed",
        type=float,
        default=12.0,
        help="目标巡航空速 [m/s]，与训练数据一致 (10.5-13.5 m/s，默认 12 m/s)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="实验时长上限 [s]；不设则持续到 Ctrl+C（完整五阶段约 1200 s）",
    )
    parser.add_argument(
        "--no-send-wind",
        action="store_true",
        help="pirnn_akf 模式下禁止发送 WIND_COV（只推理不注入）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # 自动生成输出路径
    if args.output is None:
        sitl_dir = _PROJECT_ROOT / "SITL"
        sitl_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.output = str(sitl_dir / f"sitl_{args.mode}_{ts}.csv")

    evaluator = SITLClosedLoopEvaluator(
        mode=args.mode,
        connection_str=args.connection,
        output_csv=args.output,
        config_path=args.config,
        target_airspeed=args.target_airspeed,
        eval_duration_s=args.duration,
        send_wind_cov=not args.no_send_wind,
    )
    evaluator.run()
