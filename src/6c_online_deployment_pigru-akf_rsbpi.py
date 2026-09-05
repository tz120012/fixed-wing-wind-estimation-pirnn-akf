"""
在线部署模块 v3.0 (EKF融合增强版)
功能：
  - 实时接收MAVLink数据
  - 模型推理（支持字典输出和多通道q_scale/r_scale）
  - 对数域EMA平滑
  - 扩展的MAVLink输出（包含q_scale/r_scale和小角修正）
  - 性能监控和日志记录
  - 预留EKF融合接口
"""

import numpy as np
import pickle
import yaml
import time
import logging
from collections import deque
from pymavlink import mavutil
import sys
import os

from inference_backends import create_backend
from online_feature_extractor import StreamingFeatureExtractor, select_model_features
import importlib.util

_akf_module_path = os.path.join(os.path.dirname(__file__), '5_pigru_akf_fusion.py')
_akf_spec = importlib.util.spec_from_file_location('pigru_akf_fusion', _akf_module_path)
if _akf_spec and _akf_spec.loader:
    _akf_module = importlib.util.module_from_spec(_akf_spec)
    _akf_spec.loader.exec_module(_akf_module)
    AdaptiveKalmanFilter = _akf_module.AdaptiveKalmanFilter
else:
    raise ImportError('Cannot load 5_pigru_akf_fusion.py')


class ExponentialMovingAverageFilter:
    """指数移动平均滤波器（用于输出平滑）"""
    
    def __init__(self, alpha=0.2, initial_value=None):
        """
        Args:
            alpha: 平滑系数 (0-1)，越小越平滑
            initial_value: 初始值
        """
        self.alpha = alpha
        self.value = initial_value
    
    def update(self, new_value):
        """更新滤波器"""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def reset(self):
        """重置滤波器"""
        self.value = None


class OnlineWindEstimator:
    """在线风速估计器"""
    
    def __init__(self, config_path=None):
        """
        初始化在线估计器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if config_path is None:
            config_path = os.path.join(script_dir, '..', 'config', 'config.yaml')
        elif not os.path.isabs(config_path):
            config_path = os.path.normpath(os.path.join(script_dir, config_path))
        config_path = os.path.abspath(config_path)

        # 统一项目根目录
        self.project_root = os.path.dirname(os.path.dirname(config_path))

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 将配置中的相对路径转为基于项目根目录的绝对路径
        def _resolve(path_str):
            if path_str and not os.path.isabs(path_str):
                return os.path.normpath(os.path.join(self.project_root, path_str))
            return path_str
        
        self.config['training']['model_save_path'] = _resolve(self.config['training']['model_save_path'])
        self.config['deployment']['model_path_torch'] = _resolve(self.config['deployment']['model_path_torch'])
        self.config['deployment']['model_path_ascend'] = _resolve(self.config['deployment']['model_path_ascend'])
        self.config['logging']['save_dir'] = _resolve(self.config['logging']['save_dir'])
        
        # 部署配置
        deploy_config = self.config.get('deployment', {})
        self.backend_name = str(deploy_config.get('backend', 'torch')).lower()
        self.mavlink_connection = deploy_config.get('mavlink_connection', 'udpin:0.0.0.0:14550')
        self.baudrate = int(deploy_config.get('baudrate', 921600))
        self.send_wind_cov = bool(deploy_config.get('send_wind_cov', True))
        self.send_named_values = bool(deploy_config.get('send_named_values', True))
        self.inference_rate = deploy_config.get('inference_rate', 10.0)  # Hz
        self.sequence_length = self.config['data']['sequence_length']
        self.hitl_session_id = os.environ.get(
            'HITL_SESSION_ID',
            str(deploy_config.get('hitl_session_id', 'unspecified')),
        )
        self.hitl_condition = os.environ.get(
            'HITL_CONDITION',
            str(deploy_config.get('hitl_condition', 'unspecified')),
        ).lower()
        
        # EMA配置
        self.ema_alpha_wind = deploy_config.get('ema_alpha_wind', 0.2)
        self.ema_alpha_params = deploy_config.get('ema_alpha_params', 0.1)  # q_scale/r_scale的EMA系数
        
        # 安全限制
        self.max_wind_speed = deploy_config.get('max_wind_speed', 20.0)
        self.max_inference_time = deploy_config.get('max_inference_time_ms', 50.0) / 1000.0
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=self.sequence_length)

        # 先装配完整 45 维原子特征；加载归一化元数据后按冻结协议选择 45D 或最终 41D。
        self.feature_extractor = StreamingFeatureExtractor(
            sampling_rate=float(self.inference_rate)
        )
        # 可选 MAVLink 消息（目标量/舵面）last-known-value 缓存
        self._latest_msgs = {}
        # 方案C: 消息到达计数（用于诊断期望量/舵面是否长期缺失退化为默认值）
        self._msg_counts = {}
        # 方案C: 上一帧成功时刻（monotonic 秒），用于估计真实帧间 dt 供因果加速度
        self._last_frame_t = None
        # 诊断: 累计归一化后特征的均值/方差，定位 live 下哪些特征偏离训练分布
        self._featnorm_sum = np.zeros(45, dtype=np.float64)
        self._featnorm_sqsum = np.zeros(45, dtype=np.float64)
        self._featnorm_n = 0
        # 方案C 核心修复: HITL/MAVLink 下舵面/油门(cmd+act)来自 SERVO_OUTPUT_RAW/RC_CHANNELS，
        # 其 PWM->[-1,1] 全量程尺度与训练所用 JSBSim fcs-cmd-norm(极小、带配平偏置、std~0.02-0.1)
        # 完全不同，直接喂会造成 ±15~48σ 越界并主导 GRU、毁掉风估计。
        # 由于风信号主要来自运动学特征(速度/空速/姿态)，这里将无法忠实复现的舵面/油门特征
        # 用训练均值中性填充(归一化后≈0)。可用 deployment.neutralize_ctrl_features=false 关闭。
        self.neutralize_ctrl_features = bool(deploy_config.get('neutralize_ctrl_features', True))
        self._neutral_feat_idx = [15, 16, 17, 18, 38, 39, 40, 41]
        
        # 风速输出滤波器
        self.output_filter = ExponentialMovingAverageFilter(
            alpha=self.ema_alpha_wind
        ) if deploy_config.get('enable_output_filter', True) else None
        
        # 模型状态（用于q_scale/r_scale平滑）
        self.prev_log_q = None
        self.prev_log_r = None
        self.backend = None
        self.device = 'unknown'
        
        # AKF融合模块
        dt = 1.0 / self.inference_rate
        akf_cfg = self.config.get('akf', {}) or {}
        self.akf_constants = {
            'prediction_delta_gain': 0.25,
            'mahalanobis_gate': 9.0,
            'maneuver_gyro_weight': 0.45,
            'maneuver_acc_weight': 0.30,
            'maneuver_control_weight': 0.20,
            'maneuver_throttle_weight': 0.05,
            'disagreement_scale': 3.0,
            'kinematic_trust_base': 0.90,
            'kinematic_trust_slope': 0.70,
            'kinematic_trust_min': 0.20,
            'kinematic_trust_max': 0.90,
            'q_maneuver_gain': 0.45,
            'r_maneuver_gain': 0.20,
            'r_disagreement_gain': 1.50,
            'r_outlier_multiplier': 2.0,
            'fusion_base': 0.58,
            'fusion_confidence_gain': 0.10,
            'fusion_disagreement_gain': 0.28,
            'fusion_covariance_gain': 0.10,
            'fusion_min': 0.18,
            'fusion_max': 0.72,
            'fusion_outlier_cap': 0.28,
        }
        self.akf_constants.update(akf_cfg.get('constants', {}) or {})
        q_nominal = np.diag(akf_cfg.get('q_nominal', [0.30, 0.30, 0.12]))
        r_kin_nominal = np.diag(akf_cfg.get('r_kin_nominal', [1.50, 1.50, 0.60]))
        r_nn_nominal = np.diag(akf_cfg.get('r_nn_nominal', [0.06, 0.06, 0.04]))
        self.akf = AdaptiveKalmanFilter(
            dt=dt,
            Q_nominal=q_nominal,
            R_kin_nominal=r_kin_nominal,
            R_nn_nominal=r_nn_nominal,
            prediction_delta_gain=self.akf_constants['prediction_delta_gain'],
            outlier_threshold=self.akf_constants['mahalanobis_gate'],
        )
        self.last_msg_dict = None  # 保存最近的MAVLink消息用于AKF
        self.akf_initialized = False  # AKF是否已用首次有效PI-GRU输出初始化
        self.akf_warmup_count = 0     # PI-GRU预热计数
        self.akf_warmup_threshold = 10  # 预热帧数，前N帧只用PI-GRU
        
        # 性能统计
        self.performance = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'max_inference_time': 0.0,
            'invalid_estimates': 0,
            'start_time': time.time()
        }
        
        # 最近的估计结果
        self.last_wind_estimate = None
        self.last_q_scale = None
        self.last_r_scale = None
        self.last_angles = None
        
        # 日志设置
        self.setup_logging()
        
        # CSV数据记录
        self.csv_file = None
        self.csv_writer = None
        self._init_csv_logger()
        
        # 加载后端与归一化参数
        self.load_backend()
        self.load_normalization_params()
        
        # MAVLink连接
        self.connection = None
        
        self.logger.info("="*70)
        self.logger.info("Online wind estimator initialized")
        self.logger.info("="*70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Backend: {self.backend_name}")
        self.logger.info(f"Inference rate: {self.inference_rate} Hz")
        self.logger.info(f"Sequence length: {self.sequence_length}")
        self.logger.info(f"EMA alpha (wind): {self.ema_alpha_wind}")
        self.logger.info(f"EMA alpha (q/r): {self.ema_alpha_params}")
        self.logger.info(f"Max wind speed: {self.max_wind_speed} m/s")
        self.logger.info("="*70)

    def _resolve_project_path(self, path_value, default_value):
        """将路径解析为项目根目录下的绝对路径。"""
        raw = path_value if path_value else default_value
        if os.path.isabs(raw):
            return raw
        return os.path.normpath(os.path.join(self.project_root, raw))
    
    def setup_logging(self):
        """设置日志"""
        log_dir_cfg = self.config.get('logging', {}).get('log_dir', '../logs/')
        log_dir = self._resolve_project_path(log_dir_cfg, '../logs/')
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f'online_deployment_{time.strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Log file: {log_file}")
    
    def _init_csv_logger(self):
        """Initialize CSV data logger"""
        import csv
        log_dir_cfg = self.config.get('logging', {}).get('log_dir', '../logs/')
        log_dir = self._resolve_project_path(log_dir_cfg, '../logs/')
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f'hitl_data_{time.strftime("%Y%m%d_%H%M%S")}.csv')
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'boot_time_us', 'timestamp', 'runtime_s',
            'wind_n', 'wind_e', 'wind_d', 'wind_mag', 'wind_dir',
            'q_scale_n', 'q_scale_e', 'q_scale_d',
            'r_scale_gps', 'r_scale_tas', 'r_scale_att',
            'inference_ms',
            # --- 图13/14 补充字段 ---
            't_fc_send_us',      # 飞控消息时间戳（us），与 ulog 对齐
            't_recv_us',         # 伴机收到消息的墙钟时间（us）
            'latency_comm_ms',   # 通信时延 = t_recv - t_fc_send（需时钟同步有意义）
            'latency_e2e_ms',    # 端到端时延 = inference完成 - t_recv
            'wind_gt_n', 'wind_gt_e', 'wind_gt_d',  # JSBSim 真值（来自 MAVLink WIND 消息）
            'phase',             # 扰动阶段标签（steady/gust_light/gust_strong/packet_loss）
            # --- 诊断字段（归因欠估：模型 vs 融合）---
            'wind_nn_n', 'wind_nn_e', 'wind_nn_d',  # PI-GRU 原始输出（AKF/EMA 融合前）
            'gs_mag', 'airspeed_ms',                 # 地速幅值 / 空速（运动学一致性检查）
            # --- repeated-HITL frozen protocol fields ---
            'session_id', 'condition', 'fc_boot_time_us',
            'companion_monotonic_ns', 'truth_time_us',
            'truth_wind_n_mps', 'truth_wind_e_mps', 'truth_wind_d_mps',
            'estimated_wind_n_mps', 'estimated_wind_e_mps', 'estimated_wind_d_mps',
            'inference_latency_ms', 'companion_processing_latency_ms',
            'deadline_missed',
        ])
        self.logger.info(f"CSV data file: {csv_path}")
    
    def _log_csv(self, result, msg_dict, t_recv_ns=None, phase='steady'):
        """Write one row to CSV.

        Args:
            result:      推理结果字典（含 wind_estimate / inference_time 等）
            msg_dict:    本帧 MAVLink 消息字典
            t_recv_ns:   伴机收到 MAVLink 包时的 time.monotonic_ns() 值（由 run() 传入）
            phase:       扰动阶段标签，如 'steady'/'gust_light'/'gust_strong'/'packet_loss'
        """
        if self.csv_writer is None:
            return

        now_ns = time.monotonic_ns()

        # ── 飞控时间戳（用于与 ulog 对齐，也用于计算通信时延）──
        t_fc_send_us = 0
        for key in ('HIGHRES_IMU', 'GLOBAL_POSITION_INT'):
            if key in msg_dict:
                t_fc_send_us = int(getattr(msg_dict[key], 'time_usec', 0) or
                                   getattr(msg_dict[key], 'time_boot_ms', 0) * 1000)
                break

        boot_time_us = t_fc_send_us

        # ── 时延计算 ──
        t_recv_us   = int(t_recv_ns / 1000) if t_recv_ns is not None else int(now_ns / 1000)
        t_output_us = int(now_ns / 1000)
        if t_fc_send_us > 0 and t_recv_us > 0:
            latency_comm_ms = (t_recv_us - t_fc_send_us) / 1000.0
            if latency_comm_ms < -50.0 or latency_comm_ms > 500.0:
                latency_comm_ms = -1.0
        else:
            latency_comm_ms = -1.0
        latency_e2e_ms = (t_output_us - t_recv_us) / 1000.0

        # ── 真值风速（JSBSim 通过 WIND 消息广播，SITL 环境可用）──
        wind_gt = [float('nan'), float('nan'), float('nan')]
        if 'WIND' in msg_dict:
            wm = msg_dict['WIND']
            direction_rad = float(getattr(wm, 'direction', 0)) * np.pi / 180.0
            speed   = float(getattr(wm, 'speed',   0))
            speed_z = float(getattr(wm, 'speed_z', 0))
            wind_gt[0] =  speed * np.cos(direction_rad)
            wind_gt[1] =  speed * np.sin(direction_rad)
            wind_gt[2] = -speed_z

        # ── 运动学一致性诊断量：地速幅值 / 空速 ──
        gs_mag = float('nan')
        airspeed_ms = float('nan')
        gps = msg_dict.get('GLOBAL_POSITION_INT')
        if gps is not None:
            gs_mag = float(np.linalg.norm([gps.vx / 100.0, gps.vy / 100.0, gps.vz / 100.0]))
        hud = msg_dict.get('VFR_HUD')
        if hud is not None:
            airspeed_ms = float(hud.airspeed)

        w   = result['wind_estimate']
        a   = result['q_scale']
        b   = result['r_scale']
        mag = float(np.linalg.norm(w))
        d   = float(np.arctan2(w[1], w[0]) * 180 / np.pi)

        self.csv_writer.writerow([
            boot_time_us,
            f'{time.time():.3f}',
            f'{time.time() - self.performance["start_time"]:.3f}',
            f'{w[0]:.4f}', f'{w[1]:.4f}', f'{w[2]:.4f}',
            f'{mag:.4f}', f'{d:.1f}',
            f'{a[0]:.4f}', f'{a[1]:.4f}', f'{a[2]:.4f}',
            f'{b[0]:.4f}', f'{b[1]:.4f}', f'{b[2]:.4f}',
            f'{result["inference_time"]*1000:.2f}',
            # --- 补充字段 ---
            t_fc_send_us,
            t_recv_us,
            f'{latency_comm_ms:.3f}',
            f'{latency_e2e_ms:.3f}',
            f'{wind_gt[0]:.4f}', f'{wind_gt[1]:.4f}', f'{wind_gt[2]:.4f}',
            phase,
            # --- 诊断字段 ---
            *(f'{v:.4f}' for v in np.asarray(result.get('wind_nn', [np.nan]*3),
                                             dtype=np.float32).reshape(-1)[:3]),
            f'{gs_mag:.4f}', f'{airspeed_ms:.4f}',
            # --- repeated-HITL frozen protocol fields ---
            self.hitl_session_id, self.hitl_condition, boot_time_us,
            now_ns, t_fc_send_us,
            f'{wind_gt[0]:.4f}', f'{wind_gt[1]:.4f}', f'{wind_gt[2]:.4f}',
            f'{w[0]:.4f}', f'{w[1]:.4f}', f'{w[2]:.4f}',
            f'{result["inference_time"]*1000:.3f}',
            f'{latency_e2e_ms:.3f}',
            int(latency_e2e_ms > (1000.0 / float(self.inference_rate))),
        ])
        self.csv_file.flush()
    
    def load_backend(self):
        """加载推理后端"""
        self.backend = create_backend(self.config)
        backend_info = self.backend.load()
        self.device = self.backend.device_name

        model_path = backend_info.get('model_path')
        if model_path:
            self.logger.info(f"Loading model: {model_path}")

        model_info = backend_info.get('model_info')
        if model_info:
            self.logger.info(f"Model params: {model_info.get('total_params', 0):,}")
            self.logger.info(f"Hidden size: {model_info.get('hidden_size', 'Unknown')}")
            self.logger.info(f"GRU layers: {model_info.get('num_layers', 'Unknown')}")

        if backend_info.get('checkpoint_epoch') is not None:
            self.logger.info(f"Training epochs: {backend_info['checkpoint_epoch']}")
        if backend_info.get('best_val_loss') is not None:
            self.logger.info(f"Best val loss: {backend_info['best_val_loss']:.4f}")
    
    def load_normalization_params(self):
        """加载归一化参数"""
        norm_path = os.path.join(
            self.config['training']['model_save_path'],
            'norm_params.pkl'
        )
        
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"Normalization params not found: {norm_path}")
        
        self.logger.info(f"Loading normalization params: {norm_path}")
        
        with open(norm_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']
        self.input_size = int(metadata.get('input_size', len(self.scaler_X.mean_)))
        if self.input_size == 41:
            self.feature_keep_indices = np.asarray(
                [i for i in range(45) if i not in (38, 39, 40, 41)],
                dtype=np.int64,
            )
            self._neutral_feat_idx = [15, 16, 17, 18]
        elif self.input_size == 45:
            self.feature_keep_indices = np.arange(45, dtype=np.int64)
            self._neutral_feat_idx = [15, 16, 17, 18, 38, 39, 40, 41]
        else:
            raise ValueError(
                f"Unsupported input_size={self.input_size}; expected frozen 41D or legacy 45D"
            )
        self._featnorm_sum = np.zeros(self.input_size, dtype=np.float64)
        self._featnorm_sqsum = np.zeros(self.input_size, dtype=np.float64)
        
        # 提取风速的归一化参数（用于反归一化）
        self.y_mean = np.asarray(self.scaler_y.mean_, dtype=np.float32)
        self.y_std = np.asarray(self.scaler_y.scale_, dtype=np.float32)
        
        self.logger.info(f"Input dim: {metadata.get('input_size', 'Unknown')}")
        self.logger.info(f"Output dim: {metadata.get('output_size', 'Unknown')}")
    
    def _compute_maneuver_score(self, last_step_phys):
        """根据末帧20维特征估计当前机动强度。"""
        gyro = np.asarray(last_step_phys[12:15], dtype=np.float32)
        acc = np.asarray(last_step_phys[6:9], dtype=np.float32)
        ctrl = np.asarray(last_step_phys[15:18], dtype=np.float32)
        throttle = float(last_step_phys[18])

        gyro_score = np.linalg.norm(gyro) / 1.2
        acc_score = np.linalg.norm(acc) / 8.0
        ctrl_score = np.linalg.norm(ctrl) / 1.2
        throttle_score = abs(throttle - 0.5) * 2.0

        c = self.akf_constants
        maneuver = (
            c['maneuver_gyro_weight'] * gyro_score
            + c['maneuver_acc_weight'] * acc_score
            + c['maneuver_control_weight'] * ctrl_score
            + c['maneuver_throttle_weight'] * throttle_score
        )
        return float(np.clip(maneuver, 0.0, 3.0))

    def _stabilize_kinematic_measurement(self, wind_kin, wind_nn):
        """抑制在线wind_kin异常尖峰，避免AKF被瞬时量测带偏。"""
        wind_kin = np.asarray(wind_kin, dtype=np.float64)
        wind_nn = np.asarray(wind_nn, dtype=np.float64)
        gap_vec = wind_kin - wind_nn
        measurement_gap = float(np.linalg.norm(gap_vec))
        c = self.akf_constants
        gap_ratio = np.clip(measurement_gap / c['disagreement_scale'], 0.0, 1.0)
        kin_norm = float(np.linalg.norm(wind_kin))
        kin_outlier = measurement_gap > 4.0 or kin_norm > self.max_wind_speed * 1.2

        kin_trust = np.clip(
            c['kinematic_trust_base'] - c['kinematic_trust_slope'] * gap_ratio,
            c['kinematic_trust_min'],
            c['kinematic_trust_max'],
        )
        if kin_outlier:
            kin_trust = min(kin_trust, 0.35)

        wind_kin_stable = kin_trust * wind_kin + (1.0 - kin_trust) * wind_nn
        return wind_kin_stable, measurement_gap, bool(kin_outlier)

    def connect_mavlink(self):
        """连接到MAVLink"""
        self.logger.info(f"Connecting MAVLink: {self.mavlink_connection}")
        
        try:
            # 串口连接时显式传入波特率，UDP/网络连接走默认参数
            if self.mavlink_connection.startswith('/dev/'):
                self.connection = mavutil.mavlink_connection(self.mavlink_connection, baud=self.baudrate)
            else:
                self.connection = mavutil.mavlink_connection(self.mavlink_connection)

            self.connection.wait_heartbeat()
            self.logger.info(f"✓ MAVLink connected (System ID: {self.connection.target_system})")
            
            # 请求所有数据流
            self.connection.mav.request_data_stream_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_ALL,
                50, 1
            )

            # 逐条请求 45 维特征所需的消息 @ 50 Hz（20000 us）。
            #   33=GLOBAL_POSITION_INT 30=ATTITUDE 74=VFR_HUD 105=HIGHRES_IMU
            #   36=SERVO_OUTPUT_RAW    65=RC_CHANNELS
            #   83=ATTITUDE_TARGET（期望姿态/角速率） 85=POSITION_TARGET_LOCAL_NED（期望地速）
            required_msg_ids = {
                "GLOBAL_POSITION_INT": 33, "ATTITUDE": 30, "VFR_HUD": 74,
                "HIGHRES_IMU": 105, "SERVO_OUTPUT_RAW": 36, "RC_CHANNELS": 65,
                "ATTITUDE_TARGET": 83, "POSITION_TARGET_LOCAL_NED": 85,
            }
            for name, msg_id in required_msg_ids.items():
                self.connection.mav.command_long_send(
                    self.connection.target_system,
                    self.connection.target_component,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                    0, msg_id, 20000, 0, 0, 0, 0, 0,
                )
            self.logger.info(
                f"Data streams requested (50 Hz): {list(required_msg_ids)}")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ MAVLink connection failed: {e}")
            return False
    
    @staticmethod
    def _quat_to_euler(q):
        """MAVLink 四元数 [w, x, y, z] -> (roll, pitch, yaw) rad。"""
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return float(roll), float(pitch), float(yaw)

    def extract_features(self, msg_dict, dt=None):
        """从 MAVLink 消息装配 45 维特征（阶段 2，与训练/离线验证严格一致）。

        Args:
            msg_dict: 本帧 MAVLink 消息字典
            dt:       与上一帧的真实时间间隔 [s]。传入后因果加速度按实测 dt 求导，
                      消除"名义 50Hz 但真机只到 ~29Hz"的加速度尺度误差（方案C）。

        本函数只负责把 MAVLink 消息解析成 30 个原子信号 dict，再交给共用的
        `StreamingFeatureExtractor` 装配成 45 维向量。特征布局见
        `online_feature_extractor.FEATURE_IDX`：
          0-2   NED 地速          | 3-5   机体速度        | 6-8   机体速度加速度(因果)
          9-11  姿态角            | 12-14 机体角速率      | 15-18 指令舵面+油门
          19    空速             | 20-22 期望姿态        | 23-25 姿态误差(actual-target)
          26-28 期望角速率        | 29-31 角速率误差       | 32-34 期望地速
          35-37 地速误差          | 38-41 实际舵面        | 42-44 原始 IMU 加速度

        MAVLink 消息映射：
          GLOBAL_POSITION_INT -> 地速   ATTITUDE -> 姿态/角速率   VFR_HUD -> 空速
          HIGHRES_IMU -> 原始 IMU 加速度(42-44)
          ATTITUDE_TARGET -> 期望姿态/期望角速率
          POSITION_TARGET_LOCAL_NED -> 期望地速
          SERVO_OUTPUT_RAW -> 舵面(指令与实际同源，见下方说明)
          RC_CHANNELS -> 油门指令

        说明：JSBSim 训练数据区分"指令舵面(fcs cmd-norm)"与"实际舵面(actual)"，而
        PX4/HITL 桥接下二者最可靠的公共来源均为 SERVO_OUTPUT_RAW，故 15-17 与 38-40
        近似同源。若后续可取 ACTUATOR_OUTPUT_STATUS/舵面 setpoint，可进一步区分。
        """
        try:
            sig = {}

            gps = msg_dict.get('GLOBAL_POSITION_INT')
            if gps is not None:
                sig['vel_n'] = gps.vx / 100.0   # cm/s -> m/s
                sig['vel_e'] = gps.vy / 100.0
                sig['vel_d'] = gps.vz / 100.0

            att = msg_dict.get('ATTITUDE')
            if att is not None:
                sig['roll'] = att.roll
                sig['pitch'] = att.pitch
                sig['yaw'] = att.yaw
                sig['roll_rate'] = att.rollspeed
                sig['pitch_rate'] = att.pitchspeed
                sig['yaw_rate'] = att.yawspeed

            hud = msg_dict.get('VFR_HUD')
            if hud is not None:
                sig['airspeed'] = hud.airspeed

            # 原始 IMU 机体加速度 -> 42-44（6-8 由提取器基于机体速度因果微分得到）
            imu = msg_dict.get('HIGHRES_IMU')
            if imu is not None:
                sig['imu_ax'] = imu.xacc
                sig['imu_ay'] = imu.yacc
                sig['imu_az'] = imu.zacc

            # 舵面：SERVO_OUTPUT_RAW 归一化（servo1/2/4 -> [-1,1]，servo3 油门 -> [0,1]）
            servo = msg_dict.get('SERVO_OUTPUT_RAW')
            if servo is not None:
                ail = (servo.servo1_raw - 1500.0) / 500.0
                ele = (servo.servo2_raw - 1500.0) / 500.0
                rud = (servo.servo4_raw - 1500.0) / 500.0
                thr = float(np.clip((servo.servo3_raw - 1000.0) / 1000.0, 0.0, 1.0))
                sig['aileron_cmd'] = ail
                sig['elevator_cmd'] = ele
                sig['rudder_cmd'] = rud
                sig['aileron_actual'] = ail
                sig['elevator_actual'] = ele
                sig['rudder_actual'] = rud
                sig['throttle_actual'] = thr

            # 油门指令：RC_CHANNELS chan3 -> [0,1]
            rc = msg_dict.get('RC_CHANNELS')
            if rc is not None:
                sig['throttle_cmd'] = float(np.clip((rc.chan3_raw - 1000.0) / 1000.0, 0.0, 1.0))

            # 期望姿态 / 期望角速率（ATTITUDE_TARGET）
            atgt = msg_dict.get('ATTITUDE_TARGET')
            if atgt is not None:
                t_roll, t_pitch, t_yaw = self._quat_to_euler(atgt.q)
                sig['target_roll'] = t_roll
                sig['target_pitch'] = t_pitch
                sig['target_yaw'] = t_yaw
                sig['target_p'] = atgt.body_roll_rate
                sig['target_q'] = atgt.body_pitch_rate
                sig['target_r'] = atgt.body_yaw_rate

            # 期望地速（POSITION_TARGET_LOCAL_NED，MAV_FRAME_LOCAL_NED: vx=N,vy=E,vz=D）
            ptgt = msg_dict.get('POSITION_TARGET_LOCAL_NED')
            if ptgt is not None:
                sig['target_vn'] = ptgt.vx
                sig['target_ve'] = ptgt.vy
                sig['target_vd'] = ptgt.vz

            feat_full = self.feature_extractor.push(sig, dt=dt)

            # 方案C: 中性填充无法忠实复现的舵面/油门特征（置训练均值 -> 归一化后≈0）
            if (feat_full is not None and self.neutralize_ctrl_features
                    and getattr(self, 'scaler_X', None) is not None):
                mean_ = np.asarray(self.scaler_X.mean_, dtype=np.float32)
                for i in self._neutral_feat_idx:
                    if i < feat_full.shape[0] and i < mean_.shape[0]:
                        feat_full[i] = mean_[i]

            if feat_full is None:
                return None
            return select_model_features(feat_full, self.input_size)

        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    def collect_mavlink_data(self):
        """
        收集MAVLink数据
        
        Returns:
            msg_dict: 包含最新消息的字典，如果失败返回None
        """
        # 必需消息：每帧必须新鲜到达，否则本帧作废
        required_msgs = [
            'GLOBAL_POSITION_INT',
            'ATTITUDE',
            'VFR_HUD',
            'HIGHRES_IMU'  # 加速度必需
        ]
        # 可选消息（期望量/舵面）：可能低频或缺失，采用 last-known-value 缓存，
        # 缺失不阻塞主循环，对应特征退化为提取器默认值。
        fresh = set()

        timeout = 1.0  # 1秒超时
        start_time = time.time()

        while time.time() - start_time < timeout:
            msg = self.connection.recv_match(blocking=False)
            if msg is not None:
                msg_type = msg.get_type()
                if msg_type != 'BAD_DATA':
                    self._latest_msgs[msg_type] = msg
                    self._msg_counts[msg_type] = self._msg_counts.get(msg_type, 0) + 1
                    if msg_type in required_msgs:
                        fresh.add(msg_type)

                # 所有必需消息本帧均已刷新即返回（含缓存的可选消息快照）
                if fresh.issuperset(required_msgs):
                    return dict(self._latest_msgs)

            time.sleep(0.001)  # 短暂休眠避免CPU占用过高

        # 超时
        missing = [msg for msg in required_msgs if msg not in fresh]
        self.logger.warning(f"Data collection timeout, missing: {missing}")
        return None
    
    def run_inference(self):
        """
        执行一次推断
        
        Returns:
            result: 包含风速估计、q_scale/r_scale、angles的字典，失败返回None
        """
        if len(self.data_buffer) < self.sequence_length:
            return None
        
        start_time = time.time()
        
        try:
            # 准备输入序列
            X_seq = np.array(list(self.data_buffer))
            X_seq_normalized = self.scaler_X.transform(X_seq)

            # 诊断: 累计末帧归一化特征的一/二阶矩（理想 mean≈0, std≈1）
            if X_seq_normalized.shape[1] == self.input_size:
                last = X_seq_normalized[-1]
                self._featnorm_sum += last
                self._featnorm_sqsum += last * last
                self._featnorm_n += 1

            out = self.backend.infer(
                X_seq_normalized,
                prev_log_q=self.prev_log_q,
                prev_log_r=self.prev_log_r,
                ema_alpha=self.ema_alpha_params,
                clamp=True
            )

            # 更新平滑状态
            self.prev_log_q = out.get('log_q_scale')
            self.prev_log_r = out.get('log_r_scale')

            # 反归一化 PI-GRU 风速
            wind_estimate_norm = np.asarray(out['wind_estimate'], dtype=np.float32).reshape(-1)[:3]
            wind_mean = self.y_mean[:3]
            wind_std = self.y_std[:3]
            wind_nn = wind_estimate_norm * wind_std + wind_mean

            # 提取其他参数
            q_scale = np.asarray(out.get('q_scale', [1.0, 1.0, 1.0]), dtype=np.float32).reshape(-1)[:3]
            r_scale = np.asarray(out.get('r_scale', [1.0, 1.0, 1.0]), dtype=np.float32).reshape(-1)[:3]
            angles = np.asarray(out.get('angles', [0.0, 0.0, 1.0]), dtype=np.float32).reshape(-1)[:3]
            confidence_raw = out.get('confidence')
            if confidence_raw is not None:
                confidence = float(np.asarray(confidence_raw, dtype=np.float32).reshape(-1)[0])
            else:
                confidence = float(1.0 / (1.0 + np.std(q_scale)))

            q_scale_eff = q_scale.copy()
            r_scale_eff = r_scale.copy()

            # 调试：检查PI-GRU输出是否已含NaN
            if not np.all(np.isfinite(wind_nn)):
                self.logger.warning(f"PI-GRU output contains NaN! raw_norm={wind_estimate_norm}, mean={wind_mean}, std={wind_std}")
                if self.output_filter is not None:
                    self.output_filter.reset()
                return None

            # ===== PIRNN-AKF 融合 =====
            wind_fused = wind_nn.copy()

            # 预热阶段：只用PI-GRU，让输出稳定后再启动AKF
            self.akf_warmup_count += 1
            if self.akf_warmup_count > self.akf_warmup_threshold and self.last_msg_dict is not None:
                try:
                    gps = self.last_msg_dict.get('GLOBAL_POSITION_INT')
                    att = self.last_msg_dict.get('ATTITUDE')
                    hud = self.last_msg_dict.get('VFR_HUD')

                    if gps is not None and att is not None and hud is not None:
                        z = np.array([
                            gps.vx / 100.0,
                            gps.vy / 100.0,
                            gps.vz / 100.0,
                            max(float(hud.airspeed), 0.1)
                        ], dtype=np.float64)

                        maneuver_score = self._compute_maneuver_score(X_seq[-1])

                        wind_kin, _, _ = self.akf.construct_kinematic_wind_measurement(
                            vg_ned=z[:3],
                            tas=float(z[3]),
                            roll=float(att.roll),
                            pitch=float(att.pitch),
                            yaw=float(att.yaw),
                            angles=angles
                        )
                        wind_kin_stable, measurement_gap, kin_outlier = self._stabilize_kinematic_measurement(
                            wind_kin,
                            wind_nn
                        )

                        c = self.akf_constants
                        gap_ratio = np.clip(
                            measurement_gap / c['disagreement_scale'], 0.0, 1.0
                        )
                        q_scale_eff = np.clip(
                            q_scale * (1.0 + c['q_maneuver_gain'] * maneuver_score),
                            0.1,
                            20.0,
                        )
                        r_scale_gain = 1.0 + c['r_disagreement_gain'] * gap_ratio
                        if kin_outlier:
                            r_scale_gain *= c['r_outlier_multiplier']
                        r_scale_eff = np.clip(
                            r_scale
                            * (1.0 + c['r_maneuver_gain'] * maneuver_score)
                            * r_scale_gain,
                            0.1,
                            20.0,
                        )

                        if not self.akf_initialized:
                            self.akf.reset()
                            self.akf.x = 0.80 * wind_nn.astype(np.float64) + 0.20 * wind_kin_stable.astype(np.float64)
                            self.akf_initialized = True
                            self.logger.info("AKF initialized with stabilized kinematic + PI-GRU wind")
                            wind_delta = np.zeros(3, dtype=np.float64)
                        else:
                            wind_delta = wind_nn - self.prev_wind_nn if self.prev_wind_nn is not None else np.zeros(3, dtype=np.float64)

                        self.akf.update_noise_covariance(q_scale_eff, r_scale_eff)
                        self.akf.predict(neural_wind_delta=wind_delta)

                        if np.any(np.isnan(self.akf.P)) or np.any(np.diag(self.akf.P) > 1e4):
                            self.logger.warning("AKF covariance diverged, resetting filter")
                            self.akf.reset()
                            self.akf.x = 0.80 * wind_nn.astype(np.float64) + 0.20 * wind_kin_stable.astype(np.float64)
                            self.akf_initialized = True
                        else:
                            self.akf.update(
                                z,
                                roll=float(att.roll),
                                pitch=float(att.pitch),
                                yaw=float(att.yaw),
                                confidence=confidence,
                                angles=angles,
                                nn_measurement=wind_nn,
                                maneuver_score=maneuver_score,
                                wind_kin_override=wind_kin_stable
                            )
                            wind_akf = self.akf.get_wind_estimate()
                            diagnostics = self.akf.get_diagnostics()

                            if np.all(np.isfinite(wind_akf)) and np.linalg.norm(wind_akf) < self.max_wind_speed * 2:
                                p_ratio = np.clip(np.mean(diagnostics['P_diag']) / 2.0, 0.0, 1.0)
                                akf_weight = np.clip(
                                    c['fusion_base']
                                    + c['fusion_confidence_gain'] * confidence
                                    - c['fusion_disagreement_gain'] * gap_ratio
                                    - c['fusion_covariance_gain'] * p_ratio,
                                    c['fusion_min'],
                                    c['fusion_max'],
                                )
                                if kin_outlier:
                                    akf_weight = min(
                                        akf_weight, c['fusion_outlier_cap']
                                    )
                                nn_weight = 1.0 - akf_weight
                                wind_fused = akf_weight * wind_akf + nn_weight * wind_nn

                except Exception as e:
                    self.logger.debug(f"AKF fusion error, using PI-GRU output: {e}")

            if self.output_filter is not None:
                wind_fused = self.output_filter.update(wind_fused)

            wind_estimate = wind_fused.astype(np.float32)
            
            # 验证
            inference_time = time.time() - start_time
            if not self.validate_estimate(wind_estimate, inference_time):
                self.performance['invalid_estimates'] += 1
                return None
            
            # 更新缓存
            self.last_wind_estimate = wind_estimate
            self.last_q_scale = q_scale_eff
            self.last_r_scale = r_scale_eff
            self.last_angles = angles
            self.prev_wind_nn = wind_nn.copy()
            
            # 更新性能统计
            self.performance['inference_count'] += 1
            self.performance['total_inference_time'] += inference_time
            self.performance['max_inference_time'] = max(
                self.performance['max_inference_time'], inference_time
            )
            
            return {
                'wind_estimate': wind_estimate,
                'wind_nn': wind_nn.astype(np.float32),   # 诊断: PI-GRU 原始输出(融合前)
                'q_scale': q_scale_eff,
                'r_scale': r_scale_eff,
                'angles': angles,
                'confidence': confidence,
                'inference_time': inference_time
            }
        
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            return None
    
    def validate_estimate(self, wind_estimate, inference_time):
        """
        验证估计结果
        
        Args:
            wind_estimate: [3] 风速估计
            inference_time: 推理时间（秒）
        
        Returns:
            bool: 是否有效
        """
        # 检查NaN或Inf
        if not np.all(np.isfinite(wind_estimate)):
            self.logger.warning("Wind estimate contains NaN or Inf")
            return False
        
        # 检查风速大小
        wind_magnitude = np.linalg.norm(wind_estimate)
        if wind_magnitude > self.max_wind_speed:
            self.logger.warning(f"Wind speed too high: {wind_magnitude:.2f} m/s > {self.max_wind_speed} m/s")
            return False
        
        # 检查推理时间
        if inference_time > self.max_inference_time:
            self.logger.warning(f"Inference too slow: {inference_time*1000:.2f} ms > {self.max_inference_time*1000:.2f} ms")
            return False
        
        return True
    
    def send_to_px4(self, result):
        """
        将结果发送给PX4（扩展版，包含q_scale/r_scale和小角修正）
        
        Args:
            result: 推理结果字典
        """
        wind_estimate = result['wind_estimate']
        q_scale = result['q_scale']
        r_scale = result['r_scale']
        angles = result['angles']
        
        timestamp = int((time.time() - self.performance['start_time']) * 1000) & 0xFFFFFFFF
        
        try:
            # 统一计算风速派生量
            wind_magnitude = np.linalg.norm(wind_estimate[:2])
            wind_direction = np.arctan2(wind_estimate[1], wind_estimate[0]) * 180 / np.pi

            # ===== 标准MAVLink风消息（推荐用于飞控接管） =====
            if self.send_wind_cov:
                self.connection.mav.wind_cov_send(
                    int(time.time() * 1_000_000),
                    float(wind_estimate[0]),
                    float(wind_estimate[1]),
                    float(wind_estimate[2]),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0
                )

            # ===== NAMED_VALUE_FLOAT（调试与日志可视化） =====
            if self.send_named_values:
                self.connection.mav.named_value_float_send(
                    timestamp, b'WIND_N', float(wind_estimate[0])
                )
                self.connection.mav.named_value_float_send(
                    timestamp, b'WIND_E', float(wind_estimate[1])
                )
                self.connection.mav.named_value_float_send(
                    timestamp, b'WIND_D', float(wind_estimate[2])
                )
                self.connection.mav.named_value_float_send(
                    timestamp, b'WIND_MAG', float(wind_magnitude)
                )
                self.connection.mav.named_value_float_send(
                    timestamp, b'WIND_DIR', float(wind_direction)
                )

                # ===== 过程噪声 q_scale (N/E/D) =====
                self.connection.mav.named_value_float_send(
                    timestamp, b'QSCALE_N', float(q_scale[0])
                )
                self.connection.mav.named_value_float_send(
                    timestamp, b'QSCALE_E', float(q_scale[1])
                )
                self.connection.mav.named_value_float_send(
                    timestamp, b'QSCALE_D', float(q_scale[2])
                )

                # ===== 量测噪声 r_scale (GPS/TAS/ATT) =====
                self.connection.mav.named_value_float_send(
                    timestamp, b'RSCALE_GPS', float(r_scale[0])
                )
                self.connection.mav.named_value_float_send(
                    timestamp, b'RSCALE_TAS', float(r_scale[1])
                )
                self.connection.mav.named_value_float_send(
                    timestamp, b'RSCALE_ATT', float(r_scale[2])
                )

                # ===== 小角修正 =====
                d_alpha_deg = angles[0] * 180 / np.pi
                d_beta_deg = angles[1] * 180 / np.pi
                s_tas = angles[2]

                self.connection.mav.named_value_float_send(
                    timestamp, b'D_ALPHA', float(d_alpha_deg)
                )
                self.connection.mav.named_value_float_send(
                    timestamp, b'D_BETA', float(d_beta_deg)
                )
                self.connection.mav.named_value_float_send(
                    timestamp, b'S_TAS', float(s_tas)
                )

        except Exception as e:
            self.logger.error(f"MAVLink send failed: {e}")
    
    def print_status(self, result):
        """
        打印状态信息
        
        Args:
            result: 推理结果字典
        """
        wind = result['wind_estimate']
        q_scale = result['q_scale']
        r_scale = result['r_scale']
        angles = result['angles']
        
        wind_mag = np.linalg.norm(wind)
        wind_dir = np.arctan2(wind[1], wind[0]) * 180 / np.pi
        
        runtime = time.time() - self.performance['start_time']
        avg_time = (self.performance['total_inference_time'] / 
                   self.performance['inference_count'] * 1000) if self.performance['inference_count'] > 0 else 0
        
        print(f"\rRuntime: {runtime:.1f}s | "
              f"Infer: {self.performance['inference_count']} | "
              f"Wind: [{wind[0]:5.2f}, {wind[1]:5.2f}, {wind[2]:5.2f}] m/s | "
              f"Mag: {wind_mag:5.2f} m/s | "
              f"Dir: {wind_dir:6.1f}° | "
              f"q_scale: [{q_scale[0]:.2f}, {q_scale[1]:.2f}, {q_scale[2]:.2f}] | "
              f"r_scale: [{r_scale[0]:.2f}, {r_scale[1]:.2f}, {r_scale[2]:.2f}] | "
              f"Latency: {result['inference_time']*1000:.1f}/{avg_time:.1f} ms", 
              end='', flush=True)
    
    def print_statistics(self):
        """打印运行统计"""
        print("\n" + "="*70)
        print("  Statistics")
        print("="*70)
        
        runtime = time.time() - self.performance['start_time']
        count = self.performance['inference_count']
        
        print(f"Runtime: {runtime:.2f} s")
        print(f"Inferences: {count}")
        print(f"Inference rate: {count/runtime:.2f} Hz")
        print(f"Avg latency: {self.performance['total_inference_time']/count*1000:.2f} ms")
        print(f"Max latency: {self.performance['max_inference_time']*1000:.2f} ms")
        print(f"Invalid estimates: {self.performance['invalid_estimates']}")
        
        if self.last_wind_estimate is not None:
            print(f"\nLast wind estimate: [{self.last_wind_estimate[0]:.2f}, "
                  f"{self.last_wind_estimate[1]:.2f}, {self.last_wind_estimate[2]:.2f}] m/s")
            print(f"Last q_scale: [{self.last_q_scale[0]:.2f}, {self.last_q_scale[1]:.2f}, {self.last_q_scale[2]:.2f}]")
            print(f"Last r_scale: [{self.last_r_scale[0]:.2f}, {self.last_r_scale[1]:.2f}, {self.last_r_scale[2]:.2f}]")
            print(f"Last angle corrections: da={self.last_angles[0]*180/np.pi:.2f} deg, "
                  f"db={self.last_angles[1]*180/np.pi:.2f} deg, s_tas={self.last_angles[2]:.4f}")

        # 方案C: 消息到达统计（确认期望量/舵面是否长期缺失退化为默认值）
        if self._msg_counts:
            print("\nMAVLink message arrival (Hz over run):")
            watch = ['GLOBAL_POSITION_INT', 'ATTITUDE', 'VFR_HUD', 'HIGHRES_IMU',
                     'SERVO_OUTPUT_RAW', 'RC_CHANNELS', 'ATTITUDE_TARGET',
                     'POSITION_TARGET_LOCAL_NED']
            for name in watch:
                c = self._msg_counts.get(name, 0)
                flag = '' if c > 0 else '  <-- MISSING (feature defaults used!)'
                print(f"  {name:28s}: {c:6d}  ({c/runtime:5.1f} Hz){flag}")

        # 诊断: 归一化特征偏离训练分布的通道（|mean|>0.5 或 std∉[0.5,2]）
        if self._featnorm_n > 5:
            from online_feature_extractor import FEATURE_IDX as _FI
            full_names = [n for n, _ in sorted(_FI.items(), key=lambda kv: kv[1])]
            names = [full_names[i] for i in self.feature_keep_indices]
            mean = self._featnorm_sum / self._featnorm_n
            var = self._featnorm_sqsum / self._featnorm_n - mean * mean
            std = np.sqrt(np.clip(var, 0, None))
            bad = [(i, mean[i], std[i]) for i in range(self.input_size)
                   if abs(mean[i]) > 0.5 or std[i] < 0.5 or std[i] > 2.0]
            bad.sort(key=lambda t: -abs(t[1]))
            print(f"\n归一化特征偏离训练分布 (n={self._featnorm_n}, 理想 mean~0 std~1):")
            if bad:
                for i, mn, sd in bad:
                    print(f"  [{i:2d}] {names[i]:14s} mean={mn:+.2f}  std={sd:.2f}")
            else:
                print("  (无明显偏离——所有通道 mean~0 std~1)")

        print("="*70)
    
    def run(self, phase='steady', duration=None):
        """主运行循环

        Args:
            phase: 当前扰动阶段标签，写入 CSV 用于图14分组。
                   可在外部通过修改 estimator.current_phase 动态切换，
                   或直接以命令行参数传入。
                   取值建议: 'steady' | 'gust_light' | 'gust_strong' | 'packet_loss'
            duration: 运行时长 [s]；None 表示一直运行至 Ctrl+C（供 SITL 编排自动终止）。
        """
        self.current_phase = phase
        print("="*70)
        print(" PIRNN-AKF在线风速估计器 v3.0")
        print("="*70)
        
        # 连接MAVLink
        if not self.connect_mavlink():
            self.logger.error("Cannot connect MAVLink, exiting")
            return
        
        print("\nStarting real-time inference...")
        print("Press Ctrl+C to stop\n")
        
        inference_interval = 1.0 / self.inference_rate
        run_start = time.time()
        
        try:
            while True:
                if duration is not None and (time.time() - run_start) >= duration:
                    self.logger.info(f"Reached duration {duration}s, stopping")
                    break
                loop_start = time.time()
                
                # 收集数据，收包完成后立即记录 t_recv_ns（用于计算通信+调度时延）
                msg_dict = self.collect_mavlink_data()
                t_recv_ns = time.monotonic_ns()
                if msg_dict is None:
                    continue

                # 方案C: 估计与上一有效帧的真实间隔 dt，供因果加速度按实测速率求导
                now_t = t_recv_ns / 1e9
                frame_dt = (now_t - self._last_frame_t) if self._last_frame_t is not None else None
                self._last_frame_t = now_t

                # 提取特征
                features = self.extract_features(msg_dict, dt=frame_dt)
                if features is None:
                    continue
                
                # 添加到缓冲区
                self.data_buffer.append(features)
                self.last_msg_dict = msg_dict
                
                # 执行推理
                result = self.run_inference()
                if result is not None:
                    # 发送结果
                    self.send_to_px4(result)

                    # 记录CSV（传入 t_recv_ns 和当前 phase）
                    self._log_csv(result, msg_dict,
                                  t_recv_ns=t_recv_ns,
                                  phase=getattr(self, 'current_phase', 'steady'))

                    # 打印状态
                    self.print_status(result)
                
                # 控制推理频率
                elapsed = time.time() - loop_start
                if elapsed < inference_interval:
                    time.sleep(inference_interval - elapsed)
        
        except KeyboardInterrupt:
            print("\n\nStop signal received")
        
        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 打印统计
            self.print_statistics()

            if self.csv_file is not None:
                self.csv_file.close()
                self.logger.info("CSV data file saved")

            if self.backend is not None:
                self.backend.close()
            
            # 关闭连接
            if self.connection is not None:
                self.connection.close()
                self.logger.info("MAVLink connection closed")
            
            self.logger.info("Program exited")


if __name__ == "__main__":
    import argparse
    print("="*70)
    print(" PIRNN-AKF Online Wind Estimator v3.0 (EKF Fusion Enhanced)")
    print("="*70)

    parser = argparse.ArgumentParser(description='PIRNN-AKF online wind estimator')
    parser.add_argument('--phase', type=str, default='steady',
                        choices=['steady', 'gust_light', 'gust_strong', 'packet_loss'],
                        help='HIL 实验扰动阶段标签，写入 CSV 供图14分组（默认: steady）')
    parser.add_argument('--config', type=str, default=None,
                        help='config.yaml 路径（默认自动查找）')
    parser.add_argument('--duration', type=float, default=None,
                        help='运行时长 [s]；缺省一直运行至 Ctrl+C')
    args = parser.parse_args()

    try:
        estimator = OnlineWindEstimator(config_path=args.config)
        estimator.run(phase=args.phase, duration=args.duration)

    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)