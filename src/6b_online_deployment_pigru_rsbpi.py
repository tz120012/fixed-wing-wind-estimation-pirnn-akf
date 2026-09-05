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
        
        # EMA配置
        self.ema_alpha_wind = deploy_config.get('ema_alpha_wind', 0.2)
        self.ema_alpha_params = deploy_config.get('ema_alpha_params', 0.1)  # q_scale/r_scale的EMA系数
        
        # 安全限制
        self.max_wind_speed = deploy_config.get('max_wind_speed', 20.0)
        self.max_inference_time = deploy_config.get('max_inference_time_ms', 50.0) / 1000.0
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=self.sequence_length)
        
        # 风速输出滤波器
        self.output_filter = ExponentialMovingAverageFilter(
            alpha=self.ema_alpha_wind
        ) if deploy_config.get('enable_output_filter', True) else None
        
        # 模型状态（用于q_scale/r_scale平滑）
        self.prev_log_q = None
        self.prev_log_r = None
        self.backend = None
        self.device = 'unknown'
        
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

        # boot_time_us 复用 t_fc_send_us
        boot_time_us = t_fc_send_us

        # ── 时延计算 ──
        t_recv_us   = int(t_recv_ns / 1000) if t_recv_ns is not None else int(now_ns / 1000)
        t_output_us = int(now_ns / 1000)
        # 通信时延：需要飞控与伴机时钟同步（PTP/MAVLink TIMESYNC）才有绝对意义；
        # 未同步时记为 -1 以明确标识
        if t_fc_send_us > 0 and t_recv_us > 0:
            latency_comm_ms = (t_recv_us - t_fc_send_us) / 1000.0
            # 防止时钟未同步时出现异常大负值
            if latency_comm_ms < -50.0 or latency_comm_ms > 500.0:
                latency_comm_ms = -1.0
        else:
            latency_comm_ms = -1.0
        # 端到端时延 = 从收包到推理输出完成
        latency_e2e_ms = (t_output_us - t_recv_us) / 1000.0

        # ── 真值风速（JSBSim 通过 WIND 消息广播，SITL 环境可用）──
        wind_gt = [float('nan'), float('nan'), float('nan')]
        if 'WIND' in msg_dict:
            wm = msg_dict['WIND']
            # MAVLink WIND: direction(deg from N), speed(m/s), speed_z(m/s)
            direction_rad = float(getattr(wm, 'direction', 0)) * np.pi / 180.0
            speed    = float(getattr(wm, 'speed',   0))
            speed_z  = float(getattr(wm, 'speed_z', 0))
            wind_gt[0] =  speed * np.cos(direction_rad)   # North
            wind_gt[1] =  speed * np.sin(direction_rad)   # East
            wind_gt[2] = -speed_z                          # Down（MAVLink 正值向上，取反）

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
        
        # 提取风速的归一化参数（用于反归一化）
        self.y_mean = np.asarray(self.scaler_y.mean_, dtype=np.float32)
        self.y_std = np.asarray(self.scaler_y.scale_, dtype=np.float32)
        
        self.logger.info(f"Input dim: {metadata.get('input_size', 'Unknown')}")
        self.logger.info(f"Output dim: {metadata.get('output_size', 'Unknown')}")
    
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
            
            # 单独请求ATTITUDE消息 (30=ATTITUDE, 20000us=50Hz)
            self.connection.mav.command_long_send(
                self.connection.target_system,
                self.connection.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                30,      # ATTITUDE message ID
                20000,   # 20000 us = 50 Hz
                0, 0, 0, 0, 0
            )
            self.logger.info("Data streams requested (50 Hz)")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ MAVLink connection failed: {e}")
            return False
    
    def extract_features(self, msg_dict):
        """
        从MAVLink消息中提取特征
        
        特征顺序（20维）:
          0-2:   vel_n, vel_e, vel_d (GPS地速)
          3-5:   vel_x_body, vel_y_body, vel_z_body (机体速度)
          6-8:   acc_x, acc_y, acc_z (加速度)
          9-11:  roll, pitch, yaw (姿态角)
          12-14: gyro_x, gyro_y, gyro_z (角速度)
          15-17: aileron, elevator, rudder (舵面)
          18:    throttle (油门)
          19:    airspeed (空速)
        
        Args:
            msg_dict: 包含必要MAVLink消息的字典
        
        Returns:
            features: [20] numpy array
        """
        features = np.zeros(20, dtype=np.float32)
        
        try:
            # GPS速度 (NED)
            if 'GLOBAL_POSITION_INT' in msg_dict:
                gps = msg_dict['GLOBAL_POSITION_INT']
                features[0] = gps.vx / 100.0  # cm/s -> m/s
                features[1] = gps.vy / 100.0
                features[2] = gps.vz / 100.0
            
            # 机体速度（从NED速度和姿态计算）
            if 'GLOBAL_POSITION_INT' in msg_dict and 'ATTITUDE' in msg_dict:
                # NED速度已在features[0:2]填充
                vel_n, vel_e, vel_d = features[0], features[1], features[2]
                att = msg_dict['ATTITUDE']
                roll, pitch, yaw = att.roll, att.pitch, att.yaw
                
                # NED -> Body 旋转
                cos_roll, sin_roll = np.cos(roll), np.sin(roll)
                cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
                cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
                
                features[3] = (cos_pitch * cos_yaw) * vel_n + \
                             (cos_pitch * sin_yaw) * vel_e + \
                             (-sin_pitch) * vel_d
                
                features[4] = (sin_roll * sin_pitch * cos_yaw - cos_roll * sin_yaw) * vel_n + \
                             (sin_roll * sin_pitch * sin_yaw + cos_roll * cos_yaw) * vel_e + \
                             (sin_roll * cos_pitch) * vel_d
                
                features[5] = (cos_roll * sin_pitch * cos_yaw + sin_roll * sin_yaw) * vel_n + \
                             (cos_roll * sin_pitch * sin_yaw - sin_roll * cos_yaw) * vel_e + \
                             (cos_roll * cos_pitch) * vel_d
            
            # 加速度 (HIGHRES_IMU 单位已经是 m/s²)
            if 'HIGHRES_IMU' in msg_dict:
                imu = msg_dict['HIGHRES_IMU']
                features[6] = imu.xacc
                features[7] = imu.yacc
                features[8] = imu.zacc
            
            # 姿态角
            if 'ATTITUDE' in msg_dict:
                att = msg_dict['ATTITUDE']
                features[9] = att.roll
                features[10] = att.pitch
                features[11] = att.yaw
            
            # 角速度
            if 'ATTITUDE' in msg_dict:
                att = msg_dict['ATTITUDE']
                features[12] = att.rollspeed
                features[13] = att.pitchspeed
                features[14] = att.yawspeed
            
            # 舵面（从SERVO_OUTPUT_RAW）
            if 'SERVO_OUTPUT_RAW' in msg_dict:
                servo = msg_dict['SERVO_OUTPUT_RAW']
                # 归一化到 [-1, 1]
                features[15] = (servo.servo1_raw - 1500) / 500.0  # aileron
                features[16] = (servo.servo2_raw - 1500) / 500.0  # elevator
                features[17] = (servo.servo4_raw - 1500) / 500.0  # rudder
            
            # 油门（从RC_CHANNELS，正确归一化到[0,1]）
            if 'RC_CHANNELS' in msg_dict:
                rc = msg_dict['RC_CHANNELS']
                features[18] = np.clip((rc.chan3_raw - 1000.0) / 1000.0, 0.0, 1.0)
            
            # 空速
            if 'VFR_HUD' in msg_dict:
                hud = msg_dict['VFR_HUD']
                features[19] = hud.airspeed
            
            return features
        
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None
    
    def collect_mavlink_data(self):
        """
        收集MAVLink数据
        
        Returns:
            msg_dict: 包含最新消息的字典，如果失败返回None
        """
        msg_dict = {}
        required_msgs = [
            'GLOBAL_POSITION_INT',
            'ATTITUDE',
            'VFR_HUD',
            'HIGHRES_IMU'  # 加速度必需
        ]
        
        timeout = 1.0  # 1秒超时
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            msg = self.connection.recv_match(blocking=False)
            if msg is not None:
                msg_type = msg.get_type()
                if msg_type != 'BAD_DATA':
                    msg_dict[msg_type] = msg
                
                # 检查是否收集到所有必需消息
                if all(msg_type in msg_dict for msg_type in required_msgs):
                    return msg_dict
            
            time.sleep(0.001)  # 短暂休眠避免CPU占用过高
        
        # 超时
        missing = [msg for msg in required_msgs if msg not in msg_dict]
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

            # 反归一化风速
            wind_estimate_norm = np.asarray(out['wind_estimate'], dtype=np.float32)
            wind_mean = self.y_mean[:3]
            wind_std = self.y_std[:3]
            wind_estimate = wind_estimate_norm * wind_std + wind_mean
            
            # 提取其他参数
            q_scale = np.asarray(out.get('q_scale', [1.0, 1.0, 1.0]), dtype=np.float32)
            r_scale = np.asarray(out.get('r_scale', [1.0, 1.0, 1.0]), dtype=np.float32)
            angles = np.asarray(out.get('angles', [0.0, 0.0, 1.0]), dtype=np.float32)
            
            # 输出滤波（仅对风速，q_scale/r_scale已在模型内部平滑）
            if self.output_filter is not None:
                wind_estimate = self.output_filter.update(wind_estimate)
            
            # 验证
            inference_time = time.time() - start_time
            if not self.validate_estimate(wind_estimate, inference_time):
                self.performance['invalid_estimates'] += 1
                return None
            
            # 更新缓存
            self.last_wind_estimate = wind_estimate
            self.last_q_scale = q_scale
            self.last_r_scale = r_scale
            self.last_angles = angles
            
            # 更新性能统计
            self.performance['inference_count'] += 1
            self.performance['total_inference_time'] += inference_time
            self.performance['max_inference_time'] = max(
                self.performance['max_inference_time'], inference_time
            )
            
            return {
                'wind_estimate': wind_estimate,
                'q_scale': q_scale,
                'r_scale': r_scale,
                'angles': angles,
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
        
        print("="*70)
    
    def run(self, phase='steady'):
        """主运行循环

        Args:
            phase: 当前扰动阶段标签，写入 CSV 用于图14分组。
                   可在外部通过修改 estimator.current_phase 动态切换，
                   或直接以命令行参数传入。
                   取值建议: 'steady' | 'gust_light' | 'gust_strong' | 'packet_loss'
        """
        self.current_phase = phase
        print("="*70)
        print(" PI-GRU Online Wind Estimator v3.0")
        print("="*70)
        
        # 连接MAVLink
        if not self.connect_mavlink():
            self.logger.error("Cannot connect MAVLink, exiting")
            return
        
        print("\nStarting real-time inference...")
        print("Press Ctrl+C to stop\n")
        
        inference_interval = 1.0 / self.inference_rate
        
        try:
            while True:
                loop_start = time.time()
                
                # 收集数据，收包完成后立即记录 t_recv_ns（用于计算通信+调度时延）
                msg_dict = self.collect_mavlink_data()
                t_recv_ns = time.monotonic_ns()
                if msg_dict is None:
                    continue
                
                # 提取特征
                features = self.extract_features(msg_dict)
                if features is None:
                    continue
                
                # 添加到缓冲区
                self.data_buffer.append(features)
                
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
    print(" PI-GRU Online Wind Estimator v3.0 (EKF Fusion Enhanced)")
    print("="*70)

    parser = argparse.ArgumentParser(description='PI-GRU online wind estimator')
    parser.add_argument('--phase', type=str, default='steady',
                        choices=['steady', 'gust_light', 'gust_strong', 'packet_loss'],
                        help='HIL 实验扰动阶段标签，写入 CSV 供图14分组（默认: steady）')
    parser.add_argument('--config', type=str, default=None,
                        help='config.yaml 路径（默认自动查找）')
    args = parser.parse_args()

    try:
        estimator = OnlineWindEstimator(config_path=args.config)
        estimator.run(phase=args.phase)

    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)