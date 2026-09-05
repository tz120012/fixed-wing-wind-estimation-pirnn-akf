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
import csv
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
    
    def __init__(self, config_path='../config/config.yaml'):
        """
        初始化在线估计器
        
        Args:
            config_path: 配置文件路径
        """
        # 解析项目根目录（以配置文件位置为准）
        # 相对路径统一按当前脚本目录解析，避免受启动cwd影响
        if not os.path.isabs(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.normpath(os.path.join(script_dir, config_path))
        config_path = os.path.abspath(config_path)
        self.project_root = os.path.dirname(os.path.dirname(config_path))

        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 部署配置
        deploy_config = self.config.get('deployment', {})
        self.mode = str(deploy_config.get('mode', 'deploy')).lower()  # deploy | sitl_eval
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

        # SITL评估模式配置
        self.eval_duration_s = float(deploy_config.get('eval_duration_s', 120.0))
        self.eval_max_samples = int(deploy_config.get('eval_max_samples', 0))  # 0 表示不限制
        self.eval_send_outputs = bool(deploy_config.get('eval_send_outputs', False))
        self.eval_csv_path = deploy_config.get('eval_csv_path', '')
        
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
        self.last_msg_dict = None

        # SITL评估记录
        self.eval_csv_file = None
        self.eval_csv_writer = None
        self.eval_rows = 0
        
        # 日志设置
        self.setup_logging()

        if self.mode == 'sitl_eval':
            self.init_eval_csv()
        
        # 加载后端与归一化参数
        self.load_backend()
        self.load_normalization_params()
        
        # MAVLink连接
        self.connection = None
        
        self.logger.info("="*70)
        self.logger.info("在线风速估计器初始化完成")
        self.logger.info("="*70)
        self.logger.info(f"模式: {self.mode}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"后端: {self.backend_name}")
        self.logger.info(f"推理频率: {self.inference_rate} Hz")
        self.logger.info(f"序列长度: {self.sequence_length}")
        self.logger.info(f"EMA系数 (风速): {self.ema_alpha_wind}")
        self.logger.info(f"EMA系数 (q_scale/r_scale): {self.ema_alpha_params}")
        self.logger.info(f"最大风速: {self.max_wind_speed} m/s")
        if self.mode == 'sitl_eval':
            self.logger.info(f"评估时长上限: {self.eval_duration_s:.1f} s")
            self.logger.info(f"评估样本上限: {self.eval_max_samples if self.eval_max_samples > 0 else '不限'}")
            self.logger.info(f"评估模式发送输出: {self.eval_send_outputs}")
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
        self.logger.info(f"日志文件: {log_file}")

    def init_eval_csv(self):
        """初始化SITL评估CSV记录器"""
        default_dir_cfg = self.config.get('logging', {}).get('save_dir', '../logs/')
        default_dir = self._resolve_project_path(default_dir_cfg, '../logs/')
        os.makedirs(default_dir, exist_ok=True)

        if self.eval_csv_path:
            csv_path = self._resolve_project_path(self.eval_csv_path, '../logs/sitl_eval.csv')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        else:
            csv_path = os.path.join(default_dir, f'sitl_eval_{time.strftime("%Y%m%d_%H%M%S")}.csv')

        self.eval_csv_file = open(csv_path, 'w', newline='')
        self.eval_csv_writer = csv.writer(self.eval_csv_file)
        self.eval_csv_writer.writerow([
            'wall_time_s',
            'runtime_s',
            'boot_time_us',
            'wind_n', 'wind_e', 'wind_d',
            'wind_mag', 'wind_dir_deg',
            'q_scale_n', 'q_scale_e', 'q_scale_d',
            'r_scale_gps', 'r_scale_tas', 'r_scale_att',
            'd_alpha_deg', 'd_beta_deg', 's_tas',
            'ekf2_wind_n', 'ekf2_wind_e', 'ekf2_wind_d',
            'inference_ms'
        ])
        self.logger.info(f"SITL评估CSV: {csv_path}")

    def log_eval_row(self, result, msg_dict):
        """写入一行SITL评估数据"""
        if self.eval_csv_writer is None:
            return

        boot_time_us = 0
        if 'HIGHRES_IMU' in msg_dict:
            boot_time_us = int(getattr(msg_dict['HIGHRES_IMU'], 'time_usec', 0) or 0)
        elif 'GLOBAL_POSITION_INT' in msg_dict:
            boot_time_us = int(getattr(msg_dict['GLOBAL_POSITION_INT'], 'time_boot_ms', 0) * 1000)

        ekf_n = np.nan
        ekf_e = np.nan
        ekf_d = np.nan
        wind_cov_msg = msg_dict.get('WIND_COV')
        if wind_cov_msg is not None:
            ekf_n = float(getattr(wind_cov_msg, 'wind_x', np.nan))
            ekf_e = float(getattr(wind_cov_msg, 'wind_y', np.nan))
            ekf_d = float(getattr(wind_cov_msg, 'wind_z', np.nan))

        w = result['wind_estimate']
        a = result['q_scale']
        b = result['r_scale']
        ang = result['angles']

        wind_mag = float(np.linalg.norm(w[:2]))
        wind_dir_deg = float(np.arctan2(w[1], w[0]) * 180 / np.pi)
        runtime_s = time.time() - self.performance['start_time']

        self.eval_csv_writer.writerow([
            f"{time.time():.3f}",
            f"{runtime_s:.3f}",
            boot_time_us,
            f"{float(w[0]):.5f}", f"{float(w[1]):.5f}", f"{float(w[2]):.5f}",
            f"{wind_mag:.5f}", f"{wind_dir_deg:.3f}",
            f"{float(a[0]):.5f}", f"{float(a[1]):.5f}", f"{float(a[2]):.5f}",
            f"{float(b[0]):.5f}", f"{float(b[1]):.5f}", f"{float(b[2]):.5f}",
            f"{float(ang[0] * 180 / np.pi):.5f}",
            f"{float(ang[1] * 180 / np.pi):.5f}",
            f"{float(ang[2]):.5f}",
            f"{ekf_n:.5f}", f"{ekf_e:.5f}", f"{ekf_d:.5f}",
            f"{result['inference_time'] * 1000:.3f}"
        ])
        self.eval_csv_file.flush()
        self.eval_rows += 1
    
    def load_backend(self):
        """加载推理后端"""
        self.backend = create_backend(self.config)
        backend_info = self.backend.load()
        self.device = self.backend.device_name

        model_path = backend_info.get('model_path')
        if model_path:
            self.logger.info(f"加载模型: {model_path}")

        model_info = backend_info.get('model_info')
        if model_info:
            self.logger.info(f"模型参数量: {model_info.get('total_params', 0):,}")
            self.logger.info(f"隐藏层维度: {model_info.get('hidden_size', 'Unknown')}")
            self.logger.info(f"GRU层数: {model_info.get('num_layers', 'Unknown')}")

        if backend_info.get('checkpoint_epoch') is not None:
            self.logger.info(f"训练轮数: {backend_info['checkpoint_epoch']}")
        if backend_info.get('best_val_loss') is not None:
            self.logger.info(f"最佳验证损失: {backend_info['best_val_loss']:.4f}")
    
    def load_normalization_params(self):
        """加载归一化参数"""
        norm_path = os.path.join(
            self.config['training']['model_save_path'],
            'norm_params.pkl'
        )
        
        if not os.path.exists(norm_path):
            raise FileNotFoundError(f"归一化参数文件不存在: {norm_path}")
        
        self.logger.info(f"加载归一化参数: {norm_path}")
        
        with open(norm_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']
        
        # 提取风速的归一化参数（用于反归一化）
        self.y_mean = np.asarray(self.scaler_y.mean_, dtype=np.float32)
        self.y_std = np.asarray(self.scaler_y.scale_, dtype=np.float32)
        
        self.logger.info(f"输入维度: {metadata.get('input_size', 'Unknown')}")
        self.logger.info(f"输出维度: {metadata.get('output_size', 'Unknown')}")
    
    def connect_mavlink(self):
        """连接到MAVLink"""
        self.logger.info(f"连接MAVLink: {self.mavlink_connection}")
        
        try:
            # 串口连接时显式传入波特率，UDP/网络连接走默认参数
            if self.mavlink_connection.startswith('/dev/'):
                self.connection = mavutil.mavlink_connection(self.mavlink_connection, baud=self.baudrate)
            else:
                self.connection = mavutil.mavlink_connection(self.mavlink_connection)

            self.connection.wait_heartbeat()
            self.logger.info(f"✓ MAVLink连接成功 (系统ID: {self.connection.target_system})")
            return True
        except Exception as e:
            self.logger.error(f"❌ MAVLink连接失败: {e}")
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
            
            # 加速度 (milli-g -> m/s²)
            if 'SCALED_IMU' in msg_dict:
                imu = msg_dict['SCALED_IMU']
                features[6] = imu.xacc * 9.80665 / 1000.0  # milli-g -> m/s²
                features[7] = imu.yacc * 9.80665 / 1000.0
                features[8] = imu.zacc * 9.80665 / 1000.0
            
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
            self.logger.error(f"特征提取失败: {e}")
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
            'SCALED_IMU'  # 加速度必需
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
        self.logger.warning(f"数据收集超时，缺少: {missing}")
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
            self.logger.error(f"推理失败: {e}")
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
            self.logger.warning("风速估计包含NaN或Inf")
            return False
        
        # 检查风速大小
        wind_magnitude = np.linalg.norm(wind_estimate)
        if wind_magnitude > self.max_wind_speed:
            self.logger.warning(f"风速过大: {wind_magnitude:.2f} m/s > {self.max_wind_speed} m/s")
            return False
        
        # 检查推理时间
        if inference_time > self.max_inference_time:
            self.logger.warning(f"推理时间过长: {inference_time*1000:.2f} ms > {self.max_inference_time*1000:.2f} ms")
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
        
        timestamp = int(time.time() * 1000)
        
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
            self.logger.error(f"发送MAVLink数据失败: {e}")
    
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
        
        print(f"\r运行时间: {runtime:.1f}s | "
              f"推理: {self.performance['inference_count']} | "
              f"风速: [{wind[0]:5.2f}, {wind[1]:5.2f}, {wind[2]:5.2f}] m/s | "
              f"大小: {wind_mag:5.2f} m/s | "
              f"方向: {wind_dir:6.1f}° | "
              f"q_scale: [{q_scale[0]:.2f}, {q_scale[1]:.2f}, {q_scale[2]:.2f}] | "
              f"r_scale: [{r_scale[0]:.2f}, {r_scale[1]:.2f}, {r_scale[2]:.2f}] | "
              f"推理时间: {result['inference_time']*1000:.1f}/{avg_time:.1f} ms", 
              end='', flush=True)
    
    def print_statistics(self):
        """打印运行统计"""
        print("\n" + "="*70)
        print("  运行统计")
        print("="*70)
        
        runtime = time.time() - self.performance['start_time']
        count = self.performance['inference_count']
        
        print(f"运行时间: {runtime:.2f} s")
        print(f"推理次数: {count}")
        print(f"推理频率: {count/runtime:.2f} Hz")
        print(f"平均推理时间: {self.performance['total_inference_time']/count*1000:.2f} ms")
        print(f"最大推理时间: {self.performance['max_inference_time']*1000:.2f} ms")
        print(f"无效估计: {self.performance['invalid_estimates']}")
        
        if self.last_wind_estimate is not None:
            print(f"\n最后风速估计: [{self.last_wind_estimate[0]:.2f}, "
                  f"{self.last_wind_estimate[1]:.2f}, {self.last_wind_estimate[2]:.2f}] m/s")
            print(f"最后q_scale: [{self.last_q_scale[0]:.2f}, {self.last_q_scale[1]:.2f}, {self.last_q_scale[2]:.2f}]")
            print(f"最后r_scale: [{self.last_r_scale[0]:.2f}, {self.last_r_scale[1]:.2f}, {self.last_r_scale[2]:.2f}]")
            print(f"最后小角修正: Δα={self.last_angles[0]*180/np.pi:.2f}°, "
                  f"Δβ={self.last_angles[1]*180/np.pi:.2f}°, s_tas={self.last_angles[2]:.4f}")
        
        print("="*70)
    
    def run(self):
        """主运行循环"""
        print("="*70)
        print(" PI-GRU在线风速估计器 v3.0")
        print("="*70)
        
        # 连接MAVLink
        if not self.connect_mavlink():
            self.logger.error("无法连接MAVLink，退出")
            return
        
        if self.mode == 'sitl_eval':
            print("\n开始SITL评估推理...")
        else:
            print("\n开始实时推理...")
        print("按Ctrl+C停止\n")
        
        inference_interval = 1.0 / self.inference_rate
        
        try:
            while True:
                loop_start = time.time()
                
                # 收集数据
                msg_dict = self.collect_mavlink_data()
                if msg_dict is None:
                    continue
                
                # 提取特征
                features = self.extract_features(msg_dict)
                if features is None:
                    continue
                
                # 添加到缓冲区
                self.data_buffer.append(features)
                self.last_msg_dict = msg_dict
                
                # 执行推理
                result = self.run_inference()
                if result is not None:
                    should_send = (self.mode == 'deploy') or self.eval_send_outputs
                    if should_send:
                        self.send_to_px4(result)

                    if self.mode == 'sitl_eval':
                        self.log_eval_row(result, msg_dict)
                    
                    # 打印状态
                    self.print_status(result)

                # SITL评估自动结束条件
                if self.mode == 'sitl_eval':
                    runtime_s = time.time() - self.performance['start_time']
                    if runtime_s >= self.eval_duration_s:
                        self.logger.info(f"SITL评估达到时长上限: {runtime_s:.1f}s")
                        break
                    if self.eval_max_samples > 0 and self.eval_rows >= self.eval_max_samples:
                        self.logger.info(f"SITL评估达到样本上限: {self.eval_rows}")
                        break
                
                # 控制推理频率
                elapsed = time.time() - loop_start
                if elapsed < inference_interval:
                    time.sleep(inference_interval - elapsed)
        
        except KeyboardInterrupt:
            print("\n\n收到停止信号")
        
        except Exception as e:
            self.logger.error(f"运行时错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # 打印统计
            self.print_statistics()

            if self.backend is not None:
                self.backend.close()

            if self.eval_csv_file is not None:
                self.eval_csv_file.close()
                self.logger.info(f"SITL评估CSV已保存，样本数: {self.eval_rows}")
            
            # 关闭连接
            if self.connection is not None:
                self.connection.close()
                self.logger.info("MAVLink连接已关闭")
            
            self.logger.info("程序退出")


if __name__ == "__main__":
    print("="*70)
    print(" PI-GRU在线风速估计器 v3.0 (EKF融合增强版)")
    print("="*70)
    
    try:
        # 创建估计器
        estimator = OnlineWindEstimator(config_path='../config/config.yaml')
        
        # 运行
        estimator.run()
    
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)