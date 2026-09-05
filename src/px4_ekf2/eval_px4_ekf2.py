"""
PX4 EKF2 风速估计器 - 离线测试脚本
使用 .npy 测试集评估 PX4 EKF2 baseline 性能

PX4 EKF2 状态向量: [wind_N, wind_E, airspeed_scale] (3维)
注意: PX4 EKF2 只估水平风 (N/E)，不估垂直风 (D)
      垂直风分量评估时以 0 填充

输入特征索引 (20维):
  [0:3]  = vel_n, vel_e, vel_d  (地速 NED)
  [9:12] = roll, pitch, yaw     (姿态角)
  [19]   = airspeed             (真空速)
"""

import numpy as np
import pickle
import yaml
import os
import sys
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fuse_airspeed import fuse_airspeed


class PX4EKF2WindEstimator:
    """
    PX4 EKF2 风速估计器
    忠实复现 wind_estimator_replay.py 的算法逻辑，适配 .npy 测试集接口
    """

    def __init__(self,
                 wind_nsd: float = 1e-2,
                 scale_nsd: float = 1e-4,
                 R_airspeed: float = 1.4**2,
                 dt: float = 0.05):
        """
        Args:
            wind_nsd:    风速过程噪声谱密度 (m/s/sqrt(s))
            scale_nsd:   空速缩放因子过程噪声谱密度
            R_airspeed:  空速测量噪声方差 (1.4 m/s 标准差)
            dt:          采样时间间隔 (s)，默认 20Hz -> 0.05s
        """
        self.dt = dt
        self.Q = np.diag([wind_nsd**2, wind_nsd**2, scale_nsd**2])
        self.R = R_airspeed
        self.epsilon = 1e-8
        self.reset()

    def reset(self):
        """重置滤波器状态"""
        self.state = np.array([0.0, 0.0, 1.0])   # [wind_N, wind_E, airspeed_scale]
        self.P = np.diag([1.0, 1.0, 1e-4])

    def step(self, v_ground: np.ndarray, airspeed: float) -> np.ndarray:
        """
        单步预测+更新
        Args:
            v_ground: [3] 地速 NED (m/s)
            airspeed: 真空速 (m/s)
        Returns:
            wind_ned: [3] 风速估计 NED，垂直分量填 0
        """
        # 预测步骤: P += Q * dt
        self.P += self.Q * self.dt

        # 测量更新
        H, K, innov_var, innov = fuse_airspeed(
            v_ground, self.state, self.P, airspeed, self.R, self.epsilon
        )

        # 状态更新
        self.state += np.array(K) * innov

        # 协方差更新: P -= K * H * P
        self.P -= np.outer(K, H) @ self.P
        self.P = (self.P + self.P.T) / 2  # 保持对称

        # 返回 [wind_N, wind_E, 0]，垂直风 PX4 EKF2 不估
        return np.array([self.state[0], self.state[1], 0.0])


class PX4EKF2Evaluator:
    """PX4 EKF2 评估器"""

    def __init__(self, config_path=None):
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            config_path = os.path.join(project_root, 'config', 'config.yaml')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dt = 1.0 / self.config['data']['sampling_rate']

        self.load_normalization_params()

        self.ekf = PX4EKF2WindEstimator(dt=self.dt)

        print(f"\n【PX4 EKF2 评估器初始化】")
        print(f"  采样率: {self.config['data']['sampling_rate']} Hz  (dt={self.dt:.4f}s)")
        print(f"  fuse_airspeed: symforce版 (sym 已安装)")

    def load_normalization_params(self):
        model_save_path = self.config['training']['model_save_path']
        if not os.path.isabs(model_save_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            model_save_path = os.path.join(project_root, model_save_path.lstrip('../'))

        norm_path = os.path.join(model_save_path, 'norm_params.pkl')
        with open(norm_path, 'rb') as f:
            metadata = pickle.load(f)

        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']
        self.y_mean = self.scaler_y.mean_
        self.y_std = self.scaler_y.scale_
        self.wind_mean = self.y_mean[0:3]
        self.wind_std = self.y_std[0:3]

    def get_evaluation_dir(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        eval_dir = os.path.join(project_root, 'data', 'evaluation',
                                f'eval_px4_ekf2_{self.timestamp}')
        os.makedirs(eval_dir, exist_ok=True)
        return eval_dir

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        批量预测
        Args:
            X_test: [N, seq_len, 20] 归一化输入
        Returns:
            wind_pred_norm: [N, 3] 归一化风速估计

        注意: 测试集样本经过随机打乱，不具备时序连续性。
        每个样本用序列内最后 seq_len 步热身 EKF，取最后一步输出，
        避免跨样本状态积累导致发散。
        """
        print("\n执行 PX4 EKF2 估计...")
        N = X_test.shape[0]
        seq_len = X_test.shape[1]
        wind_pred_norm = np.zeros((N, 3))

        for i in tqdm(range(N), desc='PX4 EKF2 估计'):
            # 每个样本独立重置，用序列内所有帧热身
            self.ekf.reset()
            X_denorm = self.scaler_X.inverse_transform(X_test[i])  # [seq_len, 20]

            for t in range(seq_len):
                v_ground = X_denorm[t, 0:3]
                airspeed = X_denorm[t, 19]
                wind_ned = self.ekf.step(v_ground, airspeed)

            # 取序列最后一步的输出
            wind_pred_norm[i] = (wind_ned - self.wind_mean) / self.wind_std

        print(f"  ✓ 估计完成: {N} 样本")
        return wind_pred_norm

    def denormalize(self, wind_pred_norm, y_test_norm):
        wind_pred = wind_pred_norm * self.wind_std + self.wind_mean
        y_test_denorm = self.scaler_y.inverse_transform(y_test_norm)
        wind_true = y_test_denorm[:, 0:3]
        return wind_pred, wind_true

    def calculate_metrics(self, wind_pred, wind_true):
        print("\n计算评估指标...")
        metrics = {}

        # 整体 (注意: PX4 EKF2 垂直风恒为0，Down分量误差会偏大)
        metrics['rmse'] = np.sqrt(mean_squared_error(wind_true, wind_pred))
        metrics['mae'] = mean_absolute_error(wind_true, wind_pred)

        for i, name in enumerate(['north', 'east', 'down']):
            metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(wind_true[:, i], wind_pred[:, i]))
            metrics[f'{name}_mae'] = mean_absolute_error(wind_true[:, i], wind_pred[:, i])
            metrics[f'{name}_r2'] = r2_score(wind_true[:, i], wind_pred[:, i])

        # 水平风速（PX4 EKF2 的有效估计范围）
        wind_horiz_true = wind_true[:, :2]
        wind_horiz_pred = wind_pred[:, :2]
        metrics['horizontal_rmse'] = np.sqrt(mean_squared_error(wind_horiz_true, wind_horiz_pred))
        metrics['horizontal_mae'] = mean_absolute_error(wind_horiz_true, wind_horiz_pred)

        wind_mag_true = np.linalg.norm(wind_true, axis=1)
        wind_mag_pred = np.linalg.norm(wind_pred, axis=1)
        metrics['magnitude_rmse'] = np.sqrt(mean_squared_error(wind_mag_true, wind_mag_pred))
        metrics['magnitude_mae'] = mean_absolute_error(wind_mag_true, wind_mag_pred)
        metrics['magnitude_r2'] = r2_score(wind_mag_true, wind_mag_pred)

        print(f"  ✓ 整体 RMSE:      {metrics['rmse']:.3f} m/s")
        print(f"  ✓ 整体 MAE:       {metrics['mae']:.3f} m/s")
        print(f"  ✓ 水平风 RMSE:    {metrics['horizontal_rmse']:.3f} m/s  ← EKF2 有效范围")
        print(f"  ✓ North RMSE:     {metrics['north_rmse']:.3f} m/s")
        print(f"  ✓ East  RMSE:     {metrics['east_rmse']:.3f} m/s")
        print(f"  ✓ Down  RMSE:     {metrics['down_rmse']:.3f} m/s  (EKF2 不估垂直风，恒为0)")

        return metrics

    def plot_results(self, wind_pred, wind_true, eval_dir):
        """
        生成专业评估图表，3x3 布局：
          行1: North / East / Down 散点图 (True vs Predicted)
          行2: 风速大小散点图 / 水平风向散点图 / 误差时序图
          行3: North / East / Down 误差分布直方图
        """
        print("\n生成可视化图表...")

        wind_mag_true = np.linalg.norm(wind_true, axis=1)
        wind_mag_pred = np.linalg.norm(wind_pred, axis=1)
        wind_dir_true = np.arctan2(wind_true[:, 1], wind_true[:, 0]) * 180 / np.pi
        wind_dir_pred = np.arctan2(wind_pred[:, 1], wind_pred[:, 0]) * 180 / np.pi
        dir_error = (wind_dir_pred - wind_dir_true + 180) % 360 - 180

        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        fig.suptitle('PX4 EKF2 Wind Estimation Evaluation (Baseline)', fontsize=14, fontweight='bold', y=0.98)

        colors = {'north': '#4472C4', 'east': '#ED7D31', 'down': '#A9D18E'}
        comp_names = ['North', 'East', 'Down']

        # ── 行1: 各分量 True vs Predicted 散点图 ──
        for i, (name, color) in enumerate(zip(comp_names, colors.values())):
            ax = axes[0, i]
            ax.scatter(wind_true[:, i], wind_pred[:, i],
                       alpha=0.25, s=2, c=color, rasterized=True, label='Samples')
            lims = [min(wind_true[:, i].min(), wind_pred[:, i].min()),
                    max(wind_true[:, i].max(), wind_pred[:, i].max())]
            ax.plot(lims, lims, 'r--', lw=1.5, label='Ideal', zorder=10)
            rmse = np.sqrt(mean_squared_error(wind_true[:, i], wind_pred[:, i]))
            r2 = r2_score(wind_true[:, i], wind_pred[:, i])
            if name == 'Down':
                ax.set_title(f'{name} Wind (EKF2 outputs 0)\nRMSE={rmse:.3f} m/s, R²={r2:.3f}',
                             fontsize=10, fontweight='bold')
            else:
                ax.set_title(f'{name} Wind\nRMSE={rmse:.3f} m/s, R²={r2:.3f}',
                             fontsize=10, fontweight='bold')
            ax.set_xlabel(f'True {name} (m/s)', fontsize=9)
            ax.set_ylabel(f'Predicted {name} (m/s)', fontsize=9)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # ── 行2左: 风速大小散点图 ──
        ax = axes[1, 0]
        ax.scatter(wind_mag_true, wind_mag_pred, alpha=0.25, s=2, c='#5B9BD5', rasterized=True)
        lims = [wind_mag_true.min(), wind_mag_true.max()]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Ideal', zorder=10)
        rmse_mag = np.sqrt(mean_squared_error(wind_mag_true, wind_mag_pred))
        r2_mag = r2_score(wind_mag_true, wind_mag_pred)
        ax.set_title(f'Wind Magnitude\nRMSE={rmse_mag:.3f} m/s, R²={r2_mag:.3f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('True Magnitude (m/s)', fontsize=9)
        ax.set_ylabel('Predicted Magnitude (m/s)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ── 行2中: 水平风向散点图 ──
        ax = axes[1, 1]
        ax.scatter(wind_dir_true, wind_dir_pred, alpha=0.25, s=2, c='#7030A0', rasterized=True)
        ax.plot([-180, 180], [-180, 180], 'r--', lw=1.5, label='Ideal', zorder=10)
        mae_dir = np.mean(np.abs(dir_error))
        ax.set_title(f'Horizontal Wind Direction\nMAE={mae_dir:.2f}°',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('True Direction (°)', fontsize=9)
        ax.set_ylabel('Predicted Direction (°)', fontsize=9)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ── 行2右: 水平风速误差时序（采样）──
        ax = axes[1, 2]
        n = len(wind_pred)
        idx = np.arange(0, n, max(1, n // 2000))  # 最多显示2000点
        err_n = wind_pred[idx, 0] - wind_true[idx, 0]
        err_e = wind_pred[idx, 1] - wind_true[idx, 1]
        ax.plot(idx, err_n, alpha=0.7, lw=0.8, color='#4472C4', label='North error')
        ax.plot(idx, err_e, alpha=0.7, lw=0.8, color='#ED7D31', label='East error')
        ax.axhline(0, color='black', lw=1.0, linestyle='--')
        ax.set_title('Horizontal Wind Error (Time Series)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=9)
        ax.set_ylabel('Error (m/s)', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # ── 行3: 各分量误差分布直方图 ──
        for i, (name, color) in enumerate(zip(comp_names, colors.values())):
            ax = axes[2, i]
            errors = wind_pred[:, i] - wind_true[:, i]
            ax.hist(errors, bins=60, alpha=0.75, color=color, edgecolor='white', linewidth=0.3)
            ax.axvline(0, color='red', linestyle='--', lw=1.5, label='Zero')
            ax.axvline(errors.mean(), color='black', linestyle='-', lw=1.5,
                       label=f'μ={errors.mean():.3f}')
            ax.set_title(f'{name} Error Distribution\nμ={errors.mean():.3f}, σ={errors.std():.3f} m/s',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Error (m/s)', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(eval_dir, 'evaluation_results.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ 评估图已保存: {save_path}")

    def plot_metrics_bar(self, metrics, eval_dir):
        """各分量 RMSE / MAE 柱状图对比"""
        print("  生成指标柱状图...")

        components = ['North', 'East', 'Down', 'Magnitude']
        keys = ['north', 'east', 'down', 'magnitude']
        rmse_vals = [metrics[f'{k}_rmse'] for k in keys]
        mae_vals  = [metrics[f'{k}_mae']  for k in keys]

        x = np.arange(len(components))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width/2, rmse_vals, width, label='RMSE', color='#4472C4', alpha=0.85)
        bars2 = ax.bar(x + width/2, mae_vals,  width, label='MAE',  color='#ED7D31', alpha=0.85)

        ax.set_title('PX4 EKF2 — Per-Component Error Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error (m/s)', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(components, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 在柱顶标注数值
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

        # 标注 Down 分量说明
        ax.annotate('* Down = 0\n(EKF2 limitation)',
                    xy=(3 - width/2, rmse_vals[3] + 0.02),
                    fontsize=7, color='gray', ha='center')

        plt.tight_layout()
        save_path = os.path.join(eval_dir, 'metrics_bar.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"  ✓ 指标柱状图已保存: {save_path}")

    def plot_waveform(self, wind_pred, wind_true, eval_dir):
        """
        估计值 vs 真值 三分量 + 风速大小 波形对比图（4 行共享时间轴）
        """
        print("  生成波形对比图...")
        N = len(wind_pred)
        dt = self.dt
        time_sec = np.arange(N) * dt

        # 下采样，避免 SVG 过大
        step = max(1, N // 4000)
        t = time_sec[::step]

        wind_mag_true = np.linalg.norm(wind_true, axis=1)
        wind_mag_pred = np.linalg.norm(wind_pred, axis=1)

        comp_cfg = [
            ('North', wind_true[:, 0], wind_pred[:, 0], '#4472C4'),
            ('East',  wind_true[:, 1], wind_pred[:, 1], '#ED7D31'),
            ('Down',  wind_true[:, 2], wind_pred[:, 2], '#A9D18E'),
            ('Magnitude', wind_mag_true, wind_mag_pred, '#5B9BD5'),
        ]

        fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)
        fig.suptitle('PX4 EKF2 — Estimated vs True Wind (Time Series)',
                     fontsize=13, fontweight='bold')

        for ax, (label, true, pred, color) in zip(axes, comp_cfg):
            ax.plot(t, true[::step],  color='#555555', lw=0.7, alpha=0.8, label='True')
            ax.plot(t, pred[::step],  color=color,     lw=0.9, alpha=0.9, label='Estimated')
            err = pred - true
            rmse = np.sqrt(np.mean(err ** 2))
            ax.set_ylabel(f'{label}\n(m/s)', fontsize=9)
            ax.set_title(f'{label}  RMSE={rmse:.3f} m/s', fontsize=9, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right', ncol=2)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if label == 'Down':
                ax.text(0.01, 0.95, 'EKF2 outputs 0 for Down',
                        transform=ax.transAxes, fontsize=7.5,
                        color='gray', va='top')

        axes[-1].set_xlabel('Time (s)', fontsize=10)
        plt.tight_layout()

        save_path = os.path.join(eval_dir, 'wind_waveform.svg')
        fig.savefig(save_path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ 波形图已保存: {save_path}")

    def run(self, X_test, y_test):
        print("\n" + "="*70)
        print("  PX4 EKF2 评估 (Baseline)")
        print("="*70)
        print("  注意: PX4 EKF2 只估水平风 (N/E)，垂直风 (D) 输出恒为 0")

        wind_pred_norm = self.predict(X_test)
        wind_pred, wind_true = self.denormalize(wind_pred_norm, y_test)
        metrics = self.calculate_metrics(wind_pred, wind_true)

        eval_dir = self.get_evaluation_dir()
        with open(os.path.join(eval_dir, 'evaluation_metrics.pkl'), 'wb') as f:
            pickle.dump({'metrics': metrics, 'model_type': 'PX4_EKF2',
                         'config': self.config}, f)

        self.plot_results(wind_pred, wind_true, eval_dir)
        self.plot_metrics_bar(metrics, eval_dir)
        self.plot_waveform(wind_pred, wind_true, eval_dir)

        print(f"\n  ✓ 结果已保存: {eval_dir}")
        print("\n" + "="*70)
        print("  ✅ PX4 EKF2 评估完成")
        print("="*70)

        return metrics, wind_pred, wind_true


if __name__ == "__main__":
    print("="*70)
    print(" PX4 EKF2 风速估计器评估 (Baseline)")
    print("="*70)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        config_path = os.path.join(project_root, 'config', 'config.yaml')

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        print("\n加载测试数据...")
        data_dir = config['data']['processed_dir']
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(project_root, data_dir.lstrip('../'))

        x_id = os.path.join(data_dir, 'X_test_id.npy')
        y_id = os.path.join(data_dir, 'y_test_id.npy')
        x_leg = os.path.join(data_dir, 'X_test.npy')
        y_leg = os.path.join(data_dir, 'y_test.npy')

        if os.path.exists(x_id) and os.path.exists(y_id):
            X_test = np.load(x_id)
            y_test = np.load(y_id)
            print(f"  ✓ X_test_id: {X_test.shape}")
            print(f"  ✓ y_test_id: {y_test.shape}")
        else:
            X_test = np.load(x_leg)
            y_test = np.load(y_leg)
            print(f"  ✓ X_test (legacy): {X_test.shape}")
            print(f"  ✓ y_test (legacy): {y_test.shape}")

        evaluator = PX4EKF2Evaluator(config_path=config_path)
        metrics, wind_pred, wind_true = evaluator.run(X_test, y_test)

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
