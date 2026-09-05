"""
Vanilla GRU 评估模块 - 论文对比基线
与 4_eval_pigru.py 保持相同接口，用于多方法对比
"""

import numpy as np
import torch
import pickle
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
import argparse
from tqdm import tqdm
from datetime import datetime

# 导入Vanilla GRU模型
import importlib.util
model_file = os.path.join(os.path.dirname(__file__), '2b_vanilla_gru.py')
spec = importlib.util.spec_from_file_location("vanilla_gru", model_file)
if spec and spec.loader:
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    VanillaGRU = model_module.VanillaGRU
else:
    raise ImportError("Cannot load 2b_vanilla_gru.py")


class VanillaGRUEvaluator:
    """Vanilla GRU 模型评估器"""
    
    def __init__(self, config_path=None, model_dir=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.model_dir = model_dir
        self.output_base_dir = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.model = self.load_model()
        self.model.eval()
        
        self.load_normalization_params()
        self.results = {}
    
    def get_evaluation_dir(self):
        """获取评估结果保存目录"""
        if self.output_base_dir:
            eval_base_dir = self.output_base_dir
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            eval_base_dir = os.path.join(project_root, 'data', 'evaluation')
        eval_dir = os.path.join(eval_base_dir, f'eval_vanilla_gru_{self.timestamp}')
        os.makedirs(eval_dir, exist_ok=True)
        
        return eval_dir
    
    def load_model(self):
        """加载训练好的Vanilla GRU模型"""
        model_save_path = self.config['training']['model_save_path']
        if not os.path.isabs(model_save_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_save_path = os.path.join(project_root, model_save_path.lstrip('../'))
        
        if self.model_dir:
            model_dir = self.model_dir
        else:
            # 查找最新的 vanilla_gru_* 目录
            train_dirs = [d for d in os.listdir(model_save_path) 
                         if d.startswith('vanilla_gru_') and os.path.isdir(os.path.join(model_save_path, d))]
            if not train_dirs:
                raise FileNotFoundError(f"在 {model_save_path} 中未找到 Vanilla GRU 训练目录")
            train_dirs.sort(reverse=True)
            model_dir = os.path.join(model_save_path, train_dirs[0])
            print(f"\n自动选择最新的训练目录: {train_dirs[0]}")
        
        model_path = os.path.join(model_dir, 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"\n加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model = VanillaGRU(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=0.0
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model_info = model.get_model_info()
        print(f"  ✓ 模型类型: {model_info['model_type']}")
        print(f"  ✓ 参数量: {model_info['total_params']:,}")
        
        if 'epoch' in checkpoint:
            print(f"  ✓ 训练轮数: {checkpoint['epoch']}")
        if 'selection_metric' in checkpoint:
            print(f"  ✓ 选模指标: {checkpoint['selection_metric']}")
        if 'best_val_rmse' in checkpoint:
            print(f"  ✓ 最佳验证RMSE: {checkpoint['best_val_rmse']:.4f}")
        if 'best_val_loss' in checkpoint:
            print(f"  ✓ 最佳验证损失: {checkpoint['best_val_loss']:.4f}")
        if 'best_val_wind_mag_error' in checkpoint:
            print(f"  ✓ 最佳风速大小误差: {checkpoint['best_val_wind_mag_error']:.4f}")
        
        return model

    
    def load_normalization_params(self):
        """加载归一化参数"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        candidate_dirs = []
        for key_path in [
            self.config.get('data', {}).get('processed_dir'),
            self.config.get('experiment', {}).get('processed_dir'),
            self.config['training'].get('model_save_path'),
        ]:
            if not key_path:
                continue
            if os.path.isabs(key_path):
                candidate_dirs.append(key_path)
            else:
                candidate_dirs.append(os.path.normpath(os.path.join(project_root, key_path.lstrip('../'))))

        norm_path = None
        for directory in candidate_dirs:
            candidate = os.path.join(directory, 'norm_params.pkl')
            if os.path.exists(candidate):
                norm_path = candidate
                break
        if norm_path is None:
            raise FileNotFoundError(f"归一化参数文件不存在，已尝试目录: {candidate_dirs}")
        
        with open(norm_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']
        
        self.y_mean = self.scaler_y.mean_
        self.y_std = self.scaler_y.scale_
        
        self.wind_mean = self.y_mean[0:3]
        self.wind_std = self.y_std[0:3]
        
        print(f"\n【归一化参数加载成功】")
    
    def predict(self, X_test):
        """模型预测"""
        print("\n执行预测...")
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        all_wind_preds = []
        batch_size = 256
        
        with torch.no_grad():
            for i in tqdm(range(0, len(X_test_tensor), batch_size), desc='预测进度'):
                batch = X_test_tensor[i:i+batch_size]
                wind = self.model(batch, return_dict=False)
                all_wind_preds.append(wind.cpu().numpy())
        
        wind_pred = np.vstack(all_wind_preds)
        print(f"  ✓ 预测完成: {len(wind_pred)} 样本")
        
        return wind_pred
    
    def denormalize(self, wind_pred_norm, y_test_norm):
        """反归一化"""
        wind_pred = wind_pred_norm * self.wind_std + self.wind_mean
        y_test_denorm = self.scaler_y.inverse_transform(y_test_norm)
        wind_true = y_test_denorm[:, 0:3]
        
        return wind_pred, wind_true
    
    def calculate_metrics(self, wind_pred, wind_true):
        """计算评估指标"""
        print("\n计算评估指标...")
        
        metrics = {}
        
        metrics['rmse'] = np.sqrt(mean_squared_error(wind_true, wind_pred))
        metrics['mae'] = mean_absolute_error(wind_true, wind_pred)
        
        component_names = ['north', 'east', 'down']
        for i, name in enumerate(component_names):
            metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(wind_true[:, i], wind_pred[:, i]))
            metrics[f'{name}_mae'] = mean_absolute_error(wind_true[:, i], wind_pred[:, i])
            metrics[f'{name}_r2'] = r2_score(wind_true[:, i], wind_pred[:, i])
        
        wind_mag_true = np.linalg.norm(wind_true, axis=1)
        wind_mag_pred = np.linalg.norm(wind_pred, axis=1)
        
        metrics['magnitude_rmse'] = np.sqrt(mean_squared_error(wind_mag_true, wind_mag_pred))
        metrics['magnitude_mae'] = mean_absolute_error(wind_mag_true, wind_mag_pred)
        metrics['magnitude_r2'] = r2_score(wind_mag_true, wind_mag_pred)
        
        print(f"  ✓ 整体RMSE: {metrics['rmse']:.3f} m/s")
        print(f"  ✓ 整体MAE: {metrics['mae']:.3f} m/s")
        
        return metrics
    
    def plot_scatter(self, wind_pred, wind_true, eval_dir):
        """3×3 散点评估图"""
        print("  生成散点评估图...")
        from scipy.stats import gaussian_kde as _kde

        wind_mag_t = np.linalg.norm(wind_true, axis=1)
        wind_mag_p = np.linalg.norm(wind_pred, axis=1)
        wind_dir_t = np.degrees(np.arctan2(wind_true[:, 1], wind_true[:, 0]))
        wind_dir_p = np.degrees(np.arctan2(wind_pred[:, 1], wind_pred[:, 0]))
        dir_err    = (wind_dir_p - wind_dir_t + 180) % 360 - 180

        COLORS = {'north': '#4472C4', 'east': '#ED7D31', 'down': '#A9D18E',
                  'mag': '#5B9BD5', 'dir': '#7030A0'}
        comp_names  = ['North', 'East', 'Down']
        comp_colors = [COLORS['north'], COLORS['east'], COLORS['down']]

        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        fig.suptitle('Vanilla GRU Wind Estimation Evaluation (Data-Driven Baseline)',
                     fontsize=14, fontweight='bold', y=0.98)

        # 行1: 各分量散点
        for i, (name, color) in enumerate(zip(comp_names, comp_colors)):
            ax = axes[0, i]
            ax.scatter(wind_true[:, i], wind_pred[:, i],
                       alpha=0.2, s=2, c=color, rasterized=True, label='Samples')
            lims = [min(wind_true[:, i].min(), wind_pred[:, i].min()),
                    max(wind_true[:, i].max(), wind_pred[:, i].max())]
            ax.plot(lims, lims, 'r--', lw=1.5, label='Ideal', zorder=10)
            rmse = np.sqrt(mean_squared_error(wind_true[:, i], wind_pred[:, i]))
            r2   = r2_score(wind_true[:, i], wind_pred[:, i])
            ax.set_title(f'{name} Wind\nRMSE={rmse:.3f} m/s, R²={r2:.3f}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel(f'True {name} (m/s)', fontsize=9)
            ax.set_ylabel(f'Predicted {name} (m/s)', fontsize=9)
            ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # 行2左: 风速大小
        ax = axes[1, 0]
        ax.scatter(wind_mag_t, wind_mag_p, alpha=0.2, s=2, c=COLORS['mag'], rasterized=True)
        lims = [wind_mag_t.min(), wind_mag_t.max()]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Ideal', zorder=10)
        rmse_m = np.sqrt(mean_squared_error(wind_mag_t, wind_mag_p))
        r2_m   = r2_score(wind_mag_t, wind_mag_p)
        ax.set_title(f'Wind Magnitude\nRMSE={rmse_m:.3f} m/s, R²={r2_m:.3f}',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('True Magnitude (m/s)', fontsize=9)
        ax.set_ylabel('Predicted Magnitude (m/s)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # 行2中: 水平风向
        ax = axes[1, 1]
        ax.scatter(wind_dir_t, wind_dir_p, alpha=0.2, s=2, c=COLORS['dir'], rasterized=True)
        ax.plot([-180, 180], [-180, 180], 'r--', lw=1.5, label='Ideal', zorder=10)
        ax.set_title(f'Horizontal Wind Direction\nMAE={np.mean(np.abs(dir_err)):.2f}°',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('True Direction (°)', fontsize=9)
        ax.set_ylabel('Predicted Direction (°)', fontsize=9)
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # 行2右: 水平误差时序
        ax = axes[1, 2]
        n   = len(wind_pred)
        idx = np.arange(0, n, max(1, n // 2000))
        ax.plot(idx, (wind_pred - wind_true)[idx, 0], lw=0.8,
                color=COLORS['north'], alpha=0.8, label='North error')
        ax.plot(idx, (wind_pred - wind_true)[idx, 1], lw=0.8,
                color=COLORS['east'],  alpha=0.8, label='East error')
        ax.axhline(0, color='black', lw=1.0, linestyle='--')
        ax.set_title('Horizontal Wind Error (Time Series)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Sample Index', fontsize=9)
        ax.set_ylabel('Error (m/s)', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        # 行3: 误差分布直方图 + KDE
        for i, (name, color) in enumerate(zip(comp_names, comp_colors)):
            ax = axes[2, i]
            err = wind_pred[:, i] - wind_true[:, i]
            ax.hist(err, bins=60, alpha=0.65, color=color,
                    edgecolor='white', linewidth=0.3, density=True)
            x = np.linspace(err.min(), err.max(), 200)
            ax.plot(x, _kde(err)(x), color='navy', lw=1.5, label='KDE')
            ax.axvline(0,          color='red',  lw=1.5, linestyle='--', label='Zero')
            ax.axvline(err.mean(), color='black', lw=1.5, linestyle='-',
                       label=f'μ={err.mean():.3f}')
            ax.set_title(f'{name} Error Distribution\nμ={err.mean():.3f}, σ={err.std():.3f} m/s',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('Error (m/s)', fontsize=9)
            ax.set_ylabel('Density', fontsize=9)
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path = os.path.join(eval_dir, 'evaluation_results.svg')
        fig.savefig(path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ 散点评估图已保存: {path}")

    def plot_waveform(self, wind_pred, wind_true, eval_dir):
        """N/E/D + 风速大小 波形对比图"""
        print("  生成波形对比图...")
        dt   = 1.0 / self.config['data']['sampling_rate']
        N    = len(wind_pred)
        t    = np.arange(N) * dt
        step = max(1, N // 4000)
        mag_t = np.linalg.norm(wind_true, axis=1)
        mag_p = np.linalg.norm(wind_pred, axis=1)
        COLORS = {'north': '#4472C4', 'east': '#ED7D31',
                  'down': '#A9D18E',  'mag':  '#5B9BD5'}
        comp_cfg = [
            ('North',     wind_true[:, 0], wind_pred[:, 0], COLORS['north']),
            ('East',      wind_true[:, 1], wind_pred[:, 1], COLORS['east']),
            ('Down',      wind_true[:, 2], wind_pred[:, 2], COLORS['down']),
            ('Magnitude', mag_t,           mag_p,           COLORS['mag']),
        ]
        fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)
        fig.suptitle('Vanilla GRU — Estimated vs True Wind (Time Series)',
                     fontsize=13, fontweight='bold')
        for ax, (label, true, pred, color) in zip(axes, comp_cfg):
            ax.plot(t[::step], true[::step], color='#555555', lw=0.7, alpha=0.8, label='True')
            ax.plot(t[::step], pred[::step], color=color,     lw=0.9, alpha=0.9, label='Estimated')
            rmse = np.sqrt(np.mean((pred - true) ** 2))
            ax.set_ylabel(f'{label}\n(m/s)', fontsize=9)
            ax.set_title(f'{label}  RMSE={rmse:.3f} m/s', fontsize=9, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right', ncol=2)
            ax.grid(True, alpha=0.25, linestyle='--')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        axes[-1].set_xlabel('Time (s)', fontsize=10)
        plt.tight_layout()
        path = os.path.join(eval_dir, 'wind_waveform.svg')
        fig.savefig(path, format='svg', bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  ✓ 波形图已保存: {path}")

    def plot_metrics_bar(self, wind_pred, wind_true, eval_dir):
        """各分量 RMSE/MAE 柱状图"""
        print("  生成指标柱状图...")
        names = ['North', 'East', 'Down', 'Magnitude']
        mag_t = np.linalg.norm(wind_true, axis=1)
        mag_p = np.linalg.norm(wind_pred, axis=1)
        all_t = [wind_true[:, 0], wind_true[:, 1], wind_true[:, 2], mag_t]
        all_p = [wind_pred[:, 0], wind_pred[:, 1], wind_pred[:, 2], mag_p]
        rmse_vals = [np.sqrt(mean_squared_error(t, p)) for t, p in zip(all_t, all_p)]
        mae_vals  = [mean_absolute_error(t, p)          for t, p in zip(all_t, all_p)]
        x = np.arange(len(names)); w = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        b1 = ax.bar(x - w/2, rmse_vals, w, label='RMSE', color='#4472C4', alpha=0.85)
        b2 = ax.bar(x + w/2, mae_vals,  w, label='MAE',  color='#ED7D31', alpha=0.85)
        ax.set_title('Vanilla GRU — Per-Component Error Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error (m/s)', fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        path = os.path.join(eval_dir, 'metrics_bar.svg')
        fig.savefig(path, format='svg', bbox_inches='tight')
        plt.close()
        print(f"  ✓ 指标柱状图已保存: {path}")

    def run(self, X_test, y_test, eval_dir=None):
        """执行完整评估流程"""
        print("\n" + "="*70)
        print("  Vanilla GRU 评估 (Data-Driven Baseline)")
        print("="*70)
        
        # 1. 预测
        wind_pred_norm = self.predict(X_test)
        
        # 2. 反归一化
        wind_pred, wind_true = self.denormalize(wind_pred_norm, y_test)
        
        # 3. 计算指标
        metrics = self.calculate_metrics(wind_pred, wind_true)
        
        # 4. 保存结果
        if eval_dir is None:
            eval_dir = self.get_evaluation_dir()
        os.makedirs(eval_dir, exist_ok=True)
        results = {
            'metrics': metrics,
            'model_type': 'VanillaGRU',
            'config': self.config
        }
        
        with open(os.path.join(eval_dir, 'evaluation_metrics.pkl'), 'wb') as f:
            pickle.dump(results, f)

        print("\n生成可视化图表...")
        self.plot_scatter(wind_pred, wind_true, eval_dir)
        self.plot_waveform(wind_pred, wind_true, eval_dir)
        self.plot_metrics_bar(wind_pred, wind_true, eval_dir)

        print(f"\n  ✓ 结果已保存: {eval_dir}")
        
        print("\n" + "="*70)
        print("  ✅ Vanilla GRU 评估完成！")
        print("="*70)
        
        return metrics, wind_pred, wind_true


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vanilla GRU 评估模块")
    parser.add_argument("--config_path", type=str, default=None, help="配置文件路径，默认 config/config.yaml")
    parser.add_argument("--splits", type=str, default="test_id", help="逗号分隔 split，如 test_id,test_ood")
    parser.add_argument("--output_dir", type=str, default=None, help="评估结果根目录")
    args = parser.parse_args()

    print("="*70)
    print(" Vanilla GRU 评估模块 (Data-Driven Baseline)")
    print("="*70)
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        if args.config_path:
            config_path = args.config_path if os.path.isabs(args.config_path) else os.path.join(project_root, args.config_path.lstrip('../'))
        else:
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_dir = config['data']['processed_dir']
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(project_root, data_dir.lstrip('../'))

        output_root = args.output_dir if args.output_dir else os.path.join(project_root, 'data', 'evaluation')
        if not os.path.isabs(output_root):
            output_root = os.path.join(project_root, output_root.lstrip('../'))
        os.makedirs(output_root, exist_ok=True)

        evaluator = VanillaGRUEvaluator(config_path=config_path)
        evaluator.output_base_dir = output_root
        for split in [s.strip() for s in args.splits.split(',') if s.strip()]:
            print(f"\n加载测试数据: {split}")
            X_test = np.load(os.path.join(data_dir, f'X_{split}.npy'))
            y_test = np.load(os.path.join(data_dir, f'y_{split}.npy'))
            print(f"  ✓ X_{split}: {X_test.shape}")
            print(f"  ✓ y_{split}: {y_test.shape}")
            eval_dir = os.path.join(output_root, split)
            metrics, wind_pred, wind_true = evaluator.run(X_test, y_test, eval_dir=eval_dir)
        
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
