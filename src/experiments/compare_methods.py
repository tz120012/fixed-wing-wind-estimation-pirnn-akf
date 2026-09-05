"""
多方法对比评估框架
比较论文中的3种方法：PX4-EKF2, Vanilla GRU, PIRNN-AKF

功能：
  1. 统一评估接口
  2. 生成论文 Table 1 数据
  3. 生成对比图表
  4. 保存评估结果
"""

import numpy as np
import pandas as pd
import pickle
import yaml
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel
from tqdm import tqdm

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'px4_ekf2'))


class MethodEvaluator:
    """单个方法的评估器封装"""
    
    def __init__(self, method_name: str, estimator_class, config_path: str):
        self.method_name = method_name
        self.estimator_class = estimator_class
        self.config_path = config_path
        self.estimator = None
    
    def initialize(self, **kwargs):
        """初始化估计器"""
        self.estimator = self.estimator_class(self.config_path, **kwargs)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 scaler_y) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """
        执行评估
        
        Returns:
            metrics: 评估指标字典
            wind_pred: 预测风速 (物理空间)
            wind_true: 真实风速 (物理空间)
        """
        raise NotImplementedError


class ComparisonFramework:
    """多方法对比评估框架"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.config_path = config_path
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 加载归一化参数
        self.load_normalization_params()
        
        # 结果存储
        self.results = {}
        self.predictions = {}
        self.statistical_results = {}
        
        # 创建输出目录
        self.output_dir = os.path.join(project_root, 'data', 'comparison', f'compare_{self.timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n【多方法对比评估框架】")
        print(f"  输出目录: {self.output_dir}")
    
    def load_normalization_params(self):
        """加载归一化参数"""
        model_save_path = self.config['training']['model_save_path']
        if not os.path.isabs(model_save_path):
            model_save_path = os.path.join(project_root, model_save_path.lstrip('../'))
        
        norm_path = os.path.join(model_save_path, 'norm_params.pkl')
        
        with open(norm_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']
        
        self.wind_mean = self.scaler_y.mean_[0:3]
        self.wind_std = self.scaler_y.scale_[0:3]
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载测试数据"""
        data_dir = self.config['data']['processed_dir']
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(project_root, data_dir.lstrip('../'))
        
        # 使用同分布测试集 test_id
        X_test = np.load(os.path.join(data_dir, 'X_test_id.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test_id.npy'))
        
        print(f"\n【测试数据加载】")
        print(f"  X_test_id: {X_test.shape}")
        print(f"  y_test_id: {y_test.shape}")
        
        return X_test, y_test
    
    def evaluate_standard_ekf(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估 PX4-EKF2"""
        print("\n" + "="*70)
        print("评估 PX4-EKF2")
        print("="*70)

        import importlib.util
        module_path = os.path.join(project_root, 'src', 'px4_ekf2', 'eval_px4_ekf2.py')
        spec = importlib.util.spec_from_file_location('eval_px4_ekf2', module_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        evaluator = mod.PX4EKF2Evaluator(config_path=self.config_path)
        wind_pred_norm = evaluator.predict(X_test)
        wind_pred, wind_true = evaluator.denormalize(wind_pred_norm, y_test)

        metrics = self.calculate_metrics(wind_pred, wind_true)
        self.results['PX4-EKF2'] = metrics
        self.predictions['PX4-EKF2'] = wind_pred
        return metrics
    
    def evaluate_vanilla_gru(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估Vanilla GRU"""
        print("\n" + "="*70)
        print("评估 Vanilla GRU")
        print("="*70)
        
        import importlib.util
        model_file = os.path.join(project_root, 'src', '2b_vanilla_gru.py')
        spec = importlib.util.spec_from_file_location("vanilla_gru", model_file)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        VanillaGRU = model_module.VanillaGRU
        
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 查找最新的Vanilla GRU模型
        model_save_path = self.config['training']['model_save_path']
        if not os.path.isabs(model_save_path):
            model_save_path = os.path.join(project_root, model_save_path.lstrip('../'))
        
        train_dirs = [d for d in os.listdir(model_save_path) 
                     if d.startswith('vanilla_gru_') and os.path.isdir(os.path.join(model_save_path, d))]
        
        if not train_dirs:
            print("  ⚠️ 未找到Vanilla GRU模型，跳过...")
            return None
        
        train_dirs.sort(reverse=True)
        model_path = os.path.join(model_save_path, train_dirs[0], 'best_model.pth')
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = VanillaGRU(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=0.0
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 批量预测
        X_tensor = torch.FloatTensor(X_test).to(device)
        wind_pred_norm = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(X_tensor), 256), desc='Vanilla GRU'):
                batch = X_tensor[i:i+256]
                wind = model(batch, return_dict=False)
                wind_pred_norm.append(wind.cpu().numpy())
        
        wind_pred_norm = np.vstack(wind_pred_norm)
        wind_pred = wind_pred_norm * self.wind_std + self.wind_mean
        
        # 真值
        y_denorm = self.scaler_y.inverse_transform(y_test)
        wind_true = y_denorm[:, 0:3]
        
        metrics = self.calculate_metrics(wind_pred, wind_true)
        
        self.results['Vanilla GRU'] = metrics
        self.predictions['Vanilla GRU'] = wind_pred
        
        return metrics
    
    def evaluate_iakf(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """已移除 IAKF，保留空方法避免调用报错"""
        print("\n  ⚠️ IAKF 已移除，跳过")
        return None
    
    def evaluate_pirnn_akf(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估PIRNN-AKF"""
        print("\n" + "="*70)
        print("评估 PIRNN-AKF (Ours)")
        print("="*70)
        
        import importlib.util
        module_path = os.path.join(project_root, 'src', '5_pigru_akf_fusion.py')
        spec = importlib.util.spec_from_file_location("pirnn_akf_fusion", module_path)
        pirnn_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pirnn_module)
        PIRNN_AKF = pirnn_module.PIRNN_AKF
        
        try:
            estimator = PIRNN_AKF(self.config_path)
            wind_pred_norm, _ = estimator.estimate_batch(X_test)
            
            wind_pred = wind_pred_norm * self.wind_std + self.wind_mean
            
            y_denorm = self.scaler_y.inverse_transform(y_test)
            wind_true = y_denorm[:, 0:3]
            
            metrics = self.calculate_metrics(wind_pred, wind_true)
            
            self.results['PIRNN-AKF (Ours)'] = metrics
            self.predictions['PIRNN-AKF (Ours)'] = wind_pred
            
            return metrics
        except Exception as e:
            print(f"  ⚠️ PIRNN-AKF评估失败: {e}")
            return None
    
    def calculate_metrics(self, wind_pred: np.ndarray, wind_true: np.ndarray) -> Dict:
        """计算评估指标"""
        metrics = {}
        
        # 总体RMSE和MAE
        metrics['total_rmse'] = np.sqrt(mean_squared_error(wind_true, wind_pred))
        metrics['total_mae'] = mean_absolute_error(wind_true, wind_pred)
        
        # 各分量指标
        for i, name in enumerate(['north', 'east', 'down']):
            metrics[f'{name}_rmse'] = np.sqrt(mean_squared_error(wind_true[:, i], wind_pred[:, i]))
            metrics[f'{name}_mae'] = mean_absolute_error(wind_true[:, i], wind_pred[:, i])
        
        # 风速大小
        mag_true = np.linalg.norm(wind_true, axis=1)
        mag_pred = np.linalg.norm(wind_pred, axis=1)
        metrics['magnitude_rmse'] = np.sqrt(mean_squared_error(mag_true, mag_pred))
        
        # 总向量误差
        vector_error = np.linalg.norm(wind_pred - wind_true, axis=1)
        metrics['vector_error_mean'] = np.mean(vector_error)
        metrics['vector_error_std'] = np.std(vector_error)
        
        return metrics
    
    def run_all_evaluations(self, X_test: np.ndarray, y_test: np.ndarray):
        """运行所有方法的评估"""
        print("\n" + "="*70)
        print("  开始多方法对比评估")
        print("="*70)
        
        # 1. Standard EKF
        self.evaluate_standard_ekf(X_test, y_test)
        
        # 2. Vanilla GRU
        self.evaluate_vanilla_gru(X_test, y_test)
        
        # 3. IAKF
        self.evaluate_iakf(X_test, y_test)
        
        # 4. PIRNN-AKF
        self.evaluate_pirnn_akf(X_test, y_test)
        
        # 保存真值
        y_denorm = self.scaler_y.inverse_transform(y_test)
        self.predictions['Ground Truth'] = y_denorm[:, 0:3]
    
    def generate_table1(self) -> pd.DataFrame:
        """生成论文Table 1格式的对比表"""
        print("\n" + "="*70)
        print("生成 Table 1: Statistical Performance Comparison (RMSE)")
        print("="*70)
        
        # 表格数据
        data = []
        methods = ['PX4-EKF2', 'Vanilla GRU', 'PIRNN-AKF (Ours)']
        
        # 获取EKF2的总误差作为基准
        ekf_total = self.results.get('PX4-EKF2', {}).get('vector_error_mean', 1.0)
        
        for method in methods:
            if method not in self.results:
                continue
            
            m = self.results[method]
            
            # 计算相对EKF的改进
            improvement = (1 - m['vector_error_mean'] / ekf_total) * 100 if ekf_total > 0 else 0
            
            data.append({
                'Method': method,
                'North (m/s)': f"{m['north_rmse']:.2f}",
                'East (m/s)': f"{m['east_rmse']:.2f}",
                'Down (m/s)': f"{m['down_rmse']:.2f}",
                'Total Vector Error (m/s)': f"{m['vector_error_mean']:.2f}",
                'Improvement vs. EKF': f"{improvement:.1f}%" if method != 'Standard EKF' else '-'
            })
        
        df = pd.DataFrame(data)
        
        # 保存为CSV
        csv_path = os.path.join(self.output_dir, 'table1_comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # 打印表格
        print("\n" + df.to_string(index=False))
        print(f"\n  ✓ Table 1 已保存: {csv_path}")
        
        return df

    def _get_vector_errors(self, method_name: str) -> np.ndarray:
        """获取某方法的逐样本向量误差"""
        if method_name not in self.predictions or 'Ground Truth' not in self.predictions:
            return np.array([])
        pred = self.predictions[method_name]
        truth = self.predictions['Ground Truth']
        return np.linalg.norm(pred - truth, axis=1)

    @staticmethod
    def _cohens_dz(diff: np.ndarray) -> float:
        """配对设计效应量 Cohen's d_z"""
        if len(diff) < 2:
            return float('nan')
        std = np.std(diff, ddof=1)
        if std == 0:
            return float('nan')
        return float(np.mean(diff) / std)

    @staticmethod
    def _bootstrap_ci(values: np.ndarray,
                      stat_fn,
                      n_boot: int = 2000,
                      alpha: float = 0.05,
                      seed: int = 26) -> Tuple[float, float]:
        """Bootstrap 置信区间"""
        if len(values) < 2:
            return float('nan'), float('nan')

        rng = np.random.default_rng(seed)
        n = len(values)
        stats = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            sample = values[rng.integers(0, n, size=n)]
            stats[i] = stat_fn(sample)

        low = float(np.percentile(stats, 100 * (alpha / 2)))
        high = float(np.percentile(stats, 100 * (1 - alpha / 2)))
        return low, high

    def run_statistical_tests(self) -> pd.DataFrame:
        """对 PIRNN-AKF 与其他方法进行配对 t 检验，并补充效应量与95%CI"""
        print("\n" + "="*70)
        print("生成统计显著性检验 (Paired t-test)")
        print("="*70)

        baseline = 'PIRNN-AKF (Ours)'
        if baseline not in self.predictions:
            print("  ⚠️ 缺少 PIRNN-AKF 预测结果，跳过统计检验")
            return pd.DataFrame()

        ours_error = self._get_vector_errors(baseline)
        rows = []
        for method in ['PX4-EKF2', 'Vanilla GRU']:
            if method not in self.predictions:
                continue

            other_error = self._get_vector_errors(method)
            if len(other_error) == 0 or len(other_error) != len(ours_error):
                continue

            t_stat, p_value = ttest_rel(other_error, ours_error)
            diff = other_error - ours_error
            mean_diff = float(np.mean(diff))
            cohens_d = self._cohens_dz(diff)
            mean_ci_low, mean_ci_high = self._bootstrap_ci(diff, np.mean)
            d_ci_low, d_ci_high = self._bootstrap_ci(diff, self._cohens_dz)

            rows.append({
                'Comparison': f'{baseline} vs {method}',
                'Mean Error Diff (m/s)': mean_diff,
                'Mean Diff 95% CI Low': mean_ci_low,
                'Mean Diff 95% CI High': mean_ci_high,
                't-statistic': float(t_stat),
                'p-value': float(p_value),
                "Cohen's d_z": cohens_d,
                "d_z 95% CI Low": d_ci_low,
                "d_z 95% CI High": d_ci_high,
                'Significant(p<0.001)': bool(p_value < 0.001)
            })

        df = pd.DataFrame(rows)
        if len(df) == 0:
            print("  ⚠️ 无可用对比方法，未生成统计表")
            return df

        self.statistical_results = {row['Comparison']: row for _, row in df.iterrows()}

        csv_path = os.path.join(self.output_dir, 'ttest_results.csv')
        df.to_csv(csv_path, index=False)
        print("\n" + df.to_string(index=False))
        print(f"\n  ✓ t 检验结果已保存: {csv_path}")

        return df
    
    def plot_time_series_comparison(self, sample_range: Tuple[int, int] = (0, 1000)):
        """绘制时间序列对比图 (Figure 4)"""
        print("\n" + "="*70)
        print("生成 Figure 4: Time Series Comparison")
        print("="*70)
        
        start, end = sample_range
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # 时间轴
        dt = 1.0 / self.config['data']['sampling_rate']
        time = np.arange(end - start) * dt
        
        colors = {
            'Ground Truth': 'black',
            'PX4-EKF2': 'red',
            'Vanilla GRU': 'blue',
            'PIRNN-AKF (Ours)': 'purple'
        }
        
        linestyles = {
            'Ground Truth': '-',
            'PX4-EKF2': '--',
            'Vanilla GRU': ':',
            'PIRNN-AKF (Ours)': '-'
        }
        
        linewidths = {
            'Ground Truth': 2.0,
            'PX4-EKF2': 1.0,
            'Vanilla GRU': 1.0,
            'PIRNN-AKF (Ours)': 1.5
        }
        
        component_names = ['North Wind (m/s)', 'East Wind (m/s)', 'Down Wind (m/s)']
        
        for i, (ax, name) in enumerate(zip(axes, component_names)):
            for method, pred in self.predictions.items():
                ax.plot(time, pred[start:end, i], 
                       color=colors.get(method, 'gray'),
                       linestyle=linestyles.get(method, '-'),
                       linewidth=linewidths.get(method, 1.0),
                       label=method, alpha=0.8)
            
            ax.set_ylabel(name, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
        
        axes[-1].set_xlabel('Time (s)', fontsize=11)
        
        plt.suptitle('Figure 4: Wind Estimation Time Series Comparison', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'figure4_time_series.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"  ✓ Figure 4 已保存: {save_path}")
        plt.close()
    
    def plot_gust_response(self, gust_threshold: float = 2.0, window_size: int = 50):
        """
        绘制阵风响应分析图 (Figure 5)
        分析各方法在风速突变时的响应特性
        
        Args:
            gust_threshold: 阵风检测阈值 (m/s变化率)
            window_size: 响应窗口大小
        """
        print("\n" + "="*70)
        print("生成 Figure 5: Gust Response Analysis")
        print("="*70)
        
        ground_truth = self.predictions.get('Ground Truth')
        if ground_truth is None:
            print("  ⚠️ 无真值数据，跳过阵风分析")
            return
        
        # 计算风速变化率（检测阵风事件）
        wind_magnitude = np.linalg.norm(ground_truth, axis=1)
        wind_change_rate = np.abs(np.diff(wind_magnitude))
        
        # 检测阵风事件（变化率超过阈值）
        gust_indices = np.where(wind_change_rate > gust_threshold)[0]
        
        if len(gust_indices) == 0:
            print(f"  ⚠️ 未检测到阵风事件（阈值={gust_threshold} m/s）")
            # 降低阈值重试
            gust_threshold = np.percentile(wind_change_rate, 95)
            gust_indices = np.where(wind_change_rate > gust_threshold)[0]
            print(f"  使用自适应阈值: {gust_threshold:.2f} m/s")
        
        print(f"  检测到 {len(gust_indices)} 个阵风事件")
        
        # 选择几个代表性阵风事件
        # 过滤掉边界附近的事件
        valid_gusts = gust_indices[
            (gust_indices > window_size) & 
            (gust_indices < len(wind_magnitude) - window_size)
        ]
        
        if len(valid_gusts) < 3:
            print("  ⚠️ 有效阵风事件不足，使用全部事件")
            selected_gusts = valid_gusts
        else:
            # 选择强度最大的3个事件
            gust_intensities = wind_change_rate[valid_gusts]
            top_indices = np.argsort(gust_intensities)[-3:]
            selected_gusts = valid_gusts[top_indices]
        
        # 绘制阵风响应图
        n_events = min(len(selected_gusts), 3)
        if n_events == 0:
            print("  ⚠️ 无有效阵风事件可绘制")
            return
        
        fig, axes = plt.subplots(n_events, 1, figsize=(12, 4 * n_events))
        if n_events == 1:
            axes = [axes]
        
        dt = 1.0 / self.config['data']['sampling_rate']
        
        colors = {
            'Ground Truth': 'black',
            'PX4-EKF2': 'red',
            'Vanilla GRU': 'blue',
            'PIRNN-AKF (Ours)': 'purple'
        }
        
        for idx, (ax, gust_idx) in enumerate(zip(axes, selected_gusts)):
            start = gust_idx - window_size
            end = gust_idx + window_size
            time = np.arange(-window_size, window_size) * dt
            
            # 绘制风速大小
            for method, pred in self.predictions.items():
                mag = np.linalg.norm(pred[start:end], axis=1)
                ax.plot(time, mag, 
                       color=colors.get(method, 'gray'),
                       linewidth=2 if method in ['Ground Truth', 'PIRNN-AKF (Ours)'] else 1,
                       label=method, alpha=0.8)
            
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Gust Event')
            ax.set_ylabel('Wind Magnitude (m/s)', fontsize=11)
            ax.set_title(f'Gust Event {idx+1} (t={gust_idx * dt:.1f}s)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
        
        axes[-1].set_xlabel('Time relative to gust (s)', fontsize=11)
        
        plt.suptitle('Figure 5: Gust Response Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'figure5_gust_response.svg')
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"  ✓ Figure 5 已保存: {save_path}")
        plt.close()
        
        # 计算阵风响应统计
        self.compute_gust_response_metrics(selected_gusts, window_size)
    
    def compute_gust_response_metrics(self, gust_indices: np.ndarray, window_size: int):
        """计算阵风响应指标"""
        print("\n  【阵风响应指标】")
        
        ground_truth = self.predictions['Ground Truth']
        
        for method, pred in self.predictions.items():
            if method == 'Ground Truth':
                continue
            
            delays = []
            overshoots = []
            
            for gust_idx in gust_indices:
                start = max(0, gust_idx - window_size)
                end = min(len(pred), gust_idx + window_size)
                
                gt_window = np.linalg.norm(ground_truth[start:end], axis=1)
                pred_window = np.linalg.norm(pred[start:end], axis=1)
                
                # 计算响应延迟（互相关峰值位置）
                correlation = np.correlate(gt_window - gt_window.mean(), 
                                          pred_window - pred_window.mean(), mode='same')
                delay = (len(correlation) // 2 - np.argmax(correlation))
                delays.append(delay)
                
                # 计算过冲
                gt_change = np.max(gt_window) - np.min(gt_window)
                pred_change = np.max(pred_window) - np.min(pred_window)
                if gt_change > 0:
                    overshoot = (pred_change - gt_change) / gt_change * 100
                    overshoots.append(overshoot)
            
            avg_delay = np.mean(delays) * (1.0 / self.config['data']['sampling_rate'])
            avg_overshoot = np.mean(overshoots) if overshoots else 0
            
            print(f"  {method}:")
            print(f"    平均响应延迟: {avg_delay*1000:.1f} ms")
            print(f"    平均过冲: {avg_overshoot:.1f}%")
    
    def save_results(self):
        """保存所有结果"""
        results_path = os.path.join(self.output_dir, 'comparison_results.pkl')
        
        with open(results_path, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'predictions': self.predictions,
                'statistical_results': self.statistical_results,
                'config': self.config,
                'timestamp': self.timestamp
            }, f)
        
        print(f"\n  ✓ 结果已保存: {results_path}")
    
    def print_summary(self):
        """打印对比摘要"""
        print("\n" + "="*70)
        print("  对比评估摘要")
        print("="*70)
        
        for method, metrics in self.results.items():
            print(f"\n【{method}】")
            print(f"  总体RMSE: {metrics['total_rmse']:.3f} m/s")
            print(f"  总体MAE: {metrics['total_mae']:.3f} m/s")
            print(f"  向量误差: {metrics['vector_error_mean']:.3f} ± {metrics['vector_error_std']:.3f} m/s")
        
        print("\n" + "="*70)


def main():
    print("="*70)
    print(" 多方法对比评估框架")
    print(" Methods: PX4-EKF2, Vanilla GRU, PIRNN-AKF")
    print("="*70)
    
    try:
        # 创建对比框架
        framework = ComparisonFramework()
        
        # 加载测试数据
        X_test, y_test = framework.load_test_data()
        
        # 运行所有评估
        framework.run_all_evaluations(X_test, y_test)
        
        # 生成Table 1
        framework.generate_table1()

        # 统计显著性检验
        framework.run_statistical_tests()
        
        # 生成Figure 4
        framework.plot_time_series_comparison()
        
        # 生成Figure 5 (阵风响应分析)
        framework.plot_gust_response()
        
        # 保存结果
        framework.save_results()
        
        # 打印摘要
        framework.print_summary()
        
        print("\n" + "="*70)
        print("✅ 多方法对比评估完成！")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
