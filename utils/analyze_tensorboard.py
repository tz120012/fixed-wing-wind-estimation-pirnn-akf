#!/usr/bin/env python3
"""
TensorBoard 日志分析工具
从 .tfevents 文件中提取训练数据并生成分析报告
支持学术论文风格的可视化输出
"""

import os
import sys
import glob
from collections import defaultdict
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# ==================== 学术论文绘图风格配置 ====================
plt.rcParams.update({
    # 字体设置
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    
    # 线条和标记
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    
    # 图形尺寸和DPI
    'figure.figsize': (7, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    
    # 坐标轴
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    
    # 图例
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'gray',
    
    # LaTeX风格
    'text.usetex': False,  # 设为True需要安装LaTeX
    'mathtext.fontset': 'stix',
})

# 学术配色方案 (来自科学论文常用配色)
COLORS = {
    'train': '#1f77b4',      # 蓝色 - 训练
    'val': '#ff7f0e',        # 橙色 - 验证
    'mae': '#2ca02c',        # 绿色 - MAE
    'rmse': '#d62728',       # 红色 - RMSE
    'lr': '#9467bd',         # 紫色 - 学习率
    'physics': '#8c564b',    # 棕色 - 物理损失
    'wind_mag': '#e377c2',   # 粉色 - 风速大小误差
}

def load_tensorboard_logs(log_dir):
    """加载 TensorBoard 日志"""
    event_files = glob.glob(os.path.join(log_dir, "**/*.tfevents.*"), recursive=True)
    
    if not event_files:
        print(f"❌ 未找到 TensorBoard 日志文件: {log_dir}")
        return None
    
    print(f"✅ 找到 {len(event_files)} 个日志文件")
    
    # 加载事件
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # 0 表示加载所有数据
        }
    )
    ea.Reload()
    
    return ea

def extract_metrics(ea):
    """提取所有指标"""
    metrics = defaultdict(list)
    
    # 获取所有标签
    tags = ea.Tags()['scalars']
    
    for tag in tags:
        events = ea.Scalars(tag)
        for event in events:
            metrics[tag].append({
                'step': event.step,
                'value': event.value,
                'wall_time': event.wall_time
            })
    
    return metrics, tags

def analyze_training(metrics, tags):
    """分析训练情况"""
    print("\n" + "="*80)
    print("📊 TensorBoard 训练数据分析")
    print("="*80)
    
    # 1. 基本信息
    total_epochs = 0
    if 'Metrics/MAE' in metrics:
        total_epochs = len(metrics['Metrics/MAE'])
    
    print(f"\n【基本信息】")
    print(f"  总训练轮数: {total_epochs} epochs")
    print(f"  记录的指标数: {len(tags)}")
    
    # 2. 损失分析
    print(f"\n【损失趋势分析】")
    
    loss_metrics = {
        'Loss/Total_train': '训练总损失',
        'Loss/Total_val': '验证总损失',
        'Loss/Data_train': '训练数据损失',
        'Loss/Data_val': '验证数据损失',
        'Loss/Physics_train': '训练物理损失',
        'Loss/Physics_val': '验证物理损失',
    }
    
    for metric, name in loss_metrics.items():
        if metric in metrics:
            values = [m['value'] for m in metrics[metric]]
            if len(values) > 0:
                print(f"  {name}:")
                print(f"    初始: {values[0]:.4f}")
                print(f"    最终: {values[-1]:.4f}")
                print(f"    最小: {min(values):.4f} (Epoch {values.index(min(values)) + 1})")
                if len(values) > 1:
                    trend = "↓" if values[-1] < values[0] else "↑"
                    change = (values[-1] - values[0]) / values[0] * 100
                    print(f"    变化: {trend} {abs(change):.1f}%")
    
    # 3. 验证指标分析
    print(f"\n【验证指标分析】")
    
    eval_metrics = {
        'Metrics/MAE': 'MAE (m/s)',
        'Metrics/RMSE': 'RMSE (m/s)',
        'Metrics/WindMagError': '风速大小误差 (m/s)',
    }
    
    for metric, name in eval_metrics.items():
        if metric in metrics:
            values = [m['value'] for m in metrics[metric]]
            if len(values) > 0:
                best_idx = values.index(min(values))
                print(f"  {name}:")
                print(f"    最佳: {min(values):.3f} (Epoch {best_idx + 1})")
                print(f"    最新: {values[-1]:.3f} (Epoch {len(values)})")
                print(f"    平均: {np.mean(values):.3f}")
                print(f"    标准差: {np.std(values):.3f}")
    
    # 4. 学习率变化
    if 'Training/LearningRate' in metrics:
        lr_values = [m['value'] for m in metrics['Training/LearningRate']]
        print(f"\n【学习率】")
        print(f"  初始: {lr_values[0]:.6f}")
        print(f"  最终: {lr_values[-1]:.6f}")
        if len(set(lr_values)) > 1:
            print(f"  ⚠️  学习率发生了调整")
    
    # 5. 自适应参数分析
    print(f"\n【自适应参数】")
    
    adaptive_params = {
        'AdaptiveParams/AlphaQ_N': 'α_Q (N方向)',
        'AdaptiveParams/AlphaQ_E': 'α_Q (E方向)',
        'AdaptiveParams/AlphaQ_D': 'α_Q (D方向)',
        'AdaptiveParams/Beta_GPS': 'β (GPS)',
        'AdaptiveParams/Beta_TAS': 'β (TAS)',
        'AdaptiveParams/Beta_ATT': 'β (ATT)',
    }
    
    for metric, name in adaptive_params.items():
        if metric in metrics:
            values = [m['value'] for m in metrics[metric]]
            if len(values) > 0:
                print(f"  {name}: {values[0]:.2f} → {values[-1]:.2f}")
    
    # 6. 过拟合诊断
    print(f"\n【过拟合诊断】")
    
    if 'Loss/Total_train' in metrics and 'Loss/Total_val' in metrics:
        train_loss = [m['value'] for m in metrics['Loss/Total_train']]
        val_loss = [m['value'] for m in metrics['Loss/Total_val']]
        
        if len(train_loss) > 0 and len(val_loss) > 0:
            gap = val_loss[-1] / train_loss[-1]
            print(f"  训练损失: {train_loss[-1]:.4f}")
            print(f"  验证损失: {val_loss[-1]:.4f}")
            print(f"  验证/训练比: {gap:.2f}x")
            
            if gap > 5:
                print(f"  ⚠️  严重过拟合！验证损失是训练损失的 {gap:.1f} 倍")
            elif gap > 3:
                print(f"  ⚠️  中度过拟合，验证损失是训练损失的 {gap:.1f} 倍")
            elif gap > 2:
                print(f"  ⚠️  轻度过拟合，验证损失是训练损失的 {gap:.1f} 倍")
            else:
                print(f"  ✅ 过拟合控制良好")
    
    # 7. 收敛趋势
    print(f"\n【收敛趋势】")
    
    if 'Metrics/MAE' in metrics:
        mae_values = [m['value'] for m in metrics['Metrics/MAE']]
        
        if len(mae_values) >= 5:
            # 计算最近5个epoch的平均改善
            recent_5 = mae_values[-5:]
            improvement = recent_5[0] - recent_5[-1]
            
            print(f"  最近5个epoch MAE变化: {recent_5[0]:.3f} → {recent_5[-1]:.3f}")
            
            if abs(improvement) < 0.005:
                print(f"  ⚠️  收敛停滞（变化 < 0.005）")
            elif improvement > 0:
                print(f"  ✅ 持续改善中（改善 {improvement:.3f}）")
            else:
                print(f"  ⚠️  性能退化（退化 {abs(improvement):.3f}）")
    
    print("\n" + "="*80)
    
    return metrics

def plot_training_curves(metrics, save_dir, run_name):
    """
    绘制学术论文风格的训练曲线图
    
    Args:
        metrics: 提取的指标字典
        save_dir: 图片保存目录
        run_name: 实验名称
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # ==================== 图1: 损失曲线 (双Y轴) ====================
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # 左Y轴: 数据损失
    if 'Loss/Data_train' in metrics and 'Loss/Data_val' in metrics:
        train_data = [m['value'] for m in metrics['Loss/Data_train']]
        val_data = [m['value'] for m in metrics['Loss/Data_val']]
        epochs = range(1, len(train_data) + 1)
        
        ax1.plot(epochs, train_data, color=COLORS['train'], 
                 linestyle='-', label='Training Loss', linewidth=1.5)
        ax1.plot(epochs, val_data, color=COLORS['val'], 
                 linestyle='-', label='Validation Loss', linewidth=1.5)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Data Loss (MSE)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.set_ylim(bottom=0)
        
        # 标记最佳点
        best_idx = val_data.index(min(val_data))
        ax1.scatter([best_idx + 1], [val_data[best_idx]], 
                   color=COLORS['val'], s=80, zorder=5, marker='*',
                   label=f'Best Val: {val_data[best_idx]:.4f}')
    
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_title(f'Training and Validation Loss\n({run_name})')
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    loss_path = os.path.join(save_dir, 'loss_curves.png')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight')
    plt.savefig(loss_path.replace('.png', '.pdf'), bbox_inches='tight')  # PDF格式
    plt.close()
    print(f"  ✓ 损失曲线: {loss_path}")
    
    # ==================== 图2: 验证指标曲线 ====================
    fig, ax = plt.subplots(figsize=(8, 5))
    
    has_data = False
    if 'Metrics/MAE' in metrics:
        mae_values = [m['value'] for m in metrics['Metrics/MAE']]
        epochs = range(1, len(mae_values) + 1)
        ax.plot(epochs, mae_values, color=COLORS['mae'], 
               linestyle='-', marker='o', markersize=3,
               label='MAE', linewidth=1.5, markevery=max(1, len(epochs)//20))
        
        # 标记最佳MAE
        best_idx = mae_values.index(min(mae_values))
        ax.scatter([best_idx + 1], [mae_values[best_idx]], 
                  color=COLORS['mae'], s=100, zorder=5, marker='*')
        ax.annotate(f'Best: {mae_values[best_idx]:.3f}', 
                   xy=(best_idx + 1, mae_values[best_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, color=COLORS['mae'])
        has_data = True
    
    if 'Metrics/RMSE' in metrics:
        rmse_values = [m['value'] for m in metrics['Metrics/RMSE']]
        epochs = range(1, len(rmse_values) + 1)
        ax.plot(epochs, rmse_values, color=COLORS['rmse'], 
               linestyle='--', marker='s', markersize=3,
               label='RMSE', linewidth=1.5, markevery=max(1, len(epochs)//20))
        has_data = True
    
    if 'Metrics/WindMagError' in metrics:
        wind_mag = [m['value'] for m in metrics['Metrics/WindMagError']]
        epochs = range(1, len(wind_mag) + 1)
        ax.plot(epochs, wind_mag, color=COLORS['wind_mag'], 
               linestyle='-.', marker='^', markersize=3,
               label='Wind Magnitude Error', linewidth=1.5, markevery=max(1, len(epochs)//20))
        has_data = True
    
    if has_data:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error (m/s)')
        ax.set_title(f'Validation Metrics\n({run_name})')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        metrics_path = os.path.join(save_dir, 'validation_metrics.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.savefig(metrics_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  ✓ 验证指标: {metrics_path}")
    
    # ==================== 图3: 学习率变化 ====================
    if 'Training/LearningRate' in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        
        lr_values = [m['value'] for m in metrics['Training/LearningRate']]
        epochs = range(1, len(lr_values) + 1)
        
        ax.semilogy(epochs, lr_values, color=COLORS['lr'], 
                   linestyle='-', linewidth=2, marker='o', markersize=2,
                   markevery=max(1, len(epochs)//15))
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate (log scale)')
        ax.set_title(f'Learning Rate Schedule\n({run_name})')
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # 标注关键点
        ax.axhline(y=lr_values[0], color='gray', linestyle=':', alpha=0.5,
                  label=f'Initial: {lr_values[0]:.6f}')
        ax.axhline(y=lr_values[-1], color='gray', linestyle='--', alpha=0.5,
                  label=f'Final: {lr_values[-1]:.6f}')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        lr_path = os.path.join(save_dir, 'learning_rate.png')
        plt.savefig(lr_path, dpi=300, bbox_inches='tight')
        plt.savefig(lr_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()
        print(f"  ✓ 学习率曲线: {lr_path}")
    
    # ==================== 图4: 综合仪表板 (2x2) ====================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # (0,0) 损失曲线
    ax = axes[0, 0]
    if 'Loss/Data_train' in metrics and 'Loss/Data_val' in metrics:
        train_data = [m['value'] for m in metrics['Loss/Data_train']]
        val_data = [m['value'] for m in metrics['Loss/Data_val']]
        epochs = range(1, len(train_data) + 1)
        
        ax.plot(epochs, train_data, color=COLORS['train'], 
               linestyle='-', label='Train', linewidth=1.5)
        ax.plot(epochs, val_data, color=COLORS['val'], 
               linestyle='-', label='Val', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Data Loss')
        ax.set_title('(a) Training & Validation Loss')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    # (0,1) MAE曲线
    ax = axes[0, 1]
    if 'Metrics/MAE' in metrics:
        mae_values = [m['value'] for m in metrics['Metrics/MAE']]
        epochs = range(1, len(mae_values) + 1)
        
        ax.plot(epochs, mae_values, color=COLORS['mae'], 
               linestyle='-', linewidth=1.5)
        ax.fill_between(epochs, mae_values, alpha=0.2, color=COLORS['mae'])
        
        best_idx = mae_values.index(min(mae_values))
        ax.scatter([best_idx + 1], [mae_values[best_idx]], 
                  color='red', s=80, zorder=5, marker='*')
        ax.axhline(y=mae_values[best_idx], color='red', linestyle='--', 
                  alpha=0.5, label=f'Best: {mae_values[best_idx]:.3f} m/s')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (m/s)')
        ax.set_title('(b) Mean Absolute Error')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    
    # (1,0) 物理损失
    ax = axes[1, 0]
    if 'Loss/Physics_train' in metrics:
        phy_train = [m['value'] for m in metrics['Loss/Physics_train']]
        phy_val = [m['value'] for m in metrics['Loss/Physics_val']] if 'Loss/Physics_val' in metrics else None
        epochs = range(1, len(phy_train) + 1)
        
        ax.plot(epochs, phy_train, color=COLORS['physics'], 
               linestyle='-', label='Train', linewidth=1.5)
        if phy_val:
            ax.plot(epochs, phy_val, color=COLORS['physics'], 
                   linestyle='--', label='Val', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Physics Loss')
        ax.set_title('(c) Physics Constraint Loss')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    # (1,1) 过拟合分析
    ax = axes[1, 1]
    if 'Loss/Data_train' in metrics and 'Loss/Data_val' in metrics:
        train_data = [m['value'] for m in metrics['Loss/Data_train']]
        val_data = [m['value'] for m in metrics['Loss/Data_val']]
        epochs = range(1, len(train_data) + 1)
        
        # 计算gap比值
        gap_ratio = [v / t if t > 0 else 1 for v, t in zip(val_data, train_data)]
        
        ax.plot(epochs, gap_ratio, color='purple', linewidth=1.5)
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, 
                  label='Ideal (ratio=1)')
        ax.axhline(y=2.0, color='orange', linestyle=':', alpha=0.7,
                  label='Mild overfitting')
        ax.axhline(y=3.0, color='red', linestyle=':', alpha=0.7,
                  label='Moderate overfitting')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Val/Train Loss Ratio')
        ax.set_title('(d) Overfitting Analysis')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(4, max(gap_ratio) * 1.1))
    
    plt.suptitle(f'Training Analysis Dashboard - {run_name}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    dashboard_path = os.path.join(save_dir, 'training_dashboard.png')
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    plt.savefig(dashboard_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  ✓ 综合仪表板: {dashboard_path}")
    
    return save_dir


def main():
    """主函数"""
    # 解析命令行参数
    plot_figures = '--plot' in sys.argv or '-p' in sys.argv
    
    # 移除标志参数，获取目录参数
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    
    if args:
        log_dir = args[0]
    else:
        # 默认使用最新的日志目录
        tensorboard_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'tensorboard_logs'
        )
        
        # 找到最新的run目录
        run_dirs = glob.glob(os.path.join(tensorboard_dir, '*'))
        if not run_dirs:
            print(f"❌ 未找到日志目录: {tensorboard_dir}")
            return
        
        log_dir = max(run_dirs, key=os.path.getmtime)
        print(f"📂 分析目录: {log_dir}")
    
    # 加载日志
    ea = load_tensorboard_logs(log_dir)
    if ea is None:
        return
    
    # 提取指标
    metrics, tags = extract_metrics(ea)
    
    # 分析训练
    analyze_training(metrics, tags)
    
    # 绘制学术风格图表
    if plot_figures or '--plot' in sys.argv:
        print("\n📊 正在生成学术论文风格图表...")
        run_name = os.path.basename(log_dir)
        save_dir = os.path.join(log_dir, 'figures')
        plot_training_curves(metrics, save_dir, run_name)
        print(f"\n✅ 图表已保存到: {save_dir}")
        print("   支持格式: PNG (用于预览), PDF (用于论文)")
    else:
        print("\n💡 提示: 添加 --plot 或 -p 参数可生成学术论文风格图表")
        print(f"   示例: python {sys.argv[0]} {log_dir} --plot")


if __name__ == '__main__':
    main()
