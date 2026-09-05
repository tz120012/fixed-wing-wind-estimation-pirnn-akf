#!/usr/bin/env python3
"""
分析训练数据中的噪声统计特性
用于确定 α_Q 和 β 的合理目标值
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

def analyze_process_noise(data_path):
    """
    分析过程噪声（风速变化率）
    
    目标：确定 α_Q[N], α_Q[E], α_Q[D] 的相对比例
    """
    print("="*70)
    print("  过程噪声分析（Process Noise Analysis）")
    print("="*70)
    
    # 加载数据 - 支持 .npy 和 .npz 格式
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        y_data = data['y_train']  # [N, 7] = [wind_N, wind_E, wind_D, ...]
    elif data_path.endswith('.npy'):
        # 单个.npy文件 - 推断是y_train
        y_data = np.load(data_path)
        print(f"\n加载数据: {data_path}")
        print(f"数据形状: {y_data.shape}")
    else:
        # 假设是目录，查找 y_train.npy
        import os
        if os.path.isdir(data_path):
            y_path = os.path.join(data_path, 'y_train.npy')
            if os.path.exists(y_path):
                y_data = np.load(y_path)
                print(f"\n从目录加载: {y_path}")
                print(f"数据形状: {y_data.shape}")
            else:
                raise FileNotFoundError(f"在目录 {data_path} 中未找到 y_train.npy")
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
    
    # 提取风速
    wind_N = y_data[:, 0]
    wind_E = y_data[:, 1]
    wind_D = y_data[:, 2]
    
    # 计算风速变化率（一阶差分）
    delta_N = np.diff(wind_N)
    delta_E = np.diff(wind_E)
    delta_D = np.diff(wind_D)
    
    # 统计标准差（代表噪声强度）
    std_N = np.std(delta_N)
    std_E = np.std(delta_E)
    std_D = np.std(delta_D)
    
    print(f"\n风速变化率标准差（m/s per step）：")
    print(f"  North: {std_N:.4f}")
    print(f"  East:  {std_E:.4f}")
    print(f"  Down:  {std_D:.4f}")
    
    # 计算相对比例（归一化到最小值）
    min_std = min(std_N, std_E, std_D)
    ratio_N = std_N / min_std
    ratio_E = std_E / min_std
    ratio_D = std_D / min_std
    
    print(f"\n相对噪声比例（归一化）：")
    print(f"  North: {ratio_N:.2f}")
    print(f"  East:  {ratio_E:.2f}")
    print(f"  Down:  {ratio_D:.2f}")
    
    # 建议的 α_Q 目标值
    print(f"\n💡 建议的 α_Q 目标值：")
    print(f"  target_alpha = [{ratio_N:.2f}, {ratio_E:.2f}, {ratio_D:.2f}]")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 第一行：风速时间序列
    axes[0, 0].plot(wind_N, alpha=0.7)
    axes[0, 0].set_title(f'North Wind (std={np.std(wind_N):.3f})')
    axes[0, 0].set_ylabel('Wind Speed (m/s)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(wind_E, alpha=0.7, color='orange')
    axes[0, 1].set_title(f'East Wind (std={np.std(wind_E):.3f})')
    axes[0, 1].set_ylabel('Wind Speed (m/s)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(wind_D, alpha=0.7, color='green')
    axes[0, 2].set_title(f'Down Wind (std={np.std(wind_D):.3f})')
    axes[0, 2].set_ylabel('Wind Speed (m/s)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 第二行：风速变化率分布
    axes[1, 0].hist(delta_N, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title(f'North Δ Wind (std={std_N:.4f})')
    axes[1, 0].set_xlabel('Wind Change (m/s)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(delta_E, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_title(f'East Δ Wind (std={std_E:.4f})')
    axes[1, 1].set_xlabel('Wind Change (m/s)')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].hist(delta_D, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1, 2].set_title(f'Down Δ Wind (std={std_D:.4f})')
    axes[1, 2].set_xlabel('Wind Change (m/s)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = '/root/wind_estimation/process_noise_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 图表已保存到: {save_path}")
    plt.close()
    
    return ratio_N, ratio_E, ratio_D


def analyze_measurement_noise(data_path):
    """
    分析观测噪声（传感器误差）
    
    注意：这需要真值数据，如果没有则跳过
    """
    print("\n" + "="*70)
    print("  观测噪声分析（Measurement Noise Analysis）")
    print("="*70)
    print("\n⚠️  观测噪声分析需要传感器真值数据")
    print("建议：参考传感器数据手册中的精度规格")
    print("\n典型值参考：")
    print("  GPS位置:     ±1.5-3.0 m (水平), ±2.0-5.0 m (垂直)")
    print("  空速传感器:  ±0.5-1.5 m/s")
    print("  IMU姿态:     ±0.5-2.0° (roll/pitch), ±1.0-5.0° (yaw)")
    print("\n💡 建议的 β 目标值（示例）：")
    print("  target_beta = [1.2, 1.0, 0.8]  # GPS/TAS/ATT")
    print("  或更保守:    [1.1, 1.0, 0.9]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='分析训练数据噪声统计，确定 α_Q 和 β 目标值'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='/root/wind_estimation/data/processed',
        help='训练数据路径（可以是.npy文件、.npz文件或包含y_train.npy的目录）'
    )
    
    args = parser.parse_args()
    
    # 分析过程噪声
    try:
        ratio_N, ratio_E, ratio_D = analyze_process_noise(args.data)
        
        print("\n" + "="*70)
        print("  📝 代码修改建议")
        print("="*70)
        print("\n在 src/3_train_pigru.py 中修改：")
        print(f"\ntarget_alpha = torch.tensor([{ratio_N:.2f}, {ratio_E:.2f}, {ratio_D:.2f}], device=alpha_Q.device)")
        print("\n或使用归一化版本（平均值=1.0）：")
        avg = (ratio_N + ratio_E + ratio_D) / 3
        norm_N, norm_E, norm_D = ratio_N/avg, ratio_E/avg, ratio_D/avg
        print(f"target_alpha = torch.tensor([{norm_N:.2f}, {norm_E:.2f}, {norm_D:.2f}], device=alpha_Q.device)")
        
    except Exception as e:
        print(f"\n❌ 分析失败: {e}")
        print("请确保数据文件存在且格式正确")
    
    # 分析观测噪声
    analyze_measurement_noise(args.data)
    
    print("\n" + "="*70)
