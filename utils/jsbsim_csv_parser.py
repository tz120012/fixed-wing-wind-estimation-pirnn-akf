"""
JSBSim CSV数据解析器 v2.5
专为实际CSV格式定制 - 只处理现有数据，不添加计算列
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob


class JSBSimCSVParser:
    """JSBSim CSV数据解析器 - 适配实际CSV格式"""
    
    FT_TO_M = 0.3048  # 英尺 → 米
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        
        # 根据您实际的CSV列名映射
        self.column_mapping = {
            # 时间
            '/fdm/jsbsim/simulation/sim-time-sec': 'timestamp',
            
            # 风场真值（训练目标）
            '/fdm/jsbsim/atmosphere/wind-north-fps': 'wind_north',
            '/fdm/jsbsim/atmosphere/wind-east-fps': 'wind_east',
            '/fdm/jsbsim/atmosphere/wind-down-fps': 'wind_down',
            
            # 速度
            '/fdm/jsbsim/velocities/vc-fps': 'calibrated_airspeed',  # 校准空速
            '/fdm/jsbsim/velocities/vtrue-fps': 'airspeed',  # 真空速
            '/fdm/jsbsim/velocities/vg-fps': 'ground_speed',  # 地速
            
            # 地速分量 (NED)
            '/fdm/jsbsim/velocities/v-north-fps': 'vel_n',
            '/fdm/jsbsim/velocities/v-east-fps': 'vel_e',
            '/fdm/jsbsim/velocities/v-down-fps': 'vel_d',
            
            # 位置
            '/fdm/jsbsim/position/h-agl-ft': 'altitude',
            '/fdm/jsbsim/position/lat-geod-deg': 'latitude',
            '/fdm/jsbsim/position/long-gc-deg': 'longitude',
            
            # 姿态角
            '/fdm/jsbsim/attitude/pitch-rad': 'pitch',
            '/fdm/jsbsim/attitude/roll-rad': 'roll',
            '/fdm/jsbsim/attitude/psi-rad': 'yaw',
            
            # 角速度
            '/fdm/jsbsim/velocities/p-rad_sec': 'gyro_x',
            '/fdm/jsbsim/velocities/q-rad_sec': 'gyro_y',
            '/fdm/jsbsim/velocities/r-rad_sec': 'gyro_z',
            
            # 控制指令
            '/fdm/jsbsim/fcs/aileron-cmd-norm': 'aileron',
            '/fdm/jsbsim/fcs/elevator-cmd-norm': 'elevator',
            '/fdm/jsbsim/fcs/throttle-cmd-norm': 'throttle',
            '/fdm/jsbsim/fcs/rudder-cmd-norm': 'rudder',
        }
    
    def parse(self, csv_path, validate=True):
        """解析CSV文件"""
        if self.verbose:
            print(f"正在解析: {os.path.basename(csv_path)}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"文件不存在: {csv_path}")
        
        # 读取CSV
        df = pd.read_csv(csv_path)
        
        if self.verbose:
            print(f"  原始数据: {df.shape[0]} 行 × {df.shape[1]} 列")
        
        # 删除Time列（如果存在）
        if 'Time' in df.columns:
            df = df.drop(columns=['Time'])
        
        # 重命名列
        df_renamed = df.rename(columns=self.column_mapping)
        
        # 单位转换
        df_renamed = self._convert_units(df_renamed)
        
        # 数据验证
        if validate:
            df_renamed = self._validate_data(df_renamed)
        
        if self.verbose:
            print(f"  ✓ 解析完成: {df_renamed.shape[0]} 行有效数据\n")
        
        return df_renamed
    
    def parse_batch(self, input_paths, merge=True):
        """批量解析多个CSV文件"""
        # 处理通配符
        if isinstance(input_paths, str):
            if '*' in input_paths or '?' in input_paths:
                input_paths = glob.glob(input_paths)
            else:
                input_paths = [input_paths]
        
        if not input_paths:
            raise ValueError("未找到匹配的文件")
        
        print("\n" + "="*70)
        print(f"  批量处理：找到 {len(input_paths)} 个文件")
        print("="*70 + "\n")
        
        dataframes = []
        successful = []
        
        for i, path in enumerate(input_paths, 1):
            print(f"[{i}/{len(input_paths)}] ", end='')
            try:
                df = self.parse(path, validate=True)
                
                # 添加文件来源标记
                df['source_file'] = os.path.basename(path)
                df['flight_id'] = i
                
                dataframes.append(df)
                successful.append(path)
                
            except Exception as e:
                print(f"  ❌ 解析失败: {e}\n")
        
        print("\n" + "="*70)
        print(f"  ✓ 成功处理: {len(successful)} 个文件")
        print("="*70 + "\n")
        
        if not dataframes:
            raise ValueError("没有成功解析任何文件")
        
        if merge:
            merged_df = self._merge_dataframes(dataframes)
            return merged_df
        else:
            return dataframes
    
    def _merge_dataframes(self, dataframes):
        """合并多个DataFrame，调整时间戳使其连续"""
        if not dataframes:
            return pd.DataFrame()
        
        merged_dfs = []
        current_time = 0
        
        for i, df in enumerate(dataframes):
            df_copy = df.copy()
            
            if i == 0:
                # 第一个文件：归零
                time_offset = df_copy['timestamp'].min()
                df_copy['timestamp'] = df_copy['timestamp'] - time_offset
                current_time = df_copy['timestamp'].max()
            else:
                # 后续文件：接续
                time_offset = df_copy['timestamp'].min()
                df_copy['timestamp'] = df_copy['timestamp'] - time_offset + current_time
                current_time = df_copy['timestamp'].max()
            
            merged_dfs.append(df_copy)
        
        merged_df = pd.concat(merged_dfs, ignore_index=True)
        return merged_df
    
    def _convert_units(self, df):
        """单位转换"""
        df = df.copy()
        
        # 时间戳：秒 → 微秒
        if 'timestamp' in df.columns:
            df['timestamp'] = (df['timestamp'] * 1e6).astype(np.int64)
        
        # 速度：ft/s → m/s
        velocity_cols = [
            'wind_north', 'wind_east', 'wind_down',
            'vel_n', 'vel_e', 'vel_d',
            'airspeed', 'ground_speed', 'calibrated_airspeed'
        ]
        for col in velocity_cols:
            if col in df.columns:
                df[col] = df[col] * self.FT_TO_M
        
        # 高度：ft → m
        if 'altitude' in df.columns:
            df['altitude'] = df['altitude'] * self.FT_TO_M
        
        return df
    
    def _validate_data(self, df):
        """数据验证和清洗"""
        n_before = len(df)
        
        # 删除NaN和Inf
        df = df.dropna()
        df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]
        
        # 检查空速合理性
        if 'airspeed' in df.columns:
            invalid = (df['airspeed'] < 0) | (df['airspeed'] > 100)
            df = df[~invalid]
        
        # 检查风速合理性
        if all(col in df.columns for col in ['wind_north', 'wind_east', 'wind_down']):
            wind_mag = np.sqrt(df['wind_north']**2 + df['wind_east']**2 + df['wind_down']**2)
            invalid = wind_mag > 50
            df = df[~invalid]
        
        # 确保时间戳单调
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.drop_duplicates(subset=['timestamp'], keep='first')
        
        df = df.reset_index(drop=True)
        return df
    
    def print_statistics(self, df):
        """打印统计摘要"""
        print("\n" + "="*70)
        print("  数据统计摘要")
        print("="*70)
        
        # 时间信息
        duration = (df['timestamp'].max() - df['timestamp'].min()) / 1e6
        print(f"\n⏱️  时间信息:")
        print(f"  总采样点数: {len(df):,}")
        print(f"  总持续时间: {duration:.2f}s ({duration/60:.2f} 分钟)")
        
        # 计算采样率
        time_diffs = np.diff(df['timestamp'].values)
        avg_dt = np.mean(time_diffs) / 1e6
        sampling_rate = 1.0 / avg_dt if avg_dt > 0 else 0
        print(f"  平均采样率: {sampling_rate:.1f} Hz")
        
        # 风速统计
        print(f"\n🌬️  风速统计:")
        self._print_stat("北向风", df['wind_north'], unit="m/s")
        self._print_stat("东向风", df['wind_east'], unit="m/s")
        self._print_stat("地向风", df['wind_down'], unit="m/s")
        
        wind_mag = np.sqrt(df['wind_north']**2 + df['wind_east']**2 + df['wind_down']**2)
        self._print_stat("风速大小", wind_mag, unit="m/s")
        
        # 飞行状态统计
        print(f"\n✈️  飞行状态:")
        self._print_stat("真空速", df['airspeed'], unit="m/s")
        self._print_stat("地速", df['ground_speed'], unit="m/s")
        self._print_stat("高度", df['altitude'], unit="m")
        
        # 姿态统计
        print(f"\n📐 姿态:")
        self._print_stat("横滚角", df['roll'], unit="°", rad_to_deg=True)
        self._print_stat("俯仰角", df['pitch'], unit="°", rad_to_deg=True)
        self._print_stat("航向角", df['yaw'], unit="°", rad_to_deg=True)
        
        # 控制输入统计
        print(f"\n🎮 控制输入:")
        self._print_stat("副翼", df['aileron'])
        self._print_stat("升降舵", df['elevator'])
        self._print_stat("方向舵", df['rudder'])
        self._print_stat("油门", df['throttle'])
        
        # 如果有多个飞行文件
        if 'flight_id' in df.columns:
            print(f"\n📂 各文件详情:")
            for flight_id in sorted(df['flight_id'].unique()):
                flight_data = df[df['flight_id'] == flight_id]
                source = flight_data['source_file'].iloc[0]
                duration_i = (flight_data['timestamp'].max() - flight_data['timestamp'].min()) / 1e6
                wind_avg = np.sqrt(flight_data['wind_north']**2 + 
                                 flight_data['wind_east']**2 + 
                                 flight_data['wind_down']**2).mean()
                print(f"  Flight {flight_id} ({source}):")
                print(f"    数据点: {len(flight_data):,} | 时长: {duration_i:.1f}s | 平均风速: {wind_avg:.2f} m/s")
        
        print("\n" + "="*70 + "\n")
    
    def _print_stat(self, name, series, unit="", rad_to_deg=False):
        """打印统计信息"""
        if rad_to_deg:
            series = np.rad2deg(series)
        
        mean = series.mean()
        std = series.std()
        min_val = series.min()
        max_val = series.max()
        
        print(f"  {name:12s}: {mean:7.2f} ± {std:5.2f} {unit:4s}  "
              f"范围: [{min_val:7.2f}, {max_val:7.2f}]")
    
    def export_csv(self, df, output_path):
        """导出处理后的数据"""
        if self.verbose:
            print(f"导出数据到: {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"  ✓ 导出完成: {len(df):,} 行 × {len(df.columns)} 列\n")
    
    def _get_default_plot_dir(self):
        """返回默认的图片保存目录 (data/raw)。"""
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw'))

    def visualize(self, df, save_path=None):
        """生成7图可视化"""
        print("\n" + "="*70)
        print("  生成7图可视化")
        print("="*70)
        
        # 时间轴（秒）
        time_sec = df['timestamp'].values / 1e6
        
        # 创建图形
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # 检查是否有多个飞行
        has_multiple = 'flight_id' in df.columns
        if has_multiple:
            flight_ids = sorted(df['flight_id'].unique())
            colors = plt.cm.tab10(np.linspace(0, 1, len(flight_ids)))
        
        # ===== 图1: 风速分量 =====
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_sec, df['wind_north'], 'r-', linewidth=1.5, label='North', alpha=0.8)
        ax1.plot(time_sec, df['wind_east'], 'g-', linewidth=1.5, label='East', alpha=0.8)
        ax1.plot(time_sec, df['wind_down'], 'b-', linewidth=1.5, label='Down', alpha=0.8)
        ax1.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Wind Speed (m/s)', fontsize=11, fontweight='bold')
        ax1.set_title('1. Wind Components (NED)', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # ===== 图2: 地速分量 =====
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_sec, df['vel_n'], 'r-', linewidth=1.5, label='North', alpha=0.8)
        ax2.plot(time_sec, df['vel_e'], 'g-', linewidth=1.5, label='East', alpha=0.8)
        ax2.plot(time_sec, df['vel_d'], 'b-', linewidth=1.5, label='Down', alpha=0.8)
        ax2.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Ground Velocity (m/s)', fontsize=11, fontweight='bold')
        ax2.set_title('2. Ground Velocity (NED)', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # ===== 图3: 空速 vs 地速 =====
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(time_sec, df['airspeed'], 'b-', linewidth=2, label='Airspeed', alpha=0.9)
        ax3.plot(time_sec, df['ground_speed'], 'r--', linewidth=2, label='Ground Speed', alpha=0.9)
        if 'calibrated_airspeed' in df.columns:
            ax3.plot(time_sec, df['calibrated_airspeed'], 'g-.', linewidth=2, label='Calibrated Airspeed', alpha=0.9)
        ax3.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Speed (m/s)', fontsize=11, fontweight='bold')
        ax3.set_title('3. Airspeed vs Ground Speed', fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # ===== 图4: 姿态角 =====
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(time_sec, np.rad2deg(df['roll']), 'r-', linewidth=1.5, label='Roll', alpha=0.8)
        ax4.plot(time_sec, np.rad2deg(df['pitch']), 'g-', linewidth=1.5, label='Pitch', alpha=0.8)
        ax4.plot(time_sec, np.rad2deg(df['yaw']), 'b-', linewidth=1.5, label='Yaw', alpha=0.8)
        ax4.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Attitude (°)', fontsize=11, fontweight='bold')
        ax4.set_title('4. Attitude Angles', fontsize=13, fontweight='bold')
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # ===== 图5: 角速度 =====
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(time_sec, np.rad2deg(df['gyro_x']), 'r-', linewidth=1.5, label='p (Roll rate)', alpha=0.8)
        ax5.plot(time_sec, np.rad2deg(df['gyro_y']), 'g-', linewidth=1.5, label='q (Pitch rate)', alpha=0.8)
        ax5.plot(time_sec, np.rad2deg(df['gyro_z']), 'b-', linewidth=1.5, label='r (Yaw rate)', alpha=0.8)
        ax5.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Angular Velocity (°/s)', fontsize=11, fontweight='bold')
        ax5.set_title('5. Angular Velocities', fontsize=13, fontweight='bold')
        ax5.legend(loc='best', fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # ===== 图6: 控制输入 =====
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(time_sec, df['aileron'], 'r-', linewidth=1.5, label='Aileron', alpha=0.8)
        ax6.plot(time_sec, df['elevator'], 'g-', linewidth=1.5, label='Elevator', alpha=0.8)
        ax6.plot(time_sec, df['rudder'], 'b-', linewidth=1.5, label='Rudder', alpha=0.8)
        ax6.plot(time_sec, df['throttle'], 'm-', linewidth=1.5, label='Throttle', alpha=0.8)
        ax6.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Control Input', fontsize=11, fontweight='bold')
        ax6.set_title('6. Control Inputs', fontsize=13, fontweight='bold')
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(-1.1, 1.1)
        
        # ===== 图7: 风速大小（多文件分段显示）=====
        ax7 = fig.add_subplot(gs[2, :])
        wind_mag = np.sqrt(df['wind_north']**2 + df['wind_east']**2 + df['wind_down']**2)
        
        if has_multiple:
            for i, flight_id in enumerate(flight_ids):
                flight_data = df[df['flight_id'] == flight_id]
                t = flight_data['timestamp'].values / 1e6
                w = np.sqrt(flight_data['wind_north']**2 + 
                          flight_data['wind_east']**2 + 
                          flight_data['wind_down']**2)
                source = flight_data['source_file'].iloc[0]
                ax7.plot(t, w, linewidth=2, alpha=0.8, color=colors[i],
                        label=f'Flight {flight_id}: {source}')
            ax7.legend(loc='best', fontsize=9, ncol=min(3, len(flight_ids)))
        else:
            ax7.plot(time_sec, wind_mag, 'b-', linewidth=2, alpha=0.8)
        
        ax7.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Wind Magnitude (m/s)', fontsize=11, fontweight='bold')
        ax7.set_title('7. Wind Magnitude Over Time', fontsize=13, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 总标题
        title = 'JSBSim Flight Data Visualization'
        if has_multiple:
            title += f' ({len(flight_ids)} Flights)'
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        # 保存
        if not save_path:
            default_dir = self._get_default_plot_dir()
            os.makedirs(default_dir, exist_ok=True)
            save_path = os.path.join(default_dir, 'flight_visualization.png')
        else:
            if not os.path.isabs(save_path):
                # 相对路径改为 data/raw 下
                save_path = os.path.join(self._get_default_plot_dir(), save_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ 可视化已保存: {save_path}")
        
        print("="*70 + "\n")
        
        try:
            plt.show()
        except:
            pass


if __name__ == "__main__":
    import sys
    
    print("="*70)
    print(" JSBSim CSV 数据解析器")
    print("="*70)
    
    if len(sys.argv) < 2:
        print("\n用法:")
        print("  # 单文件")
        print("  python jsbsim_csv_parser.py flight_data.csv [--plot] [--export output.csv]")
        print("\n  # 批量处理")
        print("  python jsbsim_csv_parser.py 'data/*.csv' --plot --export merged.csv")
        print("\n示例:")
        print("  python jsbsim_csv_parser.py flight_data.csv --plot")
        print("  python jsbsim_csv_parser.py '../data/*.csv' --plot --export all_flights.csv")
        sys.exit(1)
    
    # 解析参数
    do_plot = '--plot' in sys.argv or '-p' in sys.argv
    
    output_path = None
    if '--export' in sys.argv:
        idx = sys.argv.index('--export')
        if idx + 1 < len(sys.argv):
            output_path = sys.argv[idx + 1]
    
    # 收集输入文件
    input_files = []
    for arg in sys.argv[1:]:
        if arg.startswith('--') or arg.startswith('-'):
            continue
        if arg == output_path:
            continue
        input_files.append(arg)
    
    try:
        parser = JSBSimCSVParser(verbose=True)
        
        # 判断单文件还是多文件
        if len(input_files) == 1 and '*' not in input_files[0]:
            # 单文件
            df = parser.parse(input_files[0])
            parser.print_statistics(df)
            
            if do_plot:
                base_name = os.path.splitext(os.path.basename(input_files[0]))[0]
                viz_path = f"{base_name}_plot.png"
                parser.visualize(df, save_path=viz_path)
        else:
            # 批量
            all_files = []
            for pattern in input_files:
                if '*' in pattern or '?' in pattern:
                    all_files.extend(glob.glob(pattern))
                else:
                    all_files.append(pattern)
            
            df = parser.parse_batch(all_files, merge=True)
            parser.print_statistics(df)
            
            if do_plot:
                parser.visualize(df, save_path='batch_plot.png')
        
        # 导出
        if output_path:
            parser.export_csv(df, output_path)
            print(f"✅ 数据已导出: {output_path}")
        
        print("✅ 处理完成！")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)