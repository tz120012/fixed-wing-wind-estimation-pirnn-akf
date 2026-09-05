
# PX4-SITL + JSBSim 数据集生成完整指南

## 目标

使用 PX4-SITL + JSBSim 框架生成PIRNN-AKF训练所需的完整数据集，包括：
- **训练集**：300次飞行（240次稳态 + 60次阵风）
- **验证集**：50次飞行（40次稳态 + 10次阵风）
- **测试集-ID**：35次飞行（25次稳态 + 10次阵风）
- **测试集-OOD**：15次飞行（15次强阵风/长阵风）

**总计**：400次飞行

---

## 一、环境搭建

### 1.1 安装PX4-Autopilot

```bash
# 1. 克隆PX4源码
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot

# 2. 运行安装脚本（Ubuntu 20.04/22.04）
bash ./Tools/setup/ubuntu.sh

# 3. 重启终端或执行
source ~/.bashrc

# 4. 测试编译
make px4_sitl_default
```

### 1.2 安装JSBSim

```bash
# 方法1：从源码编译（推荐，版本最新）
cd ~
git clone https://github.com/JSBSim-Team/jsbsim.git
cd jsbsim
mkdir build && cd build
cmake ..
make -j4
sudo make install

# 方法2：使用包管理器（简单但版本可能较旧）
sudo apt-get install libjsbsim-dev

# 验证安装
JSBSim --version
```

### 1.3 安装Python依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv ~/pirnn_env
source ~/pirnn_env/bin/activate

# 安装必要的包
pip install --upgrade pip
pip install mavsdk==1.4.2
pip install pymavlink==2.4.37
pip install numpy pandas scipy
pip install matplotlib seaborn
pip install pyulog  # 用于解析PX4日志

# 可选：安装JSBSim Python绑定（用于高级风场控制）
pip install jsbsim
```

### 1.4 测试SITL仿真

```bash
cd ~/PX4-Autopilot

# 启动JSBSim仿真（Rascal固定翼）
make px4_sitl jsbsim_rascal

# 应该看到：
# - PX4 shell启动
# - JSBSim引擎初始化
# - "INFO  [commander] Ready for takeoff!"

# 测试成功后，Ctrl+C停止
```

---

## 二、项目结构搭建

### 2.1 创建工作目录

```bash
cd ~
mkdir -p pirnn_dataset_generation
cd pirnn_dataset_generation

# 创建目录结构
mkdir -p {scripts,data,logs,configs,results}
mkdir -p data/{train,val,test_id,test_ood}
mkdir -p logs/{train,val,test_id,test_ood}
```

**目录说明**：
```
pirnn_dataset_generation/
├── scripts/              # Python脚本
│   ├── flight_controller.py
│   ├── wind_manager.py
│   ├── data_logger.py
│   └── generate_dataset.py
├── data/                 # 生成的飞行数据
│   ├── train/
│   ├── val/
│   ├── test_id/
│   └── test_ood/
├── logs/                 # 仿真日志
├── configs/              # 配置文件
│   ├── flight_plans.json
│   └── wind_profiles.json
└── results/              # 数据分析结果
```

---

## 三、核心脚本编写

### 3.1 风场管理器（wind_manager.py）

```python
"""
wind_manager.py
负责设置和管理仿真中的风场参数
"""

from pymavlink import mavutil
import numpy as np
import time
import random


class WindManager:
    def __init__(self, connection_string='udp:127.0.0.1:14550'):
        """
        初始化风场管理器
        
        Args:
            connection_string: MAVLink连接字符串
        """
        self.master = mavutil.mavlink_connection(connection_string)
        self.master.wait_heartbeat()
        print(f"[WindManager] 已连接到飞控 (系统ID: {self.master.target_system})")
        
    def set_constant_wind(self, speed, direction):
        """
        设置恒定风场
        
        Args:
            speed: 风速 (m/s)
            direction: 风向角 (度, 0度为北)
        """
        # 计算北向和东向分量
        wind_north = speed * np.cos(np.deg2rad(direction))
        wind_east = speed * np.sin(np.deg2rad(direction))
        
        # 设置风速（通过SIM_WIND_SPEED参数）
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            b'SIM_WIND_SPD',
            speed,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        
        # 设置风向（通过SIM_WIND_DIR参数）
        self.master.mav.param_set_send(
            self.master.target_system,
            self.master.target_component,
            b'SIM_WIND_DIR',
            direction,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        
        time.sleep(0.5)  # 等待参数设置生效
        
        print(f"[WindManager] 已设置恒定风: {speed:.1f} m/s @ {direction:.1f}°")
        print(f"              北向分量: {wind_north:.2f} m/s, 东向分量: {wind_east:.2f} m/s")
        
        return wind_north, wind_east
    
    def set_turbulence(self, intensity='light'):
        """
        设置湍流强度
        
        Args:
            intensity: 'none', 'light', 'moderate', 'severe'
        """
        intensity_map = {
            'none': 0.0,
            'light': 1.0,
            'moderate': 2.0,
            'severe': 4.0
        }
        
        value = intensity_map.get(intensity, 1.0)
        
        # 注意：PX4-SITL的SIM_WIND_TURB参数可能因版本而异
        # 某些版本使用 SIM_WIND_T_*
        try:
            self.master.mav.param_set_send(
                self.master.target_system,
                self.master.target_component,
                b'SIM_WIND_TURB',
                value,
                mavutil.mavlink.MAV_PARAM_TYPE_REAL32
            )
            print(f"[WindManager] 已设置湍流强度: {intensity} (值={value})")
        except Exception as e:
            print(f"[WindManager] 警告: 无法设置湍流参数 ({e})")
            print(f"              这可能是正常的，取决于PX4版本")
        
        time.sleep(0.5)
    
    def generate_random_wind(self, speed_range=(6, 14), direction_range=(0, 360)):
        """
        生成随机风场参数
        
        Args:
            speed_range: 风速范围 (m/s)
            direction_range: 风向范围 (度)
            
        Returns:
            (speed, direction) 元组
        """
        speed = random.uniform(*speed_range)
        direction = random.uniform(*direction_range)
        return speed, direction
    
    def inject_cosine_gust(self, magnitude, duration, direction, start_delay=0):
        """
        注入1-cosine阵风（模拟方式，通过快速改变风速）
        
        注意：这是近似实现，因为PX4-SITL不直接支持阵风模型
        真实物理阵风需要修改JSBSim XML或使用JSBSim Python API
        
        Args:
            magnitude: 阵风幅值 (m/s)
            duration: 阵风持续时间 (s)
            direction: 阵风方向 (度)
            start_delay: 延迟开始时间 (s)
        """
        print(f"[WindManager] 准备注入1-cosine阵风:")
        print(f"              幅值={magnitude:.1f} m/s, 持续时间={duration:.1f}s")
        print(f"              方向={direction:.1f}°, 延迟={start_delay:.1f}s")
        
        # 延迟
        if start_delay > 0:
            time.sleep(start_delay)
        
        # 获取当前背景风
        base_speed = 10.0  # 假设背景风速（实际应该从参数读取）
        base_dir = 0.0
        
        # 1-cosine剖面
        dt = 0.1  # 10Hz更新
        steps = int(duration / dt)
        
        for i in range(steps):
            t = i * dt
            # 1-cosine公式
            gust_factor = 0.5 * (1 - np.cos(2 * np.pi * t / duration))
            
            # 叠加阵风
            gust_north = magnitude * gust_factor * np.cos(np.deg2rad(direction))
            gust_east = magnitude * gust_factor * np.sin(np.deg2rad(direction))
            
            base_north = base_speed * np.cos(np.deg2rad(base_dir))
            base_east = base_speed * np.sin(np.deg2rad(base_dir))
            
            total_north = base_north + gust_north
            total_east = base_east + gust_east
            
            total_speed = np.sqrt(total_north**2 + total_east**2)
            total_dir = np.rad2deg(np.arctan2(total_east, total_north))
            
            # 更新风场
            self.set_constant_wind(total_speed, total_dir)
            time.sleep(dt)
        
        # 恢复背景风
        self.set_constant_wind(base_speed, base_dir)
        print(f"[WindManager] 阵风注入完成")


# 测试代码
if __name__ == '__main__':
    wm = WindManager()
    
    # 测试1: 设置恒定风
    wm.set_constant_wind(speed=10.0, direction=45.0)
    time.sleep(5)
    
    # 测试2: 生成随机风
    speed, direction = wm.generate_random_wind()
    wm.set_constant_wind(speed, direction)
    time.sleep(5)
    
    # 测试3: 注入阵风
    wm.inject_cosine_gust(magnitude=6.0, duration=5.0, direction=90.0)
```

---

### 3.2 飞行控制器（flight_controller.py）

```python
"""
flight_controller.py
负责控制无人机执行各种飞行机动
"""

import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw, OffboardError
import numpy as np
import random


class FlightController:
    def __init__(self):
        self.drone = System()
        self.is_connected = False
        
    async def connect(self, system_address="udp://:14540"):
        """连接到PX4"""
        await self.drone.connect(system_address=system_address)
        
        print("[FlightController] 等待连接...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("[FlightController] 已连接!")
                self.is_connected = True
                break
    
    async def arm_and_takeoff(self, altitude):
        """解锁并起飞"""
        print(f"[FlightController] 解锁并起飞到 {altitude:.1f}m")
        
        # 解锁
        await self.drone.action.arm()
        
        # 起飞
        await self.drone.action.set_takeoff_altitude(altitude)
        await self.drone.action.takeoff()
        
        # 等待到达目标高度
        await asyncio.sleep(15)
        
        print(f"[FlightController] 已到达 {altitude:.1f}m")
    
    async def fly_straight_line(self, heading, altitude, speed, duration):
        """
        直线飞行
        
        Args:
            heading: 航向角 (度)
            altitude: 高度 (m)
            speed: 空速 (m/s)
            duration: 持续时间 (s)
        """
        print(f"[FlightController] 直线飞行: 航向={heading:.1f}°, 速度={speed:.1f}m/s, 时长={duration:.1f}s")
        
        # 计算速度分量（NED坐标系）
        vn = speed * np.cos(np.deg2rad(heading))
        ve = speed * np.sin(np.deg2rad(heading))
        vd = 0.0
        
        # 启动offboard模式
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(vn, ve, vd, heading)
        )
        
        try:
            await self.drone.offboard.start()
        except OffboardError as error:
            print(f"[FlightController] 启动offboard失败: {error}")
            return
        
        # 执行直线飞行
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < duration:
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(vn, ve, vd, heading)
            )
            await asyncio.sleep(0.1)
        
        print(f"[FlightController] 直线飞行完成")
    
    async def fly_orbit(self, radius, altitude, direction='cw', duration=180):
        """
        盘旋飞行
        
        Args:
            radius: 盘旋半径 (m)
            altitude: 高度 (m)
            direction: 'cw'(顺时针) 或 'ccw'(逆时针)
            duration: 持续时间 (s)
        """
        print(f"[FlightController] 盘旋飞行: 半径={radius:.1f}m, 方向={direction}, 时长={duration:.1f}s")
        
        # 获取当前位置作为圆心
        async for position in self.drone.telemetry.position():
            center_lat = position.latitude_deg
            center_lon = position.longitude_deg
            break
        
        # 计算角速度
        circumference = 2 * np.pi * radius
        period = circumference / 15.0  # 假设速度15 m/s
        angular_velocity = 2 * np.pi / period
        
        if direction == 'ccw':
            angular_velocity = -angular_velocity
        
        # 执行盘旋
        start_time = asyncio.get_event_loop().time()
        t = 0
        
        while (asyncio.get_event_loop().time() - start_time) < duration:
            angle = angular_velocity * t
            
            # 计算目标位置（NED坐标系）
            north_offset = radius * np.cos(angle)
            east_offset = radius * np.sin(angle)
            
            # 简化的经纬度转换（仅适用于小范围）
            target_lat = center_lat + (north_offset / 111320.0)
            target_lon = center_lon + (east_offset / (111320.0 * np.cos(np.deg2rad(center_lat))))
            
            # 发送位置指令
            await self.drone.action.goto_location(target_lat, target_lon, altitude, 0)
            
            await asyncio.sleep(0.2)
            t += 0.2
        
        print(f"[FlightController] 盘旋飞行完成")
    
    async def fly_figure_eight(self, lobe_radius, orientation, duration=300):
        """
        8字机动
        
        Args:
            lobe_radius: 单个圆的半径 (m)
            orientation: 8字长轴方向 (度)
            duration: 持续时间 (s)
        """
        print(f"[FlightController] 8字机动: 半径={lobe_radius:.1f}m, 方向={orientation:.1f}°")
        
        # 简化实现：依次执行两个盘旋
        await self.fly_orbit(lobe_radius, 100, 'cw', duration=duration/2)
        await self.fly_orbit(lobe_radius, 100, 'ccw', duration=duration/2)
        
        print(f"[FlightController] 8字机动完成")
    
    async def fly_climb_descent(self, h_start, h_end, climb_rate, heading, duration=120):
        """
        爬升/下降机动
        
        Args:
            h_start: 起始高度 (m)
            h_end: 目标高度 (m)
            climb_rate: 爬升率 (m/s, 负值表示下降)
            heading: 水平航向 (度)
            duration: 持续时间 (s)
        """
        print(f"[FlightController] 爬升/下降: {h_start:.1f}m → {h_end:.1f}m, 爬升率={climb_rate:.1f}m/s")
        
        # 计算水平速度
        horizontal_speed = 15.0  # m/s
        vn = horizontal_speed * np.cos(np.deg2rad(heading))
        ve = horizontal_speed * np.sin(np.deg2rad(heading))
        vd = -climb_rate  # NED坐标系，向下为正
        
        # 执行爬升/下降
        start_time = asyncio.get_event_loop().time()
        
        try:
            await self.drone.offboard.start()
        except:
            pass
        
        while (asyncio.get_event_loop().time() - start_time) < duration:
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(vn, ve, vd, heading)
            )
            await asyncio.sleep(0.1)
        
        print(f"[FlightController] 爬升/下降完成")
    
    async def land(self):
        """降落"""
        print(f"[FlightController] 开始降落")
        
        try:
            await self.drone.offboard.stop()
        except:
            pass
        
        await self.drone.action.land()
        await asyncio.sleep(10)
        
        print(f"[FlightController] 已降落")


# 测试代码
async def test_flight():
    fc = FlightController()
    await fc.connect()
    
    # 起飞
    await fc.arm_and_takeoff(altitude=80)
    
    # 直线飞行
    await fc.fly_straight_line(heading=45, altitude=80, speed=15, duration=60)
    
    # 降落
    await fc.land()


if __name__ == '__main__':
    asyncio.run(test_flight())
```

---

### 3.3 数据记录器（data_logger.py）

```python
"""
data_logger.py
负责记录飞行过程中的传感器数据
"""

import asyncio
from mavsdk import System
import json
import time
import numpy as np


class DataLogger:
    def __init__(self, drone):
        """
        初始化数据记录器
        
        Args:
            drone: MAVSDK System对象
        """
        self.drone = drone
        self.data_buffer = []
        self.is_logging = False
        self.start_time = None
        
    async def start_logging(self, duration, output_file):
        """
        开始记录数据
        
        Args:
            duration: 记录时长 (s)
            output_file: 输出文件路径
        """
        print(f"[DataLogger] 开始记录数据，时长={duration:.1f}s")
        
        self.data_buffer = []
        self.is_logging = True
        self.start_time = time.time()
        
        # 启动异步记录任务
        tasks = [
            self._log_position(),
            self._log_velocity(),
            self._log_attitude(),
            self._log_imu(),
            self._log_gps()
        ]
        
        # 运行指定时长
        await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=duration
        )
        
        self.is_logging = False
        
        # 保存数据
        self._save_data(output_file)
        
        print(f"[DataLogger] 数据已保存到 {output_file}")
        print(f"[DataLogger] 共记录 {len(self.data_buffer)} 个数据点")
    
    async def _log_position(self):
        """记录位置数据"""
        async for position in self.drone.telemetry.position():
            if not self.is_logging:
                break
            
            timestamp = time.time() - self.start_time
            
            # 查找或创建当前时间戳的数据条目
            data_entry = self._get_or_create_entry(timestamp)
            data_entry['position'] = {
                'lat': position.latitude_deg,
                'lon': position.longitude_deg,
                'alt_msl': position.absolute_altitude_m,
                'alt_rel': position.relative_altitude_m
            }
            
            await asyncio.sleep(0.02)  # 50Hz
    
    async def _log_velocity(self):
        """记录速度数据"""
        async for velocity in self.drone.telemetry.velocity_ned():
            if not self.is_logging:
                break
            
            timestamp = time.time() - self.start_time
            data_entry = self._get_or_create_entry(timestamp)
            data_entry['velocity_ned'] = {
                'north': velocity.north_m_s,
                'east': velocity.east_m_s,
                'down': velocity.down_m_s
            }
            
            await asyncio.sleep(0.02)
    
    async def _log_attitude(self):
        """记录姿态数据"""
        async for attitude in self.drone.telemetry.attitude_euler():
            if not self.is_logging:
                break
            
            timestamp = time.time() - self.start_time
            data_entry = self._get_or_create_entry(timestamp)
            data_entry['attitude'] = {
                'roll': attitude.roll_deg,
                'pitch': attitude.pitch_deg,
                'yaw': attitude.yaw_deg
            }
            
            await asyncio.sleep(0.02)
    
    async def _log_imu(self):
        """记录IMU数据"""
        async for imu in self.drone.telemetry.imu():
            if not self.is_logging:
                break
            
            timestamp = time.time() - self.start_time
            data_entry = self._get_or_create_entry(timestamp)
            data_entry['imu'] = {
                'acc_x': imu.acceleration_forward_m_s2,
                'acc_y': imu.acceleration_right_m_s2,
                'acc_z': imu.acceleration_down_m_s2,
                'gyro_x': imu.angular_velocity_forward_rad_s,
                'gyro_y': imu.angular_velocity_right_rad_s,
                'gyro_z': imu.angular_velocity_down_rad_s
            }
            
            await asyncio.sleep(0.02)
    
    async def _log_gps(self):
        """记录GPS数据"""
        async for gps in self.drone.telemetry.gps_info():
            if not self.is_logging:
                break
            
            timestamp = time.time() - self.start_time
            data_entry = self._get_or_create_entry(timestamp)
            data_entry['gps'] = {
                'num_satellites': gps.num_satellites,
                'fix_type': gps.fix_type
            }
            
            await asyncio.sleep(0.1)
    
    def _get_or_create_entry(self, timestamp):
        """获取或创建指定时间戳的数据条目"""
        # 查找最近的时间戳（容差0.05秒）
        for entry in self.data_buffer:
            if abs(entry['timestamp'] - timestamp) < 0.05:
                return entry
        
        # 如果没找到，创建新条目
        new_entry = {'timestamp': timestamp}
        self.data_buffer.append(new_entry)
        return new_entry
    
    def _save_data(self, output_file):
        """保存数据到JSON文件"""
        # 按时间戳排序
        self.data_buffer.sort(key=lambda x: x['timestamp'])
        
        with open(output_file, 'w') as f:
            json.dump(self.data_buffer, f, indent=2)


# 测试代码
async def test_logging():
    drone = System()
    await drone.connect(system_address="udp://:14540")
    
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    
    logger = DataLogger(drone)
    await logger.start_logging(duration=30, output_file='test_log.json')


if __name__ == '__main__':
    asyncio.run(test_logging())
```

### 3.4 主生成脚本（generate_dataset.py）- 完整版

```python
"""
generate_dataset.py
主脚本：协调飞行控制、风场管理和数据记录
"""

import asyncio
import subprocess
import time
import random
import numpy as np
import json
from pathlib import Path

from flight_controller import FlightController
from wind_manager import WindManager
from data_logger import DataLogger


class DatasetGenerator:
    def __init__(self, output_dir='./data'):
        self.output_dir = Path(output_dir)
        self.fc = None
        self.wm = None
        self.px4_process = None
        
    def start_px4_sitl(self):
        """启动PX4 SITL仿真"""
        print("[DatasetGenerator] 启动PX4 SITL...")
        
        px4_dir = Path.home() / 'PX4-Autopilot'
        
        # 启动PX4（后台运行）
        self.px4_process = subprocess.Popen(
            ['make', 'px4_sitl', 'jsbsim_rascal'],
            cwd=str(px4_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 等待启动
        time.sleep(10)
        print("[DatasetGenerator] PX4 SITL已启动")
    
    def stop_px4_sitl(self):
        """停止PX4 SITL仿真"""
        print("[DatasetGenerator] 停止PX4 SITL...")
        
        if self.px4_process:
            self.px4_process.terminate()
            self.px4_process.wait(timeout=5)
        
        # 强制杀死残留进程
        subprocess.run(['pkill', '-9', 'px4'], stderr=subprocess.DEVNULL)
        subprocess.run(['pkill', '-9', 'jsbsim'], stderr=subprocess.DEVNULL)
        
        time.sleep(2)
        print("[DatasetGenerator] PX4 SITL已停止")
    
    async def initialize_controllers(self):
        """初始化飞行控制器和风场管理器"""
        self.fc = FlightController()
        await self.fc.connect()
        
        self.wm = WindManager()
        
        print("[DatasetGenerator] 控制器已初始化")
    
    async def execute_single_flight(self, flight_config, output_file):
        """
        执行单次飞行
        
        Args:
            flight_config: 飞行配置字典
            output_file: 输出文件路径
        """
        print(f"\n{'='*60}")
        print(f"[DatasetGenerator] 执行飞行: {flight_config['id']}")
        print(f"{'='*60}")
        
        # 1. 设置风场
        wind_speed = flight_config['wind_speed']
        wind_dir = flight_config['wind_direction']
        self.wm.set_constant_wind(wind_speed, wind_dir)
        
        if flight_config.get('turbulence'):
            self.wm.set_turbulence(flight_config['turbulence'])
        
        # 2. 起飞
        altitude = flight_config['altitude']
        await self.fc.arm_and_takeoff(altitude)
        
        # 3. 创建数据记录器
        logger = DataLogger(self.fc.drone)
        
        # 4. 执行飞行机动（同时记录数据）
        maneuver_type = flight_config['maneuver_type']
        duration = flight_config['duration']
        
        # 启动数据记录任务
        logging_task = asyncio.create_task(
            logger.start_logging(duration, output_file)
        )
        
        # 启动飞行机动任务
        if maneuver_type == 'straight_line':
            flight_task = asyncio.create_task(
                self.fc.fly_straight_line(
                    heading=flight_config['heading'],
                    altitude=altitude,
                    speed=flight_config['speed'],
                    duration=duration
                )
            )
        elif maneuver_type == 'orbit':
            flight_task = asyncio.create_task(
                self.fc.fly_orbit(
                    radius=flight_config['radius'],
                    altitude=altitude,
                    direction=flight_config.get('direction', 'cw'),
                    duration=duration
                )
            )
        elif maneuver_type == 'figure_eight':
            flight_task = asyncio.create_task(
                self.fc.fly_figure_eight(
                    lobe_radius=flight_config['radius'],
                    orientation=flight_config['heading'],
                    duration=duration
                )
            )
        elif maneuver_type == 'climb_descent':
            flight_task = asyncio.create_task(
                self.fc.fly_climb_descent(
                    h_start=altitude,
                    h_end=flight_config['target_altitude'],
                    climb_rate=flight_config['climb_rate'],
                    heading=flight_config['heading'],
                    duration=duration
                )
            )
        else:
            raise ValueError(f"未知的机动类型: {maneuver_type}")
        
        # 5. 如果有阵风，在指定时间注入
        if flight_config.get('gust'):
            gust_config = flight_config['gust']
            gust_task = asyncio.create_task(
                asyncio.sleep(gust_config['start_time'])
            )
            await gust_task
            
            # 注入阵风（在新任务中）
            asyncio.create_task(
                asyncio.to_thread(
                    self.wm.inject_cosine_gust,
                    gust_config['magnitude'],
                    gust_config['duration'],
                    gust_config['direction']
                )
            )
        
        # 6. 等待飞行和记录完成
        await asyncio.gather(flight_task, logging_task)
        
        # 7. 降落
        await self.fc.land()
        
        # 8. 保存元数据
        metadata_file = output_file.replace('.json', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(flight_config, f, indent=2)
        
        print(f"[DatasetGenerator] 飞行完成: {flight_config['id']}")
    
    def generate_flight_config(self, flight_id, dataset_type, config_type):
        """
        生成飞行配置
        
        Args:
            flight_id: 飞行编号
            dataset_type: 'train', 'val', 'test_id', 'test_ood'
            config_type: 'steady' 或 'gust'
        
        Returns:
            飞行配置字典
        """
        config = {
            'id': f"{dataset_type}_{flight_id:04d}",
            'dataset_type': dataset_type,
            'config_type': config_type
        }
        
        # 基本参数随机化
        config['wind_speed'] = random.uniform(6, 14)
        config['wind_direction'] = random.uniform(0, 360)
        config['turbulence'] = random.choice(['light', 'moderate'])
        config['altitude'] = random.uniform(80, 120)
        config['speed'] = random.uniform(12, 18)
        config['heading'] = random.uniform(0, 360)
        
        # 机动类型随机化
        maneuver_types = ['straight_line', 'orbit', 'figure_eight', 'climb_descent']
        config['maneuver_type'] = random.choice(maneuver_types)
        
        # 根据机动类型设置特定参数
        if config['maneuver_type'] == 'straight_line':
            config['duration'] = random.uniform(90, 150)
        
        elif config['maneuver_type'] == 'orbit':
            config['radius'] = random.uniform(80, 150)
            config['direction'] = random.choice(['cw', 'ccw'])
            config['duration'] = random.uniform(120, 180)
        
        elif config['maneuver_type'] == 'figure_eight':
            config['radius'] = random.uniform(60, 100)
            config['duration'] = random.uniform(180, 240)
        
        elif config['maneuver_type'] == 'climb_descent':
            config['target_altitude'] = config['altitude'] + random.uniform(-30, 30)
            config['climb_rate'] = random.uniform(1, 3)
            config['duration'] = random.uniform(90, 150)
        
        # 如果是阵风场景
        if config_type == 'gust':
            # 确定阵风参数范围
            if dataset_type == 'test_ood':
                # OOD: 超出训练范围
                gust_magnitude = random.uniform(9, 12)
                gust_duration = random.uniform(8, 12) if random.random() < 0.5 else random.uniform(3, 6)
            else:
                # ID: 训练范围内
                gust_magnitude = random.uniform(4, 8)
                gust_duration = random.uniform(3, 6)
            
            config['gust'] = {
                'magnitude': gust_magnitude,
                'duration': gust_duration,
                'direction': random.uniform(0, 360),
                'start_time': random.uniform(30, 60)  # 在飞行中段注入
            }
        
        return config
    
    async def generate_dataset_batch(self, dataset_type, num_steady, num_gust):
        """
        生成一批数据
        
        Args:
            dataset_type: 'train', 'val', 'test_id', 'test_ood'
            num_steady: 稳态飞行次数
            num_gust: 阵风飞行次数
        """
        output_subdir = self.output_dir / dataset_type
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        total_flights = num_steady + num_gust
        flight_counter = 0
        
        print(f"\n{'#'*60}")
        print(f"# 开始生成 {dataset_type.upper()} 数据集")
        print(f"# 稳态飞行: {num_steady}次")
        print(f"# 阵风飞行: {num_gust}次")
        print(f"# 总计: {total_flights}次")
        print(f"{'#'*60}\n")
        
        # 生成稳态飞行
        for i in range(num_steady):
            flight_counter += 1
            
            # 生成配置
            config = self.generate_flight_config(flight_counter, dataset_type, 'steady')
            
            # 输出文件
            output_file = output_subdir / f"flight_{flight_counter:04d}.json"
            
            try:
                # 重启SITL（每次飞行独立）
                self.stop_px4_sitl()
                self.start_px4_sitl()
                await self.initialize_controllers()
                
                # 执行飞行
                await self.execute_single_flight(config, str(output_file))
                
                print(f"[进度] {flight_counter}/{total_flights} 完成 ({flight_counter/total_flights*100:.1f}%)")
                
            except Exception as e:
                print(f"[错误] 飞行 {flight_counter} 失败: {e}")
                continue
            
            finally:
                self.stop_px4_sitl()
                await asyncio.sleep(2)
        
        # 生成阵风飞行
        for i in range(num_gust):
            flight_counter += 1
            
            # 生成配置
            config = self.generate_flight_config(flight_counter, dataset_type, 'gust')
            
            # 输出文件
            output_file = output_subdir / f"flight_{flight_counter:04d}.json"
            
            try:
                # 重启SITL
                self.stop_px4_sitl()
                self.start_px4_sitl()
                await self.initialize_controllers()
                
                # 执行飞行
                await self.execute_single_flight(config, str(output_file))
                
                print(f"[进度] {flight_counter}/{total_flights} 完成 ({flight_counter/total_flights*100:.1f}%)")
                
            except Exception as e:
                print(f"[错误] 飞行 {flight_counter} 失败: {e}")
                continue
            
            finally:
                self.stop_px4_sitl()
                await asyncio.sleep(2)
        
        print(f"\n{'#'*60}")
        print(f"# {dataset_type.upper()} 数据集生成完成！")
        print(f"# 成功: {flight_counter}次")
        print(f"{'#'*60}\n")
    
    async def generate_complete_dataset(self):
        """生成完整数据集（400次飞行）"""
        print("\n" + "="*60)
        print("开始生成完整数据集（400次飞行）")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        # 训练集: 300次（240稳态 + 60阵风）
        await self.generate_dataset_batch('train', num_steady=240, num_gust=60)
        
        # 验证集: 50次（40稳态 + 10阵风）
        await self.generate_dataset_batch('val', num_steady=40, num_gust=10)
        
        # 测试集-ID: 35次（25稳态 + 10阵风）
        await self.generate_dataset_batch('test_id', num_steady=25, num_gust=10)
        
        # 测试集-OOD: 15次（15阵风）
        await self.generate_dataset_batch('test_ood', num_steady=0, num_gust=15)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("数据集生成全部完成！")
        print(f"总耗时: {elapsed_time/3600:.2f} 小时")
        print(f"数据保存在: {self.output_dir}")
        print("="*60 + "\n")


# 主函数
async def main():
    generator = DatasetGenerator(output_dir='./data')
    
    try:
        await generator.generate_complete_dataset()
    except KeyboardInterrupt:
        print("\n[中断] 用户终止程序")
    finally:
        generator.stop_px4_sitl()


if __name__ == '__main__':
    asyncio.run(main())
```

---

## 四、配置文件

### 4.1 创建配置文件（configs/dataset_config.json）

```json
{
  "dataset_name": "PIRNN_WindEstimation_Dataset",
  "version": "1.0",
  "description": "完整的固定翼风场估计数据集，包含训练集、验证集和测试集",
  
  "flight_parameters": {
    "altitude_range": [80, 120],
    "speed_range": [12, 18],
    "wind_speed_range": [6, 14],
    "wind_direction_range": [0, 360]
  },
  
  "turbulence": {
    "types": ["light", "moderate"],
    "light_intensity": 1.0,
    "moderate_intensity": 2.0
  },
  
  "gust_parameters": {
    "id_magnitude_range": [4, 8],
    "id_duration_range": [3, 6],
    "ood_magnitude_range": [9, 12],
    "ood_duration_range_short": [3, 6],
    "ood_duration_range_long": [8, 12]
  },
  
  "maneuver_types": {
    "straight_line": {
      "duration_range": [90, 150],
      "probability": 0.4
    },
    "orbit": {
      "radius_range": [80, 150],
      "duration_range": [120, 180],
      "probability": 0.3
    },
    "figure_eight": {
      "radius_range": [60, 100],
      "duration_range": [180, 240],
      "probability": 0.2
    },
    "climb_descent": {
      "climb_rate_range": [1, 3],
      "duration_range": [90, 150],
      "probability": 0.1
    }
  },
  
  "dataset_split": {
    "train": {
      "total": 300,
      "steady": 240,
      "gust": 60
    },
    "val": {
      "total": 50,
      "steady": 40,
      "gust": 10
    },
    "test_id": {
      "total": 35,
      "steady": 25,
      "gust": 10
    },
    "test_ood": {
      "total": 15,
      "steady": 0,
      "gust": 15
    }
  }
}
```

---

## 五、执行数据生成

### 5.1 准备工作

```bash
# 1. 激活虚拟环境
source ~/pirnn_env/bin/activate

# 2. 进入工作目录
cd ~/pirnn_dataset_generation

# 3. 确保所有脚本可执行
chmod +x scripts/*.py

# 4. 验证PX4路径
ls ~/PX4-Autopilot  # 应该看到PX4源码目录
```

### 5.2 测试单次飞行

```bash
# 先测试单次飞行，确保一切正常
cd ~/pirnn_dataset_generation/scripts

python3 << EOF
import asyncio
from generate_dataset import DatasetGenerator

async def test():
    gen = DatasetGenerator(output_dir='../data/test')
    gen.start_px4_sitl()
    await gen.initialize_controllers()
    
    # 生成测试配置
    config = gen.generate_flight_config(1, 'test', 'steady')
    
    # 执行飞行
    await gen.execute_single_flight(config, '../data/test/test_flight.json')
    
    gen.stop_px4_sitl()

asyncio.run(test())
EOF
```

如果成功，应该看到：
- PX4启动日志
- 无人机起飞
- 执行机动
- 数据保存
- `test_flight.json` 和 `test_flight_metadata.json` 生成

### 5.3 生成完整数据集

```bash
# 方式1：直接运行（前台运行）
cd ~/pirnn_dataset_generation/scripts
python3 generate_dataset.py

# 方式2：后台运行（推荐，防止SSH断开）
nohup python3 generate_dataset.py > ../logs/generation.log 2>&1 &

# 查看进度
tail -f ../logs/generation.log

# 查看进程
ps aux | grep generate_dataset
```

### 5.4 监控生成进度

```bash
# 新开一个终端，实时查看已生成的文件数
watch -n 5 'find ~/pirnn_dataset_generation/data -name "*.json" | wc -l'

# 查看各数据集的文件数
find ~/pirnn_dataset_generation/data/train -name "flight_*.json" | wc -l
find ~/pirnn_dataset_generation/data/val -name "flight_*.json" | wc -l
find ~/pirnn_dataset_generation/data/test_id -name "flight_*.json" | wc -l
find ~/pirnn_dataset_generation/data/test_ood -name "flight_*.json" | wc -l
```

---

## 六、数据验证与后处理

### 6.1 数据验证脚本（scripts/validate_dataset.py）

```python
"""
validate_dataset.py
验证生成的数据集完整性和质量
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class DatasetValidator:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        
    def validate_completeness(self):
        """检查数据集完整性"""
        print("\n" + "="*60)
        print("数据集完整性检查")
        print("="*60)
        
        expected = {
            'train': 300,
            'val': 50,
            'test_id': 35,
            'test_ood': 15
        }
        
        for dataset_type, expected_count in expected.items():
            flight_files = list((self.data_dir / dataset_type).glob('flight_*.json'))
            actual_count = len(flight_files)
            
            status = "✓" if actual_count == expected_count else "✗"
            print(f"{status} {dataset_type:10s}: {actual_count:3d}/{expected_count:3d} 飞行")
            
            if actual_count < expected_count:
                print(f"  警告: 缺少 {expected_count - actual_count} 次飞行")
    
    def validate_data_quality(self, dataset_type='train', sample_size=10):
        """检查数据质量"""
        print(f"\n" + "="*60)
        print(f"数据质量检查 ({dataset_type})")
        print("="*60)
        
        flight_files = list((self.data_dir / dataset_type).glob('flight_*.json'))
        
        if len(flight_files) == 0:
            print("错误: 没有找到数据文件")
            return
        
        # 随机采样
        sample_files = np.random.choice(flight_files, 
                                       min(sample_size, len(flight_files)), 
                                       replace=False)
        
        for flight_file in sample_files:
            with open(flight_file, 'r') as f:
                data = json.load(f)
            
            # 检查数据点数量
            num_points = len(data)
            
            # 检查必要字段
            required_fields = ['timestamp', 'position', 'velocity_ned', 'attitude', 'imu']
            
            missing_fields = []
            for entry in data[:5]:  # 检查前5个数据点
                for field in required_fields:
                    if field not in entry:
                        missing_fields.append(field)
            
            status = "✓" if len(missing_fields) == 0 else "✗"
            print(f"{status} {flight_file.name}: {num_points} 数据点")
            
            if missing_fields:
                print(f"  缺少字段: {set(missing_fields)}")
    
    def analyze_wind_distribution(self):
        """分析风场参数分布"""
        print(f"\n" + "="*60)
        print("风场参数分布分析")
        print("="*60)
        
        wind_speeds = []
        wind_directions = []
        gust_magnitudes = []
        
        for dataset_type in ['train', 'val', 'test_id', 'test_ood']:
            metadata_files = list((self.data_dir / dataset_type).glob('*_metadata.json'))
            
            for meta_file in metadata_files:
                with open(meta_file, 'r') as f:
                    config = json.load(f)
                
                wind_speeds.append(config['wind_speed'])
                wind_directions.append(config['wind_direction'])
                
                if 'gust' in config:
                    gust_magnitudes.append(config['gust']['magnitude'])
        
        # 统计
        print(f"风速: {np.mean(wind_speeds):.2f} ± {np.std(wind_speeds):.2f} m/s")
        print(f"      范围: [{np.min(wind_speeds):.2f}, {np.max(wind_speeds):.2f}] m/s")
        
        print(f"风向: {np.mean(wind_directions):.2f} ± {np.std(wind_directions):.2f}°")
        
        if gust_magnitudes:
            print(f"阵风幅值: {np.mean(gust_magnitudes):.2f} ± {np.std(gust_magnitudes):.2f} m/s")
            print(f"          范围: [{np.min(gust_magnitudes):.2f}, {np.max(gust_magnitudes):.2f}] m/s")
        
        # 绘图
        self._plot_distributions(wind_speeds, wind_directions, gust_magnitudes)
    
    def _plot_distributions(self, wind_speeds, wind_directions, gust_magnitudes):
        """绘制分布图"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 风速分布
        axes[0].hist(wind_speeds, bins=20, edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('风速 (m/s)')
        axes[0].set_ylabel('频数')
        axes[0].set_title('风速分布')
        axes[0].grid(True, alpha=0.3)
        
        # 风向分布
        axes[1].hist(wind_directions, bins=36, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('风向 (度)')
        axes[1].set_ylabel('频数')
        axes[1].set_title('风向分布')
        axes[1].grid(True, alpha=0.3)
        
        # 阵风幅值分布
        if gust_magnitudes:
            axes[2].hist(gust_magnitudes, bins=15, edgecolor='black', alpha=0.7)
            axes[2].set_xlabel('阵风幅值 (m/s)')
            axes[2].set_ylabel('频数')
            axes[2].set_title('阵风幅值分布')
            axes[2].axvline(x=8, color='r', linestyle='--', label='ID/OOD边界')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.data_dir.parent / 'results' / 'wind_distribution.png', dpi=150)
        print(f"\n分布图已保存到: results/wind_distribution.png")
    
    def generate_report(self):
        """生成完整的验证报告"""
        self.validate_completeness()
        self.validate_data_quality('train', sample_size=10)
        self.validate_data_quality('test_ood', sample_size=5)
        self.analyze_wind_distribution()


# 运行验证
if __name__ == '__main__':
    validator = DatasetValidator(data_dir='../data')
    validator.generate_report()
```

### 6.2 运行验证

```bash
cd ~/pirnn_dataset_generation/scripts
python3 validate_dataset.py
```

---

## 七、常见问题排查

### 7.1 SITL启动失败

**症状**：PX4无法启动或JSBSim报错

**解决方案**：
```bash
# 1. 检查PX4是否正确编译
cd ~/PX4-Autopilot
make clean
make px4_sitl_default

# 2. 检查JSBSim安装
which JSBSim
JSBSim --version

# 3. 检查端口占用
sudo netstat -tulpn | grep 14540
sudo netstat -tulpn | grep 14550

# 4. 强制清理残留进程
pkill -9 px4
pkill -9 jsbsim
pkill -9 gz
```

### 7.2 MAVSDK连接超时

**症状**：`[FlightController] 等待连接...` 长时间无响应

**解决方案**：
```python
# 在 flight_controller.py 中增加超时和重试
async def connect(self, system_address="udp://:14540", timeout=30):
    await self.drone.connect(system_address=system_address)
    
    print("[FlightController] 等待连接...")
    start_time = asyncio.get_event_loop().time()
    
    async for state in self.drone.core.connection_state():
        if state.is_connected:
            print("[FlightController] 已连接!")
            self.is_connected = True
            break
        
        # 超时检查
        if asyncio.get_event_loop().time() - start_time > timeout:
            raise TimeoutError("连接PX4超时")
```

### 7.3 风场参数不生效

**症状**：无人机飞行轨迹与预期不符

**可能原因**：
- PX4版本不同，参数名称不同
- 参数设置后未等待足够时间

**解决方案**：
```python
# 在 wind_manager.py 中添加参数验证
def set_constant_wind(self, speed, direction):
    # 设置参数
    self.master.mav.param_set_send(...)
    
    time.sleep(1)  # 增加等待时间
    
    # 验证参数是否设置成功
    self.master.mav.param_request_read_send(
        self.master.target_system,
        self.master.target_component,
        b'SIM_WIND_SPD',
        -1
    )
    
    msg = self.master.recv_match(type='PARAM_VALUE', blocking=True, timeout=3)
    if msg:
        actual_value = msg.param_value
        print(f"验证: SIM_WIND_SPD = {actual_value}")
```

### 7.4 数据记录不完整

**症状**：生成的JSON文件数据点很少

**解决方案**：
```python
# 在 data_logger.py 中增加日志
async def start_logging(self, duration, output_file):
    print(f"[DataLogger] 开始记录，目标时长={duration}s")
    
    # ...记录代码...
    
    print(f"[DataLogger] 实际记录时长={time.time() - self.start_time:.1f}s")
    print(f"[DataLogger] 记录数据点={len(self.data_buffer)}")
    
    # 检查数据率
    expected_points = duration * 50  # 50Hz
    actual_points = len(self.data_buffer)
    
    if actual_points < expected_points * 0.8:
        print(f"警告: 数据点不足，预期{expected_points}，实际{actual_points}")
```

---

## 八、加速数据生成

### 8.1 使用加速因子

修改 `generate_dataset.py`：

```python
def start_px4_sitl(self, speedup=5):
    """启动PX4 SITL仿真"""
    px4_dir = Path.home() / 'PX4-Autopilot'
    
    # 设置环境变量
    env = os.environ.copy()
    env['PX4_SIM_SPEED_FACTOR'] = str(speedup)
    
    self.px4_process = subprocess.Popen(
        ['make', 'px4_sitl', 'jsbsim_rascal'],
        cwd=str(px4_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    time.sleep(10)
    print(f"[DatasetGenerator] PX4 SITL已启动 (加速{speedup}x)")
```

### 8.2 并行生成（高级）

创建 `scripts/parallel_generate.py`：

```python
"""
parallel_generate.py
并行运行多个SITL实例加速数据生成
"""

import subprocess
import multiprocessing as mp
from pathlib import Path


def run_instance(instance_id, num_flights, output_dir):
    """运行单个SITL实例"""
    # 设置不同的端口
    base_port = 14540 + instance_id * 10
    
    env = {
        'PX4_SIM_PORT_BASE': str(base_port),
        'HEADLESS': '1'
    }
    
    # 启动生成脚本
    subprocess.run([
        'python3', 'generate_dataset.py',
        '--instance', str(instance_id),
        '--num-flights', str(num_flights),
        '--output-dir', output_dir
    ], env=env)


if __name__ == '__main__':
    num_instances = 3  # 3个并行实例
    flights_per_instance = 100
    
    processes = []
    for i in range(num_instances):
        p = mp.Process(
            target=run_instance,
            args=(i, flights_per_instance, f'./data/instance_{i}')
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("所有实例完成！")
```

---

## 九、数据格式说明

### 9.1 飞行数据文件（flight_XXXX.json）

```json
[
  {
    "timestamp": 0.02,
    "position": {
      "lat": 47.3977419,
      "lon": 8.5455938,
      "alt_msl": 488.5,
      "alt_rel": 80.2
    },
    "velocity_ned": {
      "north": 12.5,
      "east": 3.2,
      "down": -0.1
    },
    "attitude": {
      "roll": 2.5,
      "pitch": 3.8,
      "yaw": 45.2
    },
    "imu": {
      "acc_x": 9.85,
      "acc_y": 0.15,
      "acc_z": -0.25,
      "gyro_x": 0.02,
      "gyro_y": -0.01,
      "gyro_z": 0.15
    },
    "gps": {
      "num_satellites": 18,
      "fix_type": 3
    }
  },
  ...
]
```

### 9.2 元数据文件（flight_XXXX_metadata.json）

```json
{
  "id": "train_0001",
  "dataset_type": "train",
  "config_type": "steady",
  "wind_speed": 10.5,
  "wind_direction": 135.0,
  "turbulence": "moderate",
  "altitude": 95.0,
  "speed": 15.2,
  "heading": 45.0,
  "maneuver_type": "straight_line",
  "duration": 120.0,
  "gust": null
}
```

或带阵风：

```json
{
  "id": "train_0241",
  "config_type": "gust",
  "gust": {
    "magnitude": 6.5,
    "duration": 5.0,
    "direction": 270.0,
    "start_time": 45.0
  },
  ...
}
```

---

## 十、总结与检查清单

### 10.1 生成前检查

- [ ] PX4-Autopilot已安装并编译成功
- [ ] JSBSim已安装（`JSBSim --version`可执行）
- [ ] Python依赖已安装（mavsdk, pymavlink等）
- [ ] 工作目录结构已创建
- [ ] 所有脚本已复制到`scripts/`目录
- [ ] 测试单次飞行成功

### 10.2 生成中监控

- [ ] 定期检查日志文件（`tail -f logs/generation.log`）
- [ ] 监控文件生成数量
- [ ] 检查磁盘空间（约需10GB）
- [ ] 监控CPU和内存使用

### 10.3 生成后验证

- [ ] 运行`validate_dataset.py`检查完整性
- [ ] 检查数据质量（缺失字段、异常值）
- [ ] 查看风场分布图
- [ ] 备份数据到外部存储

### 10.4 预计时间

| 项目 | 加速1x | 加速5x | 加速10x |
|------|--------|--------|---------|
| 单次飞行 | ~3分钟 | ~40秒 | ~20秒 |
| 训练集（300次）| ~15小时 | ~3小时 | ~1.5小时 |
| 完整数据集（400次）| ~20小时 | ~4小时 | ~2小时 |

**推荐配置**：
- 使用5x加速（稳定性与速度平衡）
- 后台运行（`nohup`）
- 总耗时约4-5小时

---

## 十一、下一步

数据生成完成后：

1. **数据预处理**：转换为训练所需格式（numpy数组）
2. **特征工程**：提取滑动窗口、计算统计特征
3. **数据增强**（可选）：添加噪声、时间扰动
4. **开始训练PIRNN-AKF**

需要我继续提供数据预处理和训练脚本吗？
