
# PX4-SITL + JSBSim 数据集生成完整指南

## 目标

使用 PX4-SITL + JSBSim 框架生成PIRNN-AKF训练所需的完整数据集，包括：
- **训练集**：300次飞行（240次稳态 + 60次阵风）
- **验证集**：50次飞行（40次稳态 + 10次阵风）
- **测试集-ID**：35次飞行（25次稳态 + 10次阵风）
- **测试集-OOD**：15次飞行（15次强阵风/长阵风）

**总计**：400次飞行

---

## 审查意见与修订建议（针对「真值风场 + 飞行状态」采集）

> 以下为对本文档的审查结论，确保与 PX4-SITL + JSBSim 框架下**真值风场**与**飞行状态**的采集、以及本仓库训练管线一致。

### 1. 真值风场未按时间序列记录（严重）

- **问题**：当前仅把风场写在**元数据**（`flight_XXXX_metadata.json`）里，如 `wind_speed`、`wind_direction`；**每条时间戳对应的真值风场并未写入飞行数据**。
- **影响**：本仓库训练时 `y_batch[:, :3]` 为**每时刻**的 3 维风真值（wind_north, wind_east, wind_down）。`src/1_data_preprocessing_csv.py` 与 `utils/jsbsim_csv_parser.py` 期望的数据源是**带逐行风场列**的 JSBSim CSV（如 `/fdm/jsbsim/atmosphere/wind-north-fps` 等），而不是「仅元数据里的单次风场」。
- **建议**：
  - **方案 A**：若数据源为 **JSBSim 导出**：直接使用/配置 JSBSim 或 PX4-SITL 的日志，导出包含 `wind-north-fps`、`wind-east-fps`、`wind-down-fps` 的 CSV，与现有解析器对接。
  - **方案 B**：若坚持用当前 MAVSDK + JSON 管线：在 `data_logger.py` 中**按时间戳同步记录真值风**。稳态风：每行写 metadata 中的 (wind_north, wind_east, wind_down)；阵风：在记录端用与 `inject_cosine_gust` 一致的 1-cos 公式按时间生成 (wind_north, wind_east, wind_down) 并写入每条记录。后处理阶段再统一转换为本仓库所需的 CSV/列格式。

### 2. 数据格式与现有训练管线不一致

- **问题**：指南输出为 **JSON**（每条为带 timestamp 的 position/velocity_ned/attitude/imu/gps），而 `JSBSimCSVParser` 和训练代码期望的是 **JSBSim 风格 CSV**（含 wind、vc-fps、vtrue-fps、v-north-fps、姿态弧度、舵面等）。
- **影响**：生成的 JSON 不能直接给 `1_data_preprocessing_csv.py` 使用，且缺少空速、舵面、真值风等列。
- **建议**：在文档中明确二选一或并行支持：（1）**优先路径**：从 JSBSim/PX4-SITL 导出与 README 中「JSBSim数据字段」一致的 CSV，再走现有预处理与训练；（2）**备用路径**：若仅用 MAVSDK 日志，则增加「后处理脚本」：将 JSON + 元数据转为上述 CSV 格式（含逐行真值风、空速、舵面等），并说明如何与 `config.yaml` 中的 `raw_log_dir` / `processed_dir` 对接。

### 3. 风场必须在 JSBSim 中设置（已采用）

- **结论**：**修改 PX4 的 SIM_WIND_* 参数在 PX4-SITL + JSBSim 下无效**，动力学与风场均由 JSBSim 负责。本指南已改为**仅在 JSBSim 中设置风场**：通过**启动前修改 JSBSim XML**（`wind_north` / `wind_east` / `wind_down`）或**纯 JSBSim Python 脚本**（方案 B）实现；真值风从 JSBSim 输出或由元数据 + 后处理写入 CSV。

### 4. 连接端口统一

- **建议**：所有 MAVLink/MAVSDK 连接统一使用 **14540**（与 PX4 SITL 默认端口一致）。

### 5. 数据记录器时间对齐与缺失字段

- **问题**：`DataLogger` 用多个异步任务分别拉取 position、velocity、attitude、imu、gps，再用 `_get_or_create_entry(timestamp)` 合并；各流时间戳不完全一致，易产生**同一时刻多行、或单行缺字段**，且无**空速、舵面、真值风**。
- **影响**：训练需要「每行一个时间步、含完整状态+真值风」；当前设计难以直接得到对齐的单表。
- **建议**：改为「以统一时间步（如 50 Hz）为驱动，每步主动请求各 telemetry 并写一行」，或在后处理中按统一时间网格插值/重采样，并补全空速（若 MAVSDK 提供）、舵面（若可从 MAVLink 或 JSBSim 获取）和真值风（见第 1 条）。

### 6. 阵风场景下的真值风与任务同步

- **问题**：阵风通过 `inject_cosine_gust` 在**仿真运行中**动态改风，但**数据里没有记录每一时刻对应的真值风**；且阵风任务用 `asyncio.create_task` 未 await，可能与飞行/记录结束顺序错位。
- **建议**：阵风场景下，（1）在记录端按与 `inject_cosine_gust` 相同的 1-cos 公式和 timeline 生成每时刻 (wind_north, wind_east, wind_down) 并写入数据；（2）或确保从 JSBSim 直接导出带时间戳的风场。同时 await 阵风注入相关任务或与 flight_task/logging_task 协调，避免提前结束记录。

### 7. 小结与修订优先级

| 优先级 | 项目 | 说明 |
|--------|------|------|
| 高 | 真值风场按时间序列写入数据 | 否则无法训练/评估 PIRNN-AKF 的风估计 |
| 高 | 数据格式与 CSV 管线一致 | 能与 `JSBSimCSVParser` 和 `1_data_preprocessing_csv.py` 对接 |
| 高 | 风场仅在 JSBSim 中设置 | 已采用：XML 启动前修改 或 纯 JSBSim Python 方案 |
| 中 | 连接端口统一为 14540 | 与 PX4 SITL 默认一致 |
| 中 | 记录器时间对齐与缺列 | 保证每行完整、可训练 |
| 低 | 阵风任务同步与真值记录 | 阵风场景下真值风与状态一致 |

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
├── scripts/                  # Python脚本
│   ├── jsbsim_wind_config.py # JSBSim 风场配置（修改 XML）
│   ├── flight_controller.py
│   ├── data_logger.py
│   ├── generate_dataset.py
│   ├── postprocess_to_jsbsim_csv.py  # 方案A：MAVSDK 数据 → JSBSim 格式 CSV
│   └── jsbsim_standalone_run.py     # 方案B：纯 JSBSim 脚本导出 CSV（可选）
├── data/
│   ├── train/ | val/ | test_id/ | test_ood/
├── logs/
├── configs/
└── results/
```

---

## 三、可用的数据采集方案（风场仅在 JSBSim 中设置）

> **重要**：PX4 的 `SIM_WIND_*` 参数在 PX4-SITL + JSBSim 下**不生效**，风场必须通过 **JSBSim** 设置。下面给出两种可用的数据采集方式，输出均为本仓库训练所需的 **JSBSim 风格 CSV**（含逐行真值风与状态）。

### 3.0 方案总览

| 方案 | 风场设置 | 飞行控制 | 数据来源 | 输出格式 | 适用场景 |
|------|----------|----------|----------|----------|----------|
| **A** | 每次飞行前修改 JSBSim XML | PX4 + MAVSDK | MAVSDK 遥测 + 元数据风场 → 后处理 | 单 CSV/多 CSV | 需要 PX4 飞控行为的轨迹 |
| **B** | JSBSim Python API 每步设风 | 脚本/简单自动驾驶 | JSBSim FDM 直接输出 | 直接 JSBSim CSV | 批量生成、阵风/湍流灵活 |

- **方案 A**：适合「真实飞控逻辑 + 多种机动」；真值风由本架次 XML 设定，后处理时按时间序列填满每行。
- **方案 B**：适合「最快得到训练用 CSV」；无 PX4，控制逻辑需在脚本中实现（或接简单自动驾驶）。

以下先给出 **JSBSim 风场配置**（方案 A/B 共用或仅 A 用），再分别写方案 A 与方案 B 的脚本要点。

---

### 3.1 JSBSim 风场配置（jsbsim_wind_config.py）

在 **启动 PX4-SITL 之前** 修改 JSBSim 的 XML 配置文件，写入本架次风场（N/E/D，单位 m/s），**不使用任何 PX4 参数**。Rascal 的配置文件通常在 PX4 工程下的 `Tools/simulation/jsbsim/models/rascal/` 或 `Tools/sitl_gazebo/models/rascal/`，请按实际路径调整 `jsbsim_xml_path`。

```python
"""
jsbsim_wind_config.py
在 JSBSim 中设置风场：修改 XML 配置文件（启动 SITL 前调用）。
不依赖 PX4 的 SIM_WIND_* 参数。
"""

import xml.etree.ElementTree as ET
import numpy as np
import random
from pathlib import Path


# JSBSim 使用 英尺/秒 (ft/s)，转换系数
MPS_TO_FPS = 3.28084


def get_jsbsim_rascal_xml_path():
    """返回 Rascal 的 JSBSim 配置文件路径（按你的 PX4 安装调整）"""
    px4_dir = Path.home() / 'PX4-Autopilot'
    candidates = [
        px4_dir / 'Tools/simulation/jsbsim/models/rascal/rascal_jsbsim.xml',
        px4_dir / 'Tools/sitl_gazebo/models/rascal/rascal_jsbsim.xml',
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError("未找到 rascal_jsbsim.xml，请检查 PX4 目录下 Tools 结构")


def set_jsbsim_wind(xml_path, wind_north_mps, wind_east_mps, wind_down_mps=0.0, turb_gain=1.0):
    """
    在 JSBSim XML 中设置风场与湍流强度。
    
    Args:
        xml_path: rascal_jsbsim.xml 的路径
        wind_north_mps: 北向风速 (m/s)
        wind_east_mps: 东向风速 (m/s)
        wind_down_mps: 垂向风速 (m/s)，通常 0
        turb_gain: 湍流增益（如 1.0=轻度, 2.0=中度）
    
    Returns:
        (wind_north, wind_east, wind_down) 用于写入元数据/后处理
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    winds = root.find('.//winds')
    if winds is not None:
        n = winds.find('wind_north')
        e = winds.find('wind_east')
        d = winds.find('wind_down')
        if n is not None:
            n.text = f'{wind_north_mps * MPS_TO_FPS:.4f}'
        if e is not None:
            e.text = f'{wind_east_mps * MPS_TO_FPS:.4f}'
        if d is not None:
            d.text = f'{wind_down_mps * MPS_TO_FPS:.4f}'
    
    turb = root.find('.//turbulence')
    if turb is not None:
        tg = turb.find('turbulence_gain')
        if tg is not None:
            tg.text = f'{turb_gain:.2f}'
    
    tree.write(xml_path)
    print(f"[JSBSim] 已设置风场: N={wind_north_mps:.2f}, E={wind_east_mps:.2f}, D={wind_down_mps:.2f} m/s, turb_gain={turb_gain}")
    return (wind_north_mps, wind_east_mps, wind_down_mps)


def wind_from_speed_direction(speed_mps, direction_deg):
    """由风速(m/s)和风向(度, 0=北)得到 N/E/D 分量 (m/s)。"""
    rad = np.deg2rad(direction_deg)
    wn = speed_mps * np.cos(rad)
    we = speed_mps * np.sin(rad)
    return wn, we, 0.0


def set_random_wind_in_jsbsim(xml_path=None, speed_range=(6, 14), direction_range=(0, 360), turb_gain=None):
    """
    随机生成风场并写入 JSBSim XML。用于每架次前调用。
    
    Returns:
        dict: wind_north, wind_east, wind_down (m/s), wind_speed, wind_direction
    """
    if xml_path is None:
        xml_path = get_jsbsim_rascal_xml_path()
    
    speed = random.uniform(*speed_range)
    direction = random.uniform(*direction_range)
    wn, we, wd = wind_from_speed_direction(speed, direction)
    if turb_gain is None:
        turb_gain = random.choice([1.0, 2.0])
    
    set_jsbsim_wind(xml_path, wn, we, wd, turb_gain=turb_gain)
    return {
        'wind_north': wn, 'wind_east': we, 'wind_down': wd,
        'wind_speed': speed, 'wind_direction': direction,
        'turbulence_gain': turb_gain,
    }


if __name__ == '__main__':
    xml_path = get_jsbsim_rascal_xml_path()
    info = set_random_wind_in_jsbsim(xml_path)
    print("当前随机风场(用于元数据):", info)
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
from data_logger import DataLogger
from jsbsim_wind_config import set_random_wind_in_jsbsim, get_jsbsim_rascal_xml_path


class DatasetGenerator:
    def __init__(self, output_dir='./data'):
        self.output_dir = Path(output_dir)
        self.fc = None
        self.px4_process = None
        self.jsbsim_xml_path = None
        
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
        """初始化飞行控制器（风场已在启动前通过 JSBSim XML 设置）"""
        self.fc = FlightController()
        await self.fc.connect()
        print("[DatasetGenerator] 控制器已初始化")
    
    def _apply_jsbsim_wind_for_flight(self, flight_config):
        """根据本架次配置在 JSBSim XML 中写入风场（必须在 start_px4_sitl 之前调用）"""
        if self.jsbsim_xml_path is None:
            self.jsbsim_xml_path = get_jsbsim_rascal_xml_path()
        wn = flight_config['wind_north']
        we = flight_config['wind_east']
        wd = flight_config.get('wind_down', 0.0)
        turb = 2.0 if flight_config.get('turbulence') == 'moderate' else 1.0
        from jsbsim_wind_config import set_jsbsim_wind
        set_jsbsim_wind(self.jsbsim_xml_path, wn, we, wd, turb_gain=turb)
    
    async def execute_single_flight(self, flight_config, output_file):
        """
        执行单次飞行。风场已在本架次启动 SITL 前通过 JSBSim XML 设置，此处不再调用 PX4 参数。
        
        Args:
            flight_config: 飞行配置字典（含 wind_north, wind_east, wind_down 等）
            output_file: 输出文件路径（方案 A 为 JSON，后处理生成 CSV）
        """
        print(f"\n{'='*60}")
        print(f"[DatasetGenerator] 执行飞行: {flight_config['id']}")
        print(f"{'='*60}")
        
        # 1. 风场已在 start_px4_sitl 前由 _apply_jsbsim_wind_for_flight 写入 JSBSim XML，无需再设
        
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
        
        # 5. 阵风：若需阵风，需在 JSBSim 脚本/XML 中配置（如方案 B），或后处理时按 1-cos 公式合成真值风列
        
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
        
        # 风场随机化（用于写入 JSBSim XML，单位 m/s）
        config['wind_speed'] = random.uniform(6, 14)
        config['wind_direction'] = random.uniform(0, 360)
        wn = config['wind_speed'] * np.cos(np.deg2rad(config['wind_direction']))
        we = config['wind_speed'] * np.sin(np.deg2rad(config['wind_direction']))
        config['wind_north'] = wn
        config['wind_east'] = we
        config['wind_down'] = 0.0
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
        
        # 如果是阵风场景（方案 A 下 XML 仍设恒定风；阵风真值可在后处理中按 1-cos 合成）
        if config_type == 'gust':
            if dataset_type == 'test_ood':
                gust_magnitude = random.uniform(9, 12)
                gust_duration = random.uniform(8, 12) if random.random() < 0.5 else random.uniform(3, 6)
            else:
                gust_magnitude = random.uniform(4, 8)
                gust_duration = random.uniform(3, 6)
            config['gust'] = {
                'magnitude': gust_magnitude,
                'duration': gust_duration,
                'direction': random.uniform(0, 360),
                'start_time': random.uniform(30, 60)
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
                # 先在本架次配置下写入 JSBSim 风场，再启动 SITL（顺序不可颠倒）
                self._apply_jsbsim_wind_for_flight(config)
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
                self._apply_jsbsim_wind_for_flight(config)
                self.stop_px4_sitl()
                self.start_px4_sitl()
                await self.initialize_controllers()
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

### 3.5 方案 A：MAVSDK 数据后处理为 JSBSim 格式 CSV

方案 A 每架次得到的是 **JSON（遥测）+ 元数据（含 wind_north/wind_east/wind_down）**。要接入本仓库的 `JSBSimCSVParser` 和 `1_data_preprocessing_csv.py`，需要将每架次合并为**与 `Dataset/flight_data_all_in_one.csv` 列名一致**的 CSV，且**每行都带真值风**（本架次为恒定风则每行相同）。

后处理脚本要点（保存为 `scripts/postprocess_to_jsbsim_csv.py`）：

1. **输入**：`data/<split>/flight_XXXX.json`、`flight_XXXX_metadata.json`。
2. **列与单位**：与 README 中「JSBSim数据字段」一致；速度/风用 **fps**（英尺/秒），姿态用 **rad**。例如：
   - `Time` = 相对时间 (s)
   - `/fdm/jsbsim/simulation/sim-time-sec` = 同上
   - `/fdm/jsbsim/atmosphere/wind-north-fps` = metadata 的 `wind_north` × 3.28084（每行相同）
   - `/fdm/jsbsim/atmosphere/wind-east-fps`、`wind-down-fps` 同理
   - `/fdm/jsbsim/velocities/v-north-fps` = velocity_ned.north × 3.28084
   - `/fdm/jsbsim/velocities/vtrue-fps` 若无空速则可用地速模长近似或留空
   - `/fdm/jsbsim/attitude/roll-rad` = roll 度转弧度，pitch/psi 同理
   - 舵面若 MAVSDK 无，可填 0
3. **输出**：每架次一个 CSV，或合并为一个 `flight_data_all_in_one.csv`，供 `utils/jsbsim_csv_parser.py` 与 `src/1_data_preprocessing_csv.py` 使用。

若本架次为**阵风**，元数据中有 `gust` 时：可按 `start_time`、`duration`、`magnitude`、`direction` 用 1-cos 公式生成每时刻的 (wind_north, wind_east, wind_down) 再写入对应行。

---

### 3.6 方案 B：纯 JSBSim Python 脚本直接导出 CSV（可选）

不启动 PX4，仅用 **JSBSim Python 库** 跑仿真：在脚本中设置风场（恒定 + 可选阵风/湍流），每步读取 FDM 状态并写入 CSV，列名与 `Dataset/flight_data_all_in_one.csv` 一致，可直接被本仓库解析器使用。

要点：

1. 安装：`pip install jsbsim`
2. 初始化：`fdm = jsbsim.FGFDMExec(root_dir=...)`，`fdm.load_model('rascal')`，`fdm.load_ic(...)`
3. 设风：`fdm['atmosphere/wind-north-fps'] = wn_mps * 3.28084`，同理 east/down；阵风可在循环中按时间修改 `atmosphere/gust-*-fps`
4. 每步：`fdm.run()`，然后读取 `fdm['simulation/sim-time-sec']`、`fdm['atmosphere/wind-north-fps']`、`fdm['velocities/v-north-fps']` 等，写入 CSV 一行（单位保持 fps/rad）
5. 控制：需在循环中根据目标轨迹设置舵面/油门（如 `fdm['fcs/aileron-cmd-norm'] = ...`），或使用 JSBSim 自带自动驾驶脚本

这样得到的 CSV 天然包含**逐行真值风**与完整状态，无需后处理。代价是飞控逻辑需在脚本或 JSBSim 脚本中实现，而非 PX4。

---

### 3.7 单次运行多段采集与无人监管（推荐用于大数据量）

为减少 SITL 重启次数并支持**无人监管**下长时间采集，可采用 **「一次 SITL 运行、多段机动、分段存盘」** 的方式：在同一风场下连续执行多段飞行，每段保存为一条数据记录，**仅在一轮多段全部结束后再重启 SITL 更换风场**。

#### 3.7.1 为何可以“一次运行多段”

- **风场**：由 JSBSim XML 在**进程启动时**决定，单次 SITL 进程内风场不变。要获得不同风场，必须重启 SITL 并重新写 XML。
- **机动与记录**：在同一进程内可以**连续执行多段机动**（直线、盘旋、8 字、爬升等），每段单独记录并存成一个 `flight_*.json` + 元数据，**共享同一风场**。
- **效果**：若每轮运行做 **N 段**（例如 8 段），则 400 条数据只需 **400/N 次** SITL 重启（例如 50 次），大幅缩短总时间并便于无人值守。

#### 3.7.2 单轮流程（一次 SITL，多段采集）

1. **设定本轮风场**：调用 `set_jsbsim_wind(...)` 写入 XML。
2. **启动 SITL 一次**：`start_px4_sitl()` → 等待就绪 → `initialize_controllers()`。
3. **起飞一次**：`arm_and_takeoff(altitude)`。
4. **多段循环**（同一风、不关仿真）：
   - 对 `segment_id = 0 .. N-1`：
     - 生成本段机动配置（航向、速度、半径等可随机或按表），**风场用本轮统一的 wind_north/wind_east/wind_down**；
     - 启动本段数据记录（独立 output 文件）；
     - 执行本段机动（如 `fly_straight_line` / `fly_orbit` / …）；
     - 停止记录，保存本段 JSON + metadata（metadata 中含同一风场 + segment_id / flight_id）；
     - 可选：短时悬停或过渡，再进入下一段（**不降落**）。
5. **降落一次**：多段全部完成后 `land()`。
6. **关闭 SITL**：`stop_px4_sitl()`，进入下一轮（下一风场）重复 1～6。

这样即可在**一次飞行会话**内得到 N 条不同机动、同一风场的数据；多条风场轮次即可覆盖全部 400 条且重启次数最少。

#### 3.7.3 脚本设计要点

- **`execute_single_segment(config_segment, output_file)`**  
  只执行「一段」机动 + 本段数据记录，**不**起飞、**不**降落。入参为该段机动参数与输出路径；metadata 写入本段 wind（= 本轮风场）、segment_id、maneuver_type 等。

- **`run_multi_segment_session(wind_config, segment_configs, output_dir, base_flight_id)`**  
  - 入参：本轮风场 `wind_config`（wind_north, wind_east, wind_down）、本轮 N 段配置列表 `segment_configs`、输出目录、起始 `flight_id`。
  - 流程：`_apply_jsbsim_wind_for_flight(wind_config)` → `start_px4_sitl()` → `initialize_controllers()` → `arm_and_takeoff()` → 对每段调用 `execute_single_segment(segment_config, output_file)` → `land()` → `stop_px4_sitl()`。
  - 每段输出文件可命名为 `flight_{base_flight_id + k:04d}.json`，便于与 train/val/test 划分一致。

- **`generate_dataset_batch_multi_segment(dataset_type, num_wind_runs, segments_per_run)`**  
  - `num_wind_runs`：不同风场轮数（即 SITL 重启次数），例如 50。
  - `segments_per_run`：每轮运行的段数，例如 8；总条数 = `num_wind_runs * segments_per_run`（如 50×8 = 400）。
  - 每轮随机生成一个风场，再随机生成 `segments_per_run` 个机动配置（机动类型、时长、航向等），调用 `run_multi_segment_session`；根据 `dataset_type` 与 `base_flight_id` 将输出写到 `data/train`、`data/val` 等。

- **train/val/test 划分**：可按「轮」或「段」划分。例如前 40 轮为 train（320 条），接下来 5 轮为 val（40 条），最后 5 轮为 test_id（40 条）；或按全局 flight_id 随机划分。在写 metadata 时带上 `dataset_type` 和 `flight_id` 即可。

#### 3.7.4 无人监管与鲁棒性

- **单脚本串行**：按顺序执行「风场 1 多段 → 风场 2 多段 → …」，无需人工切换。
- **超时与异常**：每段或每轮加 `asyncio.wait_for(..., timeout=segment_timeout)`；异常时 `except` 记录日志、可选跳过本段或本轮，继续下一段/下一轮，避免一次失败导致全部中断。
- **日志与进度**：将每段/每轮的 flight_id、wind、成功与否写入 `logs/`，便于事后统计与排查；可同时打印进度（如「第 k/N 轮，第 j/M 段」）。
- **长时间运行**：建议用 `nohup` 或 `tmux` 在后台运行，并定期检查磁盘空间与日志。

#### 3.7.5 与“每架次重启”的对比

| 方式           | SITL 重启次数 | 每轮耗时（示意） | 总条数 | 适用场景           |
|----------------|---------------|------------------|--------|--------------------|
| 每架次重启     | 400           | 约 3～5 分钟/次  | 400    | 每条不同风、逻辑简单 |
| 单次多段（推荐）| 400/N 段      | 约 1 次起飞 + N 段机动 + 1 次降落 | 400    | 大数据量、无人监管   |

示例：N=8 时，400 条仅需 **50 次** SITL 启动；若每次启动+起飞约 2 分钟、每段机动约 2 分钟，则每轮约 2 + 8×2 + 1 ≈ 19 分钟，50 轮约 16 小时（相比 400 次重启可节省大量时间）。

#### 3.7.6 示例：多段执行与单轮会话（伪代码）

```python
# 在 DatasetGenerator 类中新增

async def execute_single_segment(self, segment_config, output_file):
    """执行单段机动并记录，不起飞不降落。segment_config 含 maneuver_type, duration, heading 等，且含 wind_north/wind_east/wind_down（本轮统一）。"""
    logger = DataLogger(self.fc.drone)
    duration = segment_config['duration']
    logging_task = asyncio.create_task(logger.start_logging(duration, output_file))
    maneuver_type = segment_config['maneuver_type']
    altitude = segment_config.get('altitude', 100)
    if maneuver_type == 'straight_line':
        flight_task = asyncio.create_task(self.fc.fly_straight_line(
            heading=segment_config['heading'], altitude=altitude,
            speed=segment_config['speed'], duration=duration))
    elif maneuver_type == 'orbit':
        flight_task = asyncio.create_task(self.fc.fly_orbit(
            radius=segment_config['radius'], altitude=altitude,
            direction=segment_config.get('direction', 'cw'), duration=duration))
    # ... 其他机动类型
    else:
        flight_task = asyncio.create_task(asyncio.sleep(duration))
    await asyncio.gather(flight_task, logging_task)
    metadata_file = output_file.replace('.json', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(segment_config, f, indent=2)

async def run_multi_segment_session(self, wind_config, segment_configs, output_subdir, base_flight_id):
    """单轮：一种风场下连续多段，只起飞/降落各一次。"""
    self._apply_jsbsim_wind_for_flight(wind_config)
    self.stop_px4_sitl()
    self.start_px4_sitl()
    await self.initialize_controllers()
    await self.fc.arm_and_takeoff(altitude=segment_configs[0].get('altitude', 100))
    for k, seg_cfg in enumerate(segment_configs):
        seg_cfg['wind_north'] = wind_config['wind_north']
        seg_cfg['wind_east'] = wind_config['wind_east']
        seg_cfg['wind_down'] = wind_config.get('wind_down', 0.0)
        out_path = output_subdir / f"flight_{base_flight_id + k:04d}.json"
        await self.execute_single_segment(seg_cfg, str(out_path))
    await self.fc.land()
    self.stop_px4_sitl()
```

主循环（无人监管）：按 train/val/test 分配 `num_wind_runs` 与 `segments_per_run`，对每种 split 循环调用 `run_multi_segment_session`，每轮生成一组随机风场和 N 个随机段配置即可。

#### 3.7.7 按 400 条目标的一次 SITL 多段采集详细方案

以下方案严格对应数据集目标：**训练 300（240 稳态 + 60 阵风）、验证 50（40 稳态 + 10 阵风）、测试-ID 35（25 稳态 + 10 阵风）、测试-OOD 15（15 强/长阵风）**，并采用「每轮一次 SITL、多段连续采集」以最少重启次数、无人监管完成。

---

**一、总览：SITL 轮次与每轮段数**

| 数据集     | 稳态条数 | 阵风条数 | 小计 | 每轮段数 | SITL 轮数 | 每轮内稳态/阵风分配 |
|------------|----------|----------|------|----------|-----------|----------------------|
| train      | 240      | 60       | 300  | 5        | 60        | 每轮 4 稳态 + 1 阵风 |
| val        | 40       | 10       | 50   | 5        | 10        | 每轮 4 稳态 + 1 阵风 |
| test_id    | 25       | 10       | 35   | 5        | 7         | 见下表「test_id 每轮分配」 |
| test_ood   | 0        | 15       | 15   | 5        | 3         | 每轮 5 段均为阵风（OOD） |
| **合计**   | **305**  | **95**   | **400** | —     | **80**    | —                    |

- **总 SITL 重启次数：80 次**（每轮一次 SITL，每轮 5 段，共 400 段）。
- test_id 的 7 轮分配（保证恰好 25 稳态 + 10 阵风）：
  - **前 6 轮**：每轮 **4 稳态 + 1 阵风** → 24 稳态 + 6 阵风；
  - **第 7 轮**：**1 稳态 + 4 阵风** → 1 稳态 + 4 阵风；
  - 合计 **25 稳态 + 10 阵风**。

---

**二、每轮 SITL 的风场设置（JSBSim XML）**

每轮只在**启动 SITL 前**设置一次风场（写入 JSBSim XML），该轮内所有段共用此风场。

| 项目         | 取值说明 |
|--------------|----------|
| 风速         | 在 **[6, 14] m/s** 内均匀随机（train/val/test_id/test_ood 均同） |
| 风向         | 在 **[0, 360)°** 内均匀随机 |
| 垂向风       | **0 m/s**（或 [-0.5, 0.5] 若需弱垂向） |
| 湍流增益     | 每轮在 **light(1.0) / moderate(2.0)** 中随机选一 |
| 换算到 XML   | `wind_north = speed×cos(dir°)`, `wind_east = speed×sin(dir°)`，单位 **ft/s**（×3.28084）写入 `<winds>` |

**说明**：  
- 阵风/强阵风**不在仿真里动态注入**，仿真始终为恒定风；「阵风」与「OOD 阵风」在后处理或元数据中通过**真值风叠加 1-cos 阵风**体现（见下「段类型与阵风真值」）。

---

**三、每段轨迹设置（机动类型与参数）**

每段在**该轮共用风场**下，独立随机选择机动类型与参数，保证轨迹多样性。

| 机动类型        | 概率（约） | 时长范围 (s) | 其他参数（均匀随机） |
|-----------------|------------|--------------|----------------------|
| straight_line   | 0.4        | [90, 150]    | heading [0, 360)°, speed [12, 18] m/s, altitude 同轮首段或 [80, 120] m |
| orbit           | 0.3        | [120, 180]   | radius [80, 150] m, direction cw/ccw 等概, altitude 同上 |
| figure_eight    | 0.2        | [180, 240]   | lobe radius [60, 100] m, orientation [0, 360)°, altitude 同上 |
| climb_descent   | 0.1        | [90, 150]    | climb_rate [1, 3] m/s（或 [-3,-1] 下降）, target_altitude 当前±[20, 40] m, heading [0, 360)° |

- **高度**：可整轮固定为 80–120 m 内随机一个值，或每段在该范围内重新随机（建议至少首段随机，后续段可同高或略变）。
- **段间**：段与段之间不降落，可短时保持当前高度/速度再进入下一段，避免大机动突变。

---

**四、段类型与阵风真值（metadata / 后处理）**

- **稳态段**：metadata 中 `config_type: "steady"`，真值风 = 本轮风场（wind_north, wind_east, wind_down）恒定写满整段。
- **阵风段（ID）**：metadata 中 `config_type: "gust"`，并写入 `gust`：  
  `magnitude` [4, 8] m/s，`duration` [3, 6] s，`direction` [0, 360)°，`start_time` 段内 [30, 60] s。  
  后处理时该段真值风 = 本轮恒定风 + 1-cos 阵风（按 start_time/duration/magnitude/direction 生成）。
- **阵风段（OOD）**：metadata 中 `config_type: "gust"`, `gust_ood: true`，`gust`:  
  `magnitude` [9, 12] m/s，`duration` 在 [8, 12] s 与 [3, 6] s 中按需分配（强/长阵风），`direction` [0, 360)°，`start_time` 段内 [30, 60] s。  
  后处理同上，用 1-cos 生成 OOD 阵风真值。

---

**五、执行顺序（无人监管单脚本）**

建议按**数据集类型顺序**执行，便于管理和重跑：

1. **Train**：60 轮  
   - 每轮：写 XML（随机风场）→ 启 SITL → 起飞 → 执行 5 段（前 4 段稳态、第 5 段阵风 ID）→ 降落 → 关 SITL。  
   - 输出：`data/train/flight_0001.json` … `flight_0300.json`（及对应 metadata）。
2. **Val**：10 轮  
   - 每轮：同上，5 段为 4 稳态 + 1 阵风 ID；输出 `data/val/flight_0001.json` … `flight_0050.json`。
3. **Test-ID**：7 轮  
   - 前 6 轮：每轮 4 稳态 + 1 阵风 ID；第 7 轮：1 稳态 + 4 阵风 ID。  
   - 输出：`data/test_id/flight_0001.json` … `flight_0035.json`。
4. **Test-OOD**：3 轮  
   - 每轮 5 段均为阵风 OOD（metadata 带 gust_ood 与 9–12 m/s、8–12 s 等）；输出 `data/test_ood/flight_0001.json` … `flight_0015.json`。

每轮内段顺序（例如先 4 稳态再 1 阵风）可固定，也可随机排列；只要 metadata 中 `config_type` 与 `gust` 正确即可。

---

**六、参数汇总表（便于实现）**

| 参数               | 取值 |
|--------------------|------|
| 总 SITL 轮数       | 80   |
| 总段数（条数）     | 400  |
| 每轮段数           | 5    |
| 风速范围 (m/s)     | [6, 14] |
| 风向范围 (°)       | [0, 360) |
| 湍流              | 1.0 或 2.0 随机 |
| 直线速度 (m/s)     | [12, 18] |
| 直线时长 (s)       | [90, 150] |
| 盘旋半径 (m)       | [80, 150] |
| 盘旋时长 (s)       | [120, 180] |
| 8 字半径 (m)       | [60, 100] |
| 8 字时长 (s)       | [180, 240] |
| 爬升率 (m/s)       | [1, 3] 或 [-3, -1] |
| 阵风 ID 幅值 (m/s) | [4, 8] |
| 阵风 ID 时长 (s)   | [3, 6] |
| 阵风 OOD 幅值 (m/s)| [9, 12] |
| 阵风 OOD 时长 (s)  | [8, 12] 或 [3, 6] |

按上述执行即可在**一次 SITL 多段、无人监管**下，用 **80 次 SITL 启动** 完成全部 400 条数据，且满足训练/验证/测试-ID/测试-OOD 的条数与稳态/阵风比例要求。

---

#### 3.7.8 80 轮 SITL 全自动执行实现

下面给出**单脚本串行跑满 80 轮**的自动化实现思路与代码骨架，使 80 次重启 SITL 无需人工干预。

**1. 运行计划表（80 轮）**

预先构造一个「轮次计划」列表，每一项包含：数据集类型、该轮起始 flight_id、该轮 5 段中稳态/阵风的数量与类型（ID 或 OOD）。

```python
def build_80_run_schedule():
    """返回 80 轮的 (dataset_type, base_flight_id, segment_types) 列表。
    segment_types: 长度为 5 的列表，每项为 'steady' | 'gust_id' | 'gust_ood'
    """
    schedule = []
    # Train: 60 轮，每轮 4 稳态 + 1 阵风 ID
    for r in range(60):
        schedule.append(('train', 1 + r * 5, ['steady']*4 + ['gust_id']))
    # Val: 10 轮，每轮 4 稳态 + 1 阵风 ID
    for r in range(10):
        schedule.append(('val', 1 + r * 5, ['steady']*4 + ['gust_id']))
    # Test-ID: 前 6 轮 4 稳态 + 1 阵风，第 7 轮 1 稳态 + 4 阵风
    for r in range(6):
        schedule.append(('test_id', 1 + r * 5, ['steady']*4 + ['gust_id']))
    schedule.append(('test_id', 31, ['steady'] + ['gust_id']*4))
    # Test-OOD: 3 轮，每轮 5 段均为阵风 OOD
    for r in range(3):
        schedule.append(('test_ood', 1 + r * 5, ['gust_ood']*5))
    return schedule
```

**2. 单轮风场与段配置生成**

每轮开始时：随机生成**一个**风场（供该轮 5 段共用）；再根据该轮的 `segment_types` 为每一段生成机动参数（机动类型、时长、航向等），并给阵风段写上 `gust` / `gust_ood` 元数据。

```python
def generate_wind_config():
    """本轮风场（所有数据集统一范围）。"""
    speed = random.uniform(6, 14)
    direction = random.uniform(0, 360)
    wn = speed * np.cos(np.deg2rad(direction))
    we = speed * np.sin(np.deg2rad(direction))
    return {
        'wind_north': wn, 'wind_east': we, 'wind_down': 0.0,
        'wind_speed': speed, 'wind_direction': direction,
        'turbulence': random.choice(['light', 'moderate']),
    }

def generate_segment_configs_for_run(dataset_type, base_flight_id, segment_types):
    """根据本轮的 segment_types 生成 5 个段配置（含机动类型与参数、steady/gust 标记）。
    每段调用与 generate_flight_config 类似的逻辑，但 config_type 与 gust 按 segment_types[i] 设定。
    """
    segment_configs = []
    altitude = random.uniform(80, 120)
    for i, seg_type in enumerate(segment_types):
        cfg = generate_one_segment_config(
            flight_id=base_flight_id + i,
            dataset_type=dataset_type,
            config_type='gust' if seg_type != 'steady' else 'steady',
            gust_ood=(seg_type == 'gust_ood'),
            altitude=altitude,
        )
        segment_configs.append(cfg)
    return segment_configs
```

其中 `generate_one_segment_config` 与现有 `generate_flight_config` 类似：随机选择机动类型（straight_line/orbit/figure_eight/climb_descent）及对应参数；若为阵风则写入 `gust`（ID 用 magnitude [4,8]、duration [3,6]，OOD 用 [9,12]、[8,12] 等）。**风场先不填**，在调用 `run_multi_segment_session` 时用本轮的 `wind_config` 统一写入每段的 `wind_north/wind_east/wind_down`。

**3. 主循环：80 轮全自动**

单入口函数顺序执行 80 轮；每轮内：生成风场 → 生成 5 段配置 → 调用 `run_multi_segment_session`；**无论成功与否，在 finally 中关掉 SITL**，并写日志，再进入下一轮。

```python
async def run_full_400_automated(self, log_path=None):
    """80 轮全自动：每轮重启 SITL，多段采集，无人监管。"""
    import datetime
    log_path = log_path or (self.output_dir.parent / 'logs' / 'multi_segment_80runs.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    schedule = build_80_run_schedule()
    
    for run_index, (dataset_type, base_flight_id, segment_types) in enumerate(schedule):
        wind_config = generate_wind_config()
        segment_configs = generate_segment_configs_for_run(dataset_type, base_flight_id, segment_types)
        output_subdir = self.output_dir / dataset_type
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(log_path, 'a') as f:
                f.write(f"[{datetime.datetime.now().isoformat()}] Run {run_index+1}/80 {dataset_type} "
                        f"base_id={base_flight_id} wind=({wind_config['wind_speed']:.1f}m/s, {wind_config['wind_direction']:.0f}°)\n")
            await run_multi_segment_session(self, wind_config, segment_configs, output_subdir, base_flight_id)
            with open(log_path, 'a') as f:
                f.write(f"  -> OK\n")
        except Exception as e:
            with open(log_path, 'a') as f:
                f.write(f"  -> FAILED: {e}\n")
            # 可选：重试一次或仅记录后继续
        finally:
            self.stop_px4_sitl()
            await asyncio.sleep(2)
    
    with open(log_path, 'a') as f:
        f.write(f"[{datetime.datetime.now().isoformat()}] All 80 runs finished.\n")
```

**4. 超时与重试（可选）**

- 对单段或整轮加超时：`await asyncio.wait_for(run_multi_segment_session(...), timeout=round_duration_sec)`，超时后 catch `asyncio.TimeoutError`，在 finally 里 `stop_px4_sitl()`，写日志后继续下一轮。
- 某轮失败后可重试 1～2 次：在 except 里重试 `run_multi_segment_session`（同一 wind_config 与 segment_configs），再失败则写失败日志并进入下一轮。

**5. 启动方式（无人值守）**

在服务器上建议用 **nohup** 或 **tmux** 跑主入口，避免 SSH 断开导致中断：

```bash
cd ~/pirnn_dataset_generation/scripts
nohup python3 -u generate_dataset.py --mode multi_segment_80 > ../logs/automated_80runs.log 2>&1 &
# 或
tmux new -s collect
python3 -u generate_dataset.py --mode multi_segment_80
# Ctrl+B D 分离会话
```

脚本入口中根据 `--mode multi_segment_80` 调用 `asyncio.run(gen.run_full_400_automated())` 即可。这样 **80 轮、每轮重启 SITL** 即可全自动完成，无需人工参与。

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
    config = gen.generate_flight_config(1, 'test', 'steady')
    # 先写 JSBSim 风场再启动 SITL
    gen._apply_jsbsim_wind_for_flight(config)
    gen.start_px4_sitl()
    await gen.initialize_controllers()
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
