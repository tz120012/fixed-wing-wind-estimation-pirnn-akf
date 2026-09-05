
# PX4-SITL + JSBSim 随机化方案实施指南

## 一、能力评估

### ✅ PX4-SITL + JSBSim 完全可以实现随机化方案！

**核心原因**：
1. JSBSim 是**完整的6自由度飞行动力学引擎**，支持任意轨迹
2. PX4自驾仪可以通过**MAVSDK/MAVROS**编程控制
3. 风场模型可以通过**JSBSim XML配置**或**实时注入**

---

## 二、实现方式对比

### 方式1：MAVSDK Python API（推荐）

**优点**：
- ✅ 高层接口，简单易用
- ✅ 支持offboard模式控制任意轨迹
- ✅ 可以实时获取传感器数据
- ✅ Python生态完善，便于数据处理

**示例代码**：

```python
import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, VelocityNedYaw
import numpy as np
import random

class RandomizedFlightGenerator:
    def __init__(self):
        self.drone = System()
        
    async def connect(self):
        """连接到PX4-SITL"""
        await self.drone.connect(system_address="udp://:14540")
        
        print("等待无人机连接...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("无人机已连接!")
                break
    
    async def arm_and_takeoff(self, altitude):
        """解锁并起飞"""
        print("-- 解锁")
        await self.drone.action.arm()
        
        print(f"-- 起飞到 {altitude}m")
        await self.drone.action.set_takeoff_altitude(altitude)
        await self.drone.action.takeoff()
        
        # 等待达到目标高度
        await asyncio.sleep(10)
    
    async def fly_straight_line_random(self):
        """随机化直线飞行"""
        # 随机参数
        heading = random.uniform(0, 360)
        altitude = random.uniform(50, 150)
        speed = random.uniform(12, 18)
        duration = 120  # 2分钟
        
        print(f"直线飞行 - 航向:{heading:.1f}°, 高度:{altitude:.1f}m, 速度:{speed:.1f}m/s")
        
        # 启动offboard模式
        await self.drone.offboard.set_velocity_ned(
            VelocityNedYaw(0.0, 0.0, 0.0, heading)
        )
        await self.drone.offboard.start()
        
        # 计算速度分量
        vn = speed * np.cos(np.deg2rad(heading))
        ve = speed * np.sin(np.deg2rad(heading))
        vd = 0.0
        
        # 执行直线飞行
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < duration:
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(vn, ve, vd, heading)
            )
            await asyncio.sleep(0.1)  # 10Hz控制频率
        
        print("直线飞行完成")
    
    async def fly_orbit_random(self):
        """随机化盘旋飞行"""
        # 随机参数
        radius = random.uniform(50, 150)
        altitude = random.uniform(80, 120)
        angular_velocity = random.uniform(0.1, 0.3)  # rad/s
        direction = random.choice([1, -1])  # 1=顺时针, -1=逆时针
        duration = 180  # 3分钟
        
        print(f"盘旋飞行 - 半径:{radius:.1f}m, 高度:{altitude:.1f}m, 方向:{'顺时针' if direction==1 else '逆时针'}")
        
        # 获取当前位置作为圆心
        async for position in self.drone.telemetry.position():
            center_lat = position.latitude_deg
            center_lon = position.longitude_deg
            break
        
        start_time = asyncio.get_event_loop().time()
        t = 0
        
        while (asyncio.get_event_loop().time() - start_time) < duration:
            # 计算圆周上的目标位置
            angle = direction * angular_velocity * t
            
            # 相对位置 (NED坐标系)
            north_offset = radius * np.cos(angle)
            east_offset = radius * np.sin(angle)
            
            # 转换为经纬度并发送
            # 注意: 这里需要经纬度转换函数
            target_lat = center_lat + (north_offset / 111320.0)  # 简化转换
            target_lon = center_lon + (east_offset / (111320.0 * np.cos(np.deg2rad(center_lat))))
            
            await self.drone.action.goto_location(target_lat, target_lon, altitude, 0)
            
            await asyncio.sleep(0.1)
            t += 0.1
        
        print("盘旋飞行完成")
    
    async def log_telemetry(self, duration, log_file):
        """记录遥测数据"""
        data_log = []
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < duration:
            # 获取传感器数据
            position = None
            velocity = None
            imu = None
            
            async for pos in self.drone.telemetry.position():
                position = pos
                break
            
            async for vel in self.drone.telemetry.velocity_ned():
                velocity = vel
                break
            
            async for imu_data in self.drone.telemetry.imu():
                imu = imu_data
                break
            
            # 记录数据
            if all([position, velocity, imu]):
                data_log.append({
                    'time': asyncio.get_event_loop().time() - start_time,
                    'position': {
                        'lat': position.latitude_deg,
                        'lon': position.longitude_deg,
                        'alt': position.absolute_altitude_m
                    },
                    'velocity': {
                        'north': velocity.north_m_s,
                        'east': velocity.east_m_s,
                        'down': velocity.down_m_s
                    },
                    'imu': {
                        'acc_x': imu.acceleration_forward_m_s2,
                        'acc_y': imu.acceleration_right_m_s2,
                        'acc_z': imu.acceleration_down_m_s2,
                        'gyro_x': imu.angular_velocity_forward_rad_s,
                        'gyro_y': imu.angular_velocity_right_rad_s,
                        'gyro_z': imu.angular_velocity_down_rad_s
                    }
                })
            
            await asyncio.sleep(0.02)  # 50Hz采样
        
        # 保存数据
        import json
        with open(log_file, 'w') as f:
            json.dump(data_log, f)
        
        print(f"数据已保存至 {log_file}")


async def generate_300_flights():
    """生成300次随机飞行"""
    generator = RandomizedFlightGenerator()
    await generator.connect()
    
    for i in range(300):
        print(f"\n========== 飞行 {i+1}/300 ==========")
        
        # 随机选择飞行类型
        flight_type = random.choice(['straight', 'orbit', 'figure8', 'climb'])
        
        # 起飞
        takeoff_alt = random.uniform(50, 150)
        await generator.arm_and_takeoff(takeoff_alt)
        
        # 执行对应机动
        if flight_type == 'straight':
            await generator.fly_straight_line_random()
        elif flight_type == 'orbit':
            await generator.fly_orbit_random()
        # ... 其他机动类型
        
        # 记录数据
        await generator.log_telemetry(
            duration=120, 
            log_file=f'data/flight_{i:03d}.json'
        )
        
        # 降落
        await generator.drone.action.land()
        await asyncio.sleep(10)
        
        print(f"飞行 {i+1} 完成")


# 运行
if __name__ == "__main__":
    asyncio.run(generate_300_flights())
```

---

### 方式2：MAVROS + ROS（更强大）

**优点**：
- ✅ ROS生态完善，工具丰富
- ✅ 支持Gazebo可视化
- ✅ 便于多机仿真
- ✅ 可以使用rosbag记录数据

**示例代码**：

```python
#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from sensor_msgs.msg import Imu, NavSatFix
import numpy as np
import random

class MAVROSFlightController:
    def __init__(self):
        rospy.init_node('random_flight_generator')
        
        # 订阅者
        self.state_sub = rospy.Subscriber('/mavros/state', State, self.state_callback)
        self.imu_sub = rospy.Subscriber('/mavros/imu/data', Imu, self.imu_callback)
        self.gps_sub = rospy.Subscriber('/mavros/global_position/global', NavSatFix, self.gps_callback)
        
        # 发布者
        self.local_pos_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.vel_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        
        # 服务
        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        
        self.current_state = State()
        self.current_imu = None
        self.current_gps = None
        
        self.rate = rospy.Rate(20)  # 20Hz
    
    def state_callback(self, msg):
        self.current_state = msg
    
    def imu_callback(self, msg):
        self.current_imu = msg
    
    def gps_callback(self, msg):
        self.current_gps = msg
    
    def wait_for_connection(self):
        """等待与FCU连接"""
        while not rospy.is_shutdown() and not self.current_state.connected:
            self.rate.sleep()
        rospy.loginfo("已连接到FCU")
    
    def set_offboard_mode(self):
        """切换到OFFBOARD模式"""
        # 先发送几个setpoint
        for i in range(100):
            pose = PoseStamped()
            pose.pose.position.z = 2
            self.local_pos_pub.publish(pose)
            self.rate.sleep()
        
        # 切换模式
        offb_set_mode = SetMode()
        offb_set_mode.custom_mode = 'OFFBOARD'
        
        if self.set_mode_client.call(offb_set_mode).mode_sent:
            rospy.loginfo("OFFBOARD模式已启用")
            return True
        return False
    
    def arm(self):
        """解锁"""
        arm_cmd = CommandBool()
        arm_cmd.value = True
        
        if self.arming_client.call(arm_cmd).success:
            rospy.loginfo("无人机已解锁")
            return True
        return False
    
    def fly_straight_random(self, duration=120):
        """随机直线飞行"""
        heading = random.uniform(0, 360)
        altitude = random.uniform(50, 150)
        speed = random.uniform(12, 18)
        
        rospy.loginfo(f"直线飞行: 航向={heading:.1f}°, 高度={altitude:.1f}m, 速度={speed:.1f}m/s")
        
        # 计算速度分量
        vx = speed * np.cos(np.deg2rad(heading))
        vy = speed * np.sin(np.deg2rad(heading))
        vz = 0.0
        
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time).to_sec() < duration:
            vel_cmd = TwistStamped()
            vel_cmd.header.stamp = rospy.Time.now()
            vel_cmd.twist.linear.x = vx
            vel_cmd.twist.linear.y = vy
            vel_cmd.twist.linear.z = vz
            
            self.vel_pub.publish(vel_cmd)
            self.rate.sleep()
    
    def fly_orbit_random(self, duration=180):
        """随机盘旋飞行"""
        radius = random.uniform(50, 150)
        altitude = random.uniform(80, 120)
        angular_vel = random.uniform(0.1, 0.3)
        direction = random.choice([1, -1])
        
        rospy.loginfo(f"盘旋飞行: 半径={radius:.1f}m, 高度={altitude:.1f}m")
        
        start_time = rospy.Time.now()
        t = 0
        
        while (rospy.Time.now() - start_time).to_sec() < duration:
            angle = direction * angular_vel * t
            
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.pose.position.x = radius * np.cos(angle)
            pose.pose.position.y = radius * np.sin(angle)
            pose.pose.position.z = altitude
            
            self.local_pos_pub.publish(pose)
            
            self.rate.sleep()
            t += 0.05


# 使用示例
if __name__ == '__main__':
    try:
        controller = MAVROSFlightController()
        controller.wait_for_connection()
        controller.set_offboard_mode()
        controller.arm()
        
        # 执行随机飞行
        for i in range(300):
            rospy.loginfo(f"========== 飞行 {i+1}/300 ==========")
            
            flight_type = random.choice(['straight', 'orbit'])
            if flight_type == 'straight':
                controller.fly_straight_random(duration=120)
            else:
                controller.fly_orbit_random(duration=180)
            
            rospy.loginfo(f"飞行 {i+1} 完成")
        
    except rospy.ROSInterruptException:
        pass
```

---

## 三、风场注入方法

### 方法1：修改JSBSim XML配置文件（启动前设置）

**位置**：`Tools/sitl_gazebo/models/rascal/rascal_jsbsim.xml`

```xml
<?xml version="1.0"?>
<fdm_config name="rascal" version="2.0" release="ALPHA">
  
  <!-- ... 其他配置 ... -->
  
  <!-- 大气环境配置 -->
  <atmosphere>
    <!-- 添加恒定风 -->
    <winds>
      <wind_north unit="FT/SEC"> 26.25 </wind_north>  <!-- 8 m/s -->
      <wind_east unit="FT/SEC"> 32.81 </wind_east>    <!-- 10 m/s -->
      <wind_down unit="FT/SEC"> 0.0 </wind_down>
    </winds>
    
    <!-- Dryden湍流模型 -->
    <turbulence>
      <turb_type> ttMilspec </turb_type>
      <turbulence_gain> 1.0 </turbulence_gain>  <!-- 湍流强度 -->
      <wind_speed_20ft> 15.0 </wind_speed_20ft> <!-- 20英尺处风速(kt) -->
      <probability_of_exceedence> 5 </probability_of_exceedence> <!-- 5%超越概率=中度湍流 -->
    </turbulence>
  </atmosphere>
  
</fdm_config>
```

**每次飞行随机化风场的脚本**：

```python
import xml.etree.ElementTree as ET
import random
import subprocess

def modify_wind_config(wind_north, wind_east, wind_down, turb_intensity):
    """修改JSBSim风场配置"""
    xml_path = 'Tools/sitl_gazebo/models/rascal/rascal_jsbsim.xml'
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 修改风速
    winds = root.find('.//winds')
    winds.find('wind_north').text = f'{wind_north * 3.28084:.2f}'  # m/s转ft/s
    winds.find('wind_east').text = f'{wind_east * 3.28084:.2f}'
    winds.find('wind_down').text = f'{wind_down * 3.28084:.2f}'
    
    # 修改湍流强度
    turbulence = root.find('.//turbulence')
    turbulence.find('turbulence_gain').text = f'{turb_intensity:.2f}'
    
    tree.write(xml_path)
    print(f"已设置风场: N={wind_north:.1f}, E={wind_east:.1f}, D={wind_down:.1f} m/s, 湍流={turb_intensity}")

def launch_px4_sitl():
    """启动PX4 SITL"""
    cmd = "make px4_sitl jsbsim_rascal"
    subprocess.Popen(cmd, shell=True, cwd='/path/to/PX4-Autopilot')


# 300次飞行循环
for i in range(300):
    # 随机风场参数
    wind_speed = random.uniform(6, 14)
    wind_dir = random.uniform(0, 360)
    wind_north = wind_speed * np.cos(np.deg2rad(wind_dir))
    wind_east = wind_speed * np.sin(np.deg2rad(wind_dir))
    wind_down = random.uniform(-0.5, 0.5)
    turb_intensity = random.choice([1.0, 2.0])  # 轻度/中度
    
    # 修改配置
    modify_wind_config(wind_north, wind_east, wind_down, turb_intensity)
    
    # 启动仿真
    launch_px4_sitl()
    time.sleep(5)  # 等待启动
    
    # 执行飞行（使用MAVSDK或MAVROS）
    # ... 飞行代码 ...
    
    # 关闭仿真
    subprocess.run("pkill -9 px4", shell=True)
    time.sleep(2)
```

**优点**：
- ✅ 完全控制风场参数
- ✅ 支持Dryden湍流模型

**缺点**：
- ❌ 需要每次重启仿真（耗时）
- ❌ 不支持风场动态变化

---

### 方法2：运行时通过MAVLINK注入风场（推荐）

**PX4参数设置**：

```bash
# 在PX4控制台中设置
param set SIM_WIND_SPEED 10.0   # 风速 m/s
param set SIM_WIND_DIRECTION 90 # 风向角 度
param save
```

**Python自动化脚本**：

```python
from pymavlink import mavutil
import random
import time

def set_wind_via_mavlink(wind_north, wind_east, wind_down):
    """通过MAVLINK设置风场"""
    # 连接到PX4
    master = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    master.wait_heartbeat()
    
    # 计算风速和风向
    wind_speed = np.sqrt(wind_north**2 + wind_east**2)
    wind_dir = np.rad2deg(np.arctan2(wind_east, wind_north))
    
    # 设置参数
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        b'SIM_WIND_SPEED',
        wind_speed,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        b'SIM_WIND_DIR',
        wind_dir,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32
    )
    
    print(f"已设置风场: 速度={wind_speed:.1f}m/s, 方向={wind_dir:.1f}°")

# 使用示例
for i in range(300):
    # 随机风场
    wind_speed = random.uniform(6, 14)
    wind_dir = random.uniform(0, 360)
    wind_north = wind_speed * np.cos(np.deg2rad(wind_dir))
    wind_east = wind_speed * np.sin(np.deg2rad(wind_dir))
    
    # 运行时设置
    set_wind_via_mavlink(wind_north, wind_east, 0)
    
    # 执行飞行
    # ...
```

**优点**：
- ✅ 无需重启仿真
- ✅ 可以实时改变风场
- ✅ 速度快

**缺点**：
- ⚠️ 仅支持恒定风（不支持Dryden湍流）
- ⚠️ 需要PX4版本 >= 1.12

---

### 方法3：JSBSim脚本注入（最灵活）

**JSBSim支持通过脚本动态注入阵风**：

```xml
<!-- gust_injection.xml -->
<runscript xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
           xsi:noNamespaceSchemaLocation="http://jsbsim.sf.net/JSBSimScript.xsd"
           name="Gust Injection Test">
  
  <use aircraft="rascal" initialize="reset"/>
  
  <run start="0.0" end="300" dt="0.00833333">
    
    <!-- 在200秒时注入6m/s阵风 -->
    <event name="inject_gust" persistent="false">
      <condition> simulation/sim-time-sec >= 200 </condition>
      <set name="atmosphere/gust-north-fps" value="19.685"/>  <!-- 6 m/s -->
      <set name="atmosphere/gust-east-fps" value="0.0"/>
      <notify/>
    </event>
    
    <!-- 在205秒时移除阵风 -->
    <event name="remove_gust" persistent="false">
      <condition> simulation/sim-time-sec >= 205 </condition>
      <set name="atmosphere/gust-north-fps" value="0.0"/>
      <notify/>
    </event>
    
  </run>
  
</runscript>
```

**Python调用JSBSim脚本**：

```python
import jsbsim

def run_flight_with_gust(wind_params, gust_params):
    """运行带阵风的飞行仿真"""
    fdm = jsbsim.FGFDMExec(root_dir='/path/to/jsbsim')
    fdm.load_model('rascal')
    fdm.load_ic('reset.xml', useStoredPath=True)
    
    # 设置初始风场
    fdm['atmosphere/wind-north-fps'] = wind_params['north'] * 3.28084
    fdm['atmosphere/wind-east-fps'] = wind_params['east'] * 3.28084
    
    # 设置Dryden湍流
    fdm['atmosphere/turb-type'] = 4  # ttMilspec
    fdm['atmosphere/turbulence-magnitude-norm'] = wind_params['turb_intensity']
    
    # 运行仿真
    fdm.run_ic()
    
    data_log = []
    while fdm['simulation/sim-time-sec'] < 300:
        # 在指定时间注入阵风
        if 200 <= fdm['simulation/sim-time-sec'] < 205:
            fdm['atmosphere/gust-north-fps'] = gust_params['magnitude'] * 3.28084
        else:
            fdm['atmosphere/gust-north-fps'] = 0.0
        
        # 采集数据
        data_log.append({
            'time': fdm['simulation/sim-time-sec'],
            'position': [
                fdm['position/lat-gc-deg'],
                fdm['position/long-gc-deg'],
                fdm['position/h-sl-meters']
            ],
            'velocity': [
                fdm['velocities/v-north-fps'] / 3.28084,
                fdm['velocities/v-east-fps'] / 3.28084,
                fdm['velocities/v-down-fps'] / 3.28084
            ],
            'wind': [
                fdm['atmosphere/total-wind-north-fps'] / 3.28084,
                fdm['atmosphere/total-wind-east-fps'] / 3.28084,
                fdm['atmosphere/total-wind-down-fps'] / 3.28084
            ]
        })
        
        fdm.run()
    
    return data_log
```

**优点**：
- ✅ 完全控制风场，包括阵风注入
- ✅ 支持Dryden湍流模型
- ✅ 可以记录真实风场（ground truth）

**缺点**：
- ❌ 需要Python JSBSim绑定
- ❌ 学习曲线较陡

---

## 四、完整工作流程

### 推荐方案：MAVSDK + MAVLINK风场注入

```python
import asyncio
from mavsdk import System
from pymavlink import mavutil
import random
import numpy as np
import json

class PX4RandomFlightDataGenerator:
    def __init__(self):
        self.drone = System()
        self.mavlink_conn = None
        
    async def setup(self):
        """初始化连接"""
        # MAVSDK连接（用于飞行控制）
        await self.drone.connect(system_address="udp://:14540")
        print("等待连接...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("MAVSDK已连接")
                break
        
        # PyMAVLink连接（用于设置风场）
        self.mavlink_conn = mavutil.mavlink_connection('udp:127.0.0.1:14550')
        self.mavlink_conn.wait_heartbeat()
        print("PyMAVLink已连接")
    
    def set_random_wind(self):
        """设置随机风场"""
        wind_speed = random.uniform(6, 14)
        wind_dir = random.uniform(0, 360)
        
        # 设置风速
        self.mavlink_conn.mav.param_set_send(
            self.mavlink_conn.target_system,
            self.mavlink_conn.target_component,
            b'SIM_WIND_SPEED',
            wind_speed,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        
        # 设置风向
        self.mavlink_conn.mav.param_set_send(
            self.mavlink_conn.target_system,
            self.mavlink_conn.target_component,
            b'SIM_WIND_DIR',
            wind_dir,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32
        )
        
        print(f"风场: {wind_speed:.1f}m/s @ {wind_dir:.1f}°")
        return wind_speed, wind_dir
    
    async def execute_random_flight(self, flight_id):
        """执行单次随机飞行"""
        print(f"\n{'='*50}")
        print(f"飞行 #{flight_id}")
        print(f"{'='*50}")
        
        # 1. 设置随机风场
        wind_speed, wind_dir = self.set_random_wind()
        await asyncio.sleep(1)
        
        # 2. 随机飞行参数
        flight_type = random.choice(['straight', 'orbit'])
        heading = random.uniform(0, 360)
        altitude = random.uniform(50, 150)
        
        # 3. 起飞
        print(f"起飞到 {altitude:.1f}m")
        await self.drone.action.set_takeoff_altitude(altitude)
        await self.drone.action.arm()
        await self.drone.action.takeoff()
        await asyncio.sleep(15)
        
        # 4. 执行机动
        if flight_type == 'straight':
            await self.fly_straight(heading, altitude, duration=120)
        else:
            await self.fly_orbit(altitude, duration=180)
        
        # 5. 降落
        print("降落...")
        await self.drone.action.land()
        await asyncio.sleep(10)
        
        print(f"飞行 #{flight_id} 完成")
    
    async def fly_straight(self, heading, altitude, duration):
        """直线飞行"""
        print(f"直线飞行: 航向={heading:.1f}°, 持续{duration}秒")
        # ... (参考前面的代码)
    
    async def fly_orbit(self, altitude, duration):
        """盘旋飞行"""
        print(f"盘旋飞行: 持续{duration}秒")
        # ... (参考前面的代码)
    
    async def generate_dataset(self, num_flights=300):
        """生成完整数据集"""
        await self.setup()
        
        for i in range(num_flights):
            try:
                await self.execute_random_flight(i+1)
            except Exception as e:
                print(f"飞行 #{i+1} 出错: {e}")
                continue
        
        print("\n数据集生成完成!")


# 运行
if __name__ == "__main__":
    generator = PX4RandomFlightDataGenerator()
    asyncio.run(generator.generate_dataset(num_flights=300))
```

---

## 五、潜在问题与解决方案

### 问题1：Dryden湍流模型支持

**问题**：PX4 SITL的风场参数只支持恒定风，不支持Dryden湍流

**解决方案**：
1. **修改JSBSim XML**（方法1）：每次启动前修改配置文件
2. **使用JSBSim Python API**（方法3）：直接控制JSBSim引擎
3. **妥协方案**：在恒定风基础上，手动叠加高频噪声模拟湍流

```python
def add_synthetic_turbulence(base_wind, intensity=1.0, dt=0.02):
    """在恒定风上叠加合成湍流"""
    # Dryden湍流功率谱密度近似
    turbulence = intensity * np.random.randn(3) * np.sqrt(dt)
    
    return base_wind + turbulence
```

---

### 问题2：阵风注入

**问题**：运行时无法通过MAVLINK注入1-cosine离散阵风

**解决方案**：
1. **使用JSBSim脚本**（推荐）
2. **在控制层模拟**：在offboard控制中叠加阵风扰动

```python
async def inject_gust_at_runtime(self, gust_time, magnitude, duration):
    """在指定时间注入模拟阵风"""
    start_time = asyncio.get_event_loop().time()
    
    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # 检查是否到达阵风时间
        if gust_time <= elapsed < (gust_time + duration):
            # 1-cosine脉冲
            t_gust = elapsed - gust_time
            gust_profile = magnitude * 0.5 * (1 - np.cos(2*np.pi*t_gust/duration))
            
            # 叠加到控制指令
            # ... (修改velocity setpoint)
        
        await asyncio.sleep(0.02)
```

---

### 问题3：仿真速度

**问题**：300次飞行 × 2-5分钟 = 10-25小时实时运行

**解决方案**：
1. **加速仿真**：PX4支持快于实时运行

```bash
# 启动时设置加速因子
make px4_sitl jsbsim_rascal SPEEDUP=10
```

```python
# 或通过参数设置
master.mav.param_set_send(
    master.target_system,
    master.target_component,
    b'SIM_SPEED_FACTOR',
    10.0,  # 10倍速
    mavutil.mavlink.MAV_PARAM_TYPE_REAL32
)
```

2. **并行仿真**：运行多个SITL实例

```bash
# 终端1: 实例1
HEADLESS=1 make px4_sitl_default jsbsim_rascal

# 终端2: 实例2 (改端口)
PX4_SIM_PORT_BASE=14560 HEADLESS=1 make px4_sitl_default jsbsim_rascal
```

---

### 问题4：数据记录

**问题**：需要同步记录传感器数据和真实风场

**解决方案**：
1. **使用uORB日志**：PX4内置日志系统

```bash
# 启用所有传感器日志
param set SDLOG_PROFILE 1
param set SDLOG_MODE 2  # 从启动开始记录
```

然后用`pyulog`解析：

```python
from pyulog import ULog

ulog = ULog('log_001_2026-01-13-12-00-00.ulg')

# 提取GPS数据
gps_data = ulog.get_dataset('vehicle_gps_position').data

# 提取IMU数据
imu_data = ulog.get_dataset('sensor_combined').data
```

2. **自定义ROS bag记录**（如果使用MAVROS）

```bash
rosbag record /mavros/imu/data /mavros/global_position/raw /mavros/local_position/velocity
```

---

## 六、最终推荐方案

### 方案A：快速原型（2-3天实现）

```
工具栈:
├── PX4-SITL + JSBSim (仿真引擎)
├── MAVSDK Python (飞行控制)
├── PyMAVLink (风场设置)
└── 加速因子 = 5x (总耗时约5小时)

风场模型:
├── 恒定风 (通过MAVLINK实时设置)
├── 手动合成湍流 (高频噪声叠加)
└── 控制层模拟阵风

数据记录:
└── MAVSDK telemetry订阅 + JSON保存
```

**预计时间**：
- 环境搭建: 0.5天
- 代码开发: 1天
- 数据生成: 5小时（5倍速）
- 调试优化: 0.5天

---

### 方案B：完整实现（1-2周）

```
工具栈:
├── PX4-SITL + JSBSim (仿真引擎)
├── ROS + MAVROS (中间层)
├── JSBSim Python API (风场精确控制)
└── 多实例并行 (3个实例同时运行)

风场模型:
├── 修改JSBSim XML (Dryden湍流)
├── JSBSim脚本 (阵风注入)
└── 风切变剖面 (对数模型)

数据记录:
├── rosbag (所有ROS话题)
└── uORB日志 (PX4内部状态)
```

**预计时间**：
- 环境搭建: 2天
- JSBSim深度定制: 3天
- 多实例并行配置: 2天
- 数据生成: 3小时（3实例并行+5倍速）
- 数据后处理: 2天

---

## 七、快速开始指南

### 1. 安装依赖

```bash
# PX4固件
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
bash ./Tools/setup/ubuntu.sh

# JSBSim
sudo apt install libjsbsim-dev

# Python依赖
pip3 install mavsdk pymavlink numpy
```

### 2. 测试仿真

```bash
cd ~/PX4-Autopilot
make px4_sitl jsbsim_rascal
```

### 3. 运行示例脚本

```python
# test_random_flight.py
import asyncio
from mavsdk import System

async def test():
    drone = System()
    await drone.connect(system_address="udp://:14540")
    
    print("等待连接...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("已连接!")
            break
    
    # 简单测试
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(10)
    await drone.action.land()

asyncio.run(test())
```

---

## 八、总结

### ✅ 可行性结论

**PX4-SITL + JSBSim 完全可以实现300次随机化飞行数据采集**，但需要注意：

| 需求 | 实现难度 | 推荐方案 |
|------|----------|----------|
| 任意轨迹 | ⭐ 简单 | MAVSDK Offboard模式 |
| 恒定风 | ⭐ 简单 | MAVLINK参数设置 |
| Dryden湍流 | ⭐⭐ 中等 | 修改JSBSim XML |
| 阵风注入 | ⭐⭐⭐ 困难 | JSBSim脚本 或 控制层模拟 |
| 加速仿真 | ⭐ 简单 | SIM_SPEED_FACTOR参数 |
| 并行仿真 | ⭐⭐ 中等 | 多端口配置 |

### 🚀 行动建议

1. **第1周**：实现方案A（快速原型）
   - 验证300次飞行可行性
   - 生成基础数据集

2. **第2-3周**：升级到方案B（如需要）
   - 添加Dryden湍流
   - 实现阵风注入

3. **第4周**：数据后处理和验证
   - 检查数据多样性
   - 开始PIRNN训练

需要我提供完整的可运行代码仓库吗？

