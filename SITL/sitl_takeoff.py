#!/usr/bin/env python3.8
"""
SITL 自动起飞脚本（Python 3.8 + MAVSDK）
用法: python3.8 sitl_takeoff.py [target_altitude_m]

代理设置必须在所有 import 之前清除（gRPC 在导入时读取环境变量）。
"""
import os, sys

# ── 代理清理（必须在最前）─────────────────────────────────────────────
for v in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
    os.environ[v] = ''
os.environ['grpc_proxy'] = ''
os.environ['no_grpc_proxy'] = '*'
os.environ['no_proxy'] = os.environ['NO_PROXY'] = '127.0.0.1,localhost,::1,0.0.0.0'
os.environ['GRPC_VERBOSITY'] = 'ERROR'


import asyncio

TARGET_ALT = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0

async def main():
    from mavsdk import System

    print(f"[TAKEOFF] 目标高度: {TARGET_ALT}m")
    drone = System(mavsdk_server_address='localhost', port=50053)
    await drone.connect(system_address="udp://:14540")

    print("[TAKEOFF] 等待连接...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[TAKEOFF] ✓ 已连接")
            break

    print("[TAKEOFF] 等待 GPS+EKF 收敛...")
    async for h in drone.telemetry.health():
        gps_ok = h.is_global_position_ok and h.is_home_position_ok
        ekf_ok = h.is_local_position_ok
        print(f"  GPS={gps_ok} EKF={ekf_ok} ARMABLE={h.is_armable}")
        if gps_ok and ekf_ok:
            print("[TAKEOFF] ✓ GPS+EKF 就绪")
            break
        await asyncio.sleep(2)

    # 等待可以解锁
    print("[TAKEOFF] 等待可解锁...")
    async for h in drone.telemetry.health():
        if h.is_armable:
            print("[TAKEOFF] ✓ 可解锁")
            break
        await asyncio.sleep(2)

    print("[TAKEOFF] ARM...")
    for i in range(6):
        try:
            await drone.action.arm()
            print("[TAKEOFF] ✓ 已解锁")
            break
        except Exception as e:
            print(f"  解锁第 {i+1} 次: {e}")
            await asyncio.sleep(5)

    print(f"[TAKEOFF] 起飞到 {TARGET_ALT}m...")
    await drone.action.set_takeoff_altitude(TARGET_ALT)
    await drone.action.takeoff()

    print("[TAKEOFF] 监测高度...")
    deadline = asyncio.get_event_loop().time() + 120
    async for pos in drone.telemetry.position():
        alt = pos.relative_altitude_m
        if alt is not None:
            print(f"  高度: {alt:.1f}m")
            if alt >= TARGET_ALT - 5:
                print(f"[TAKEOFF] ✓ 达到目标高度 {alt:.1f}m")
                break
        if asyncio.get_event_loop().time() > deadline:
            print("[TAKEOFF] 超时")
            break
        await asyncio.sleep(3)

    # 切换到 LOITER 让飞机保持绕圈
    print("[TAKEOFF] 切换 LOITER 模式...")
    try:
        await drone.action.hold()
        print("[TAKEOFF] ✓ LOITER 模式已激活，飞机正在绕圈")
    except Exception as e:
        print(f"[TAKEOFF] LOITER: {e}")

    print("[TAKEOFF] 飞机已就绪，评估器可以开始")

asyncio.run(main())
