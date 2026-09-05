#!/usr/bin/env python3
"""
独立验证脚本：连接 PX4 SITL 的 MAVLink 通道，确认能否拿到
- ATTITUDE_TARGET
- POSITION_TARGET_LOCAL_NED
- HIGHRES_IMU
- NAV_CONTROLLER_OUTPUT

用法：
    1. 先启动 PX4 SITL（make px4_sitl jsbsim_rascal）
    2. 在另一个终端运行：
        python src/dataset_generation/scripts/tests/check_pymavlink_targets.py

       或指定其他端口：
        python check_pymavlink_targets.py --port 14550

注意：14540 已被 mavsdk_server 占用，本脚本默认监听 14550 (PX4 GCS 默认端口)。
若 14550 也被 QGC 占用，可以让 PX4 在启动时打开新端口。
"""

import argparse
import time
from collections import Counter

from pymavlink import mavutil


WANTED = {
    "ATTITUDE_TARGET",
    "POSITION_TARGET_LOCAL_NED",
    "HIGHRES_IMU",
    "NAV_CONTROLLER_OUTPUT",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=14550,
                    help="UDP 监听端口 (默认 14550)")
    ap.add_argument("--seconds", type=float, default=10.0,
                    help="监听时长（秒）")
    args = ap.parse_args()

    conn_str = f"udpin:0.0.0.0:{args.port}"
    print(f"[check] 连接 {conn_str}，监听 {args.seconds:.1f} 秒…")

    try:
        m = mavutil.mavlink_connection(conn_str)
    except Exception as e:
        print(f"[check] 连接失败: {e}")
        return 2

    counter: Counter = Counter()
    samples: dict = {}
    deadline = time.time() + args.seconds

    while time.time() < deadline:
        msg = m.recv_match(blocking=True, timeout=0.5)
        if msg is None:
            continue
        name = msg.get_type()
        counter[name] += 1
        if name in WANTED and name not in samples:
            samples[name] = msg.to_dict()

    print(f"\n[check] 监听结束，共收到 {sum(counter.values())} 条消息")
    print(f"[check] 消息类型直方图（top 15）:")
    for name, cnt in counter.most_common(15):
        flag = "  <-- 关注" if name in WANTED else ""
        print(f"  {name:32s} {cnt:6d}{flag}")

    missing = WANTED - set(samples.keys())
    print(f"\n[check] 关注消息覆盖情况:")
    for name in sorted(WANTED):
        if name in samples:
            sample = samples[name]
            keys = list(sample.keys())[:6]
            print(f"  [OK]    {name}  (示例字段: {keys}...)")
        else:
            print(f"  [MISS]  {name}")

    if missing:
        print(f"\n[check] 警告: 以下消息未收到 -> {missing}")
        print(f"[check] 可能原因:")
        print(f"  1) PX4 mavlink 实例未广播到端口 {args.port}")
        print(f"  2) 解锁/起飞前 ATTITUDE_TARGET 不会发送，请先 takeoff")
        print(f"  3) 端口被其他进程占用")
        return 1

    print(f"\n[check] 全部关注消息均已收到，可继续后续 plan 步骤")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
