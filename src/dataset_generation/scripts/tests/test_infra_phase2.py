"""Phase 2 单元测试：lib/infra 异常分类与生命周期。

不依赖 PX4 SITL 或 mavsdk_server 实际启动；用 monkey-patch 模拟外部进程。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "dataset_generation"))

from lib.config import load_runtime_config  # noqa: E402
from lib.errors import SitlCrashed, SitlStartFailed, MavsdkServerDead, JsbsimBridgeMissing  # noqa: E402
from lib.infra import Px4SitlProcess, MavsdkServerProcess, JsbsimBridgeConfig  # noqa: E402
from lib.infra.jsbsim_bridge import find_latest_wind_truth_csv  # noqa: E402


def _runtime_with_short_sleeps():
    """加载 runtime.yaml 但把所有 sleep 调到 0，避免测试拖时间。"""
    rc = load_runtime_config()
    return rc.with_overrides(
        sleeps={
            "px4_post_start_s": 0.05,
            "ekf_warmup_s": 0.05,
            "between_segments_s": 0.0,
            "between_sorties_s": 0.0,
            "pkill_grace_s": 0.05,
            "cleanup_post_stop_s": 0.05,
            "pre_sortie_warmup_s": 0.0,
            "segment_retry_backoff_s": 0.0,
        },
        timeouts={
            "px4_terminate_s": 0.5,
            "px4_kill_wait_s": 0.5,
            "pkill_timeout_s": 0.5,
            "mavsdk_health_s": 0.5,
            "arm_check_s": 1.0,
            "local_position_s": 1.0,
            "takeoff_total_s": 1.0,
            "round_total_s": 1.0,
            "gps_subscribe_s": 0.5,
            "bridge_ready_s": 0.5,
        },
        retries={
            "mavsdk_connect": 1,
            "arm": 1,
            "takeoff": 1,
            "segment_sample": 1,
            "jsbsim_verify": 1,
            "gps_verify": 1,
        },
    )


async def test_px4_start_failed_raises_when_make_exits():
    """PX4 进程在 px4_post_start 期间退出 → SitlStartFailed。"""
    rc = _runtime_with_short_sleeps()

    # monkey-patch subprocess.Popen：返回一个立即退出的 Popen
    real_popen = subprocess.Popen

    def fake_popen(*args, **kwargs):
        # 用 /bin/true 模拟立即退出
        kwargs.pop("env", None)
        return real_popen(["true"], stdout=kwargs.get("stdout"), stderr=kwargs.get("stderr"), start_new_session=True)

    import lib.infra.px4 as px4_mod
    orig_popen = px4_mod.subprocess.Popen
    px4_mod.subprocess.Popen = fake_popen
    try:
        # 同时 monkey-patch clear_px4_rootfs_state 避免触碰真实文件系统
        orig_clear = px4_mod.clear_px4_rootfs_state
        px4_mod.clear_px4_rootfs_state = lambda *a, **kw: None
        try:
            px4 = Px4SitlProcess(px4_root=Path("/tmp"), runtime=rc)
            try:
                await px4.start()
                raised = False
            except SitlStartFailed as e:
                raised = True
                assert "exit_code" in e.context
            assert raised, "SitlStartFailed 未抛出"
            print("[OK] test_px4_start_failed_raises_when_make_exits")
        finally:
            px4_mod.clear_px4_rootfs_state = orig_clear
    finally:
        px4_mod.subprocess.Popen = orig_popen


async def test_px4_raise_if_dead_after_crash():
    """运行中检测到 PX4 已退出 → SitlCrashed。"""
    rc = _runtime_with_short_sleeps()
    px4 = Px4SitlProcess(px4_root=Path("/tmp"), runtime=rc)
    px4.proc = subprocess.Popen(["true"])  # 立即退出
    px4.proc.wait()
    raised = False
    try:
        await px4.raise_if_dead()
    except SitlCrashed:
        raised = True
    assert raised, "SitlCrashed 未抛出"
    print("[OK] test_px4_raise_if_dead_after_crash")


async def test_mavsdk_server_no_executable_does_not_raise():
    """mavsdk_server 找不到可执行文件 → 静默返回（让上层回退到 SDK 自启）。"""
    rc = _runtime_with_short_sleeps()
    srv = MavsdkServerProcess(runtime=rc)

    # 强制 find_mavsdk_server_executable 返回 None
    import lib.infra.mavsdk_server as ms_mod
    orig = ms_mod.find_mavsdk_server_executable
    ms_mod.find_mavsdk_server_executable = lambda: None
    try:
        await srv.start()
        assert srv.proc is None
        assert not srv.is_running()
        print("[OK] test_mavsdk_server_no_executable_does_not_raise")
    finally:
        ms_mod.find_mavsdk_server_executable = orig


async def test_mavsdk_server_immediate_exit_raises():
    """mavsdk_server 启动后立即退出 → MavsdkServerDead。"""
    rc = _runtime_with_short_sleeps()
    srv = MavsdkServerProcess(runtime=rc)

    import lib.infra.mavsdk_server as ms_mod
    ms_mod.find_mavsdk_server_executable = lambda: "/bin/true"
    raised = False
    try:
        await srv.start()
    except MavsdkServerDead:
        raised = True
    assert raised, "MavsdkServerDead 未抛出"
    print("[OK] test_mavsdk_server_immediate_exit_raises")


def test_jsbsim_bridge_find_csv_missing_returns_none(tmpdir):
    """空目录下查找 wind_truth → None。"""
    result = find_latest_wind_truth_csv(Path(tmpdir))
    assert result is None
    print("[OK] test_jsbsim_bridge_find_csv_missing_returns_none")


def test_jsbsim_bridge_find_csv_returns_latest(tmpdir):
    """多个 CSV 时返回 mtime 最新的那个。"""
    import time
    p1 = Path(tmpdir) / "wind_truth_001.csv"
    p2 = Path(tmpdir) / "wind_truth_002.csv"
    p1.write_text("a")
    time.sleep(0.05)
    p2.write_text("b")
    result = find_latest_wind_truth_csv(Path(tmpdir))
    assert Path(result).name == "wind_truth_002.csv"
    print("[OK] test_jsbsim_bridge_find_csv_returns_latest")


def main():
    import tempfile
    asyncio.run(test_px4_start_failed_raises_when_make_exits())
    asyncio.run(test_px4_raise_if_dead_after_crash())
    asyncio.run(test_mavsdk_server_no_executable_does_not_raise())
    asyncio.run(test_mavsdk_server_immediate_exit_raises())
    with tempfile.TemporaryDirectory() as td:
        test_jsbsim_bridge_find_csv_missing_returns_none(td)
    with tempfile.TemporaryDirectory() as td:
        test_jsbsim_bridge_find_csv_returns_latest(td)
    print("\n[PASS] Phase 2 infra unit tests (6/6)")


if __name__ == "__main__":
    main()
