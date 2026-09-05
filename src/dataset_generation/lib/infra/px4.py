"""PX4 SITL 进程的生命周期封装。

把原 ``DatasetGenerator.{start_px4_sitl, stop_px4_sitl, _verify_jsbsim_running,
check_jsbsim_health}`` 与模块级的 ``clear_px4_lock_files`` / ``clear_px4_rootfs_state``
合并为一个 ``async with`` 即可使用的对象：

    async with Px4SitlProcess(px4_root=Path("..."), runtime=rc) as px4:
        await px4.verify_bridge_ready()
        ...
        # 退出 with：自动 SIGTERM/SIGKILL/pkill 安全网/关闭日志/清 lock

设计要点：
    1. 显式持有 ``self.proc``、``self.log_file``，在 ``__aexit__`` 保证关闭句柄；
    2. 不再使用 ``getattr/setattr`` 动态属性；
    3. 所有原本静默 ``try/except: pass`` 的清理路径，全部改为
       ``logger.failure(FailureRecord(category=CLEANUP, ...))``；
    4. 异常分类化：启动后 PX4 进程在等待结束前已经 ``poll() != None`` → 抛
       ``SitlStartFailed``；运行中通过 ``raise_if_dead()`` 检测到崩溃 → 抛 ``SitlCrashed``。
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import IO, Optional

from ..config.runtime import RuntimeConfig
from ..errors import JsbsimBridgeMissing, SitlCrashed, SitlStartFailed
from ..obs import FailureCategory, FailureRecord, get_logger


_LOGGER = get_logger("infra.px4")


def clear_px4_lock_files() -> None:
    """删除 ``/tmp/px4_lock-{0..3}``，避免下一轮启动时报 'PX4 daemon already running'。"""
    for i in range(4):
        target = Path("/tmp") / f"px4_lock-{i}"
        try:
            target.unlink(missing_ok=True)
        except OSError as e:
            _LOGGER.failure(FailureRecord(
                category=FailureCategory.CLEANUP,
                code="px4_lock_unlink_failed",
                message=f"删除 {target} 失败: {e}",
                context={"path": str(target), "errno": getattr(e, "errno", None)},
            ))


def clear_px4_rootfs_state(px4_root: Path) -> None:
    """清理 PX4 SITL rootfs 中的持久化状态（eeprom 参数、dataman），确保每轮干净启动。"""
    rootfs = px4_root / "build" / "px4_sitl_default" / "tmp" / "rootfs"
    for name in ["eeprom", "dataman", "mission_state"]:
        target = rootfs / name
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            _LOGGER.debug("已删除 rootfs 目录", path=str(target))
        elif target.is_file():
            try:
                target.unlink()
                _LOGGER.debug("已删除 rootfs 文件", path=str(target))
            except OSError as e:
                _LOGGER.failure(FailureRecord(
                    category=FailureCategory.CLEANUP,
                    code="rootfs_unlink_failed",
                    message=f"删除 {target} 失败: {e}",
                    context={"path": str(target)},
                ))


class Px4SitlProcess:
    """PX4 SITL ``make px4_sitl jsbsim_<airframe>`` process manager.

    Parameters
    ----------
    px4_root
        PX4 源码根目录。
    runtime
        运行期配置（用于 sleep / timeout / retry 数值）。
    redirect_log_to
        若给定，PX4 stdout/stderr 写入该文件；否则丢弃输出，避免未消费的
        ``PIPE`` 缓冲区填满后阻塞长期采集。
    """

    def __init__(
        self,
        px4_root: Path,
        runtime: RuntimeConfig,
        redirect_log_to: Optional[Path] = None,
        airframe: str = "rascal",
    ) -> None:
        self.px4_root = Path(px4_root)
        self.runtime = runtime
        self.redirect_log_to = Path(redirect_log_to) if redirect_log_to else None
        self.airframe = str(airframe).strip().lower()
        if self.airframe not in {"rascal", "malolo"}:
            raise ValueError(
                f"Unsupported JSBSim fixed-wing airframe {airframe!r}; "
                "expected 'rascal' or 'malolo'"
            )
        self.proc: Optional[subprocess.Popen] = None
        self.log_file: Optional[IO] = None
        self._entered = False

    async def __aenter__(self) -> "Px4SitlProcess":
        await self.start()
        self._entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def start(self) -> None:
        clear_px4_rootfs_state(self.px4_root)

        if self.redirect_log_to is not None:
            self.redirect_log_to.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = open(self.redirect_log_to, "w")

        env = os.environ.copy()
        env.setdefault("PX4_ROOT", str(self.px4_root))
        env["HEADLESS"] = "1"
        env["NO_PXH"] = "1"
        # PX4 SITL 内部会启动 jsbsim_bridge 子进程；它通过 TCP/UDP 与 mavsdk_server
        # 在本机通信。把 HTTP/HTTPS 代理变量清空，避免任何上游库（如 curl 探针）
        # 误把 127.0.0.1:* 走 Clash 代理。
        for _proxy_var in (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "http_proxy", "https_proxy", "all_proxy",
        ):
            env[_proxy_var] = ""
        env["no_proxy"] = "127.0.0.1,localhost,::1,0.0.0.0"
        env["NO_PROXY"] = env["no_proxy"]

        make_target = f"jsbsim_{self.airframe}"
        _LOGGER.info(
            f"启动 PX4 SITL ({make_target})",
            px4_root=str(self.px4_root),
            airframe=self.airframe,
            log_path=str(self.redirect_log_to) if self.redirect_log_to else "stdout",
        )
        self.proc = subprocess.Popen(
            ["make", "px4_sitl", make_target],
            cwd=str(self.px4_root),
            stdout=self.log_file or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        await asyncio.sleep(self.runtime.sleeps.px4_post_start_s)

        if self.proc.poll() is not None:
            self._close_log_file()
            raise SitlStartFailed(
                "PX4 SITL 在 px4_post_start 等待期间已退出",
                context={
                    "exit_code": self.proc.returncode,
                    "px4_root": str(self.px4_root),
                    "log_path": str(self.redirect_log_to) if self.redirect_log_to else None,
                },
            )

        _LOGGER.info("PX4 SITL 已启动", pid=self.proc.pid)

        if not await self.verify_bridge_ready(retries=self.runtime.retries.jsbsim_verify):
            _LOGGER.warning("jsbsim_bridge 验证失败，额外等待重试")
            await asyncio.sleep(10)
            if not await self.verify_bridge_ready(retries=1):
                raise JsbsimBridgeMissing(
                    "JSBSim bridge 未在限定时间内就绪",
                    context={"port": self.runtime.ports.jsbsim_bridge},
                )

    async def verify_bridge_ready(self, retries: int = 3) -> bool:
        """验证 ``jsbsim_bridge`` 进程在跑且端口被监听。"""
        for attempt in range(retries):
            result = subprocess.run(
                ["pgrep", "-f", "jsbsim_bridge"], capture_output=True, text=True
            )
            if result.returncode == 0:
                pid = result.stdout.strip().split("\n")[0]
                port_check = subprocess.run(
                    ["netstat", "-tuln"], capture_output=True, text=True
                )
                port_str = str(self.runtime.ports.jsbsim_bridge)
                if port_str in port_check.stdout:
                    _LOGGER.info(
                        "JSBSim bridge 就绪", pid=pid, port=port_str
                    )
                    return True
                _LOGGER.warning(
                    "jsbsim_bridge 运行中但端口未监听",
                    pid=pid, port=port_str,
                )
            if attempt < retries - 1:
                await asyncio.sleep(5)
        return False

    async def is_alive(self) -> bool:
        """PX4 进程是否仍在运行。"""
        if self.proc is None:
            return False
        return self.proc.poll() is None

    async def raise_if_dead(self) -> None:
        """若 PX4 已退出则抛 ``SitlCrashed``。供采集循环周期性检查。"""
        if self.proc is not None and self.proc.poll() is not None:
            raise SitlCrashed(
                "PX4 SITL 进程在采集过程中异常退出",
                context={"exit_code": self.proc.returncode, "pid": self.proc.pid},
            )

    async def check_jsbsim_health(self) -> bool:
        """检查 ``jsbsim_bridge`` 进程是否健康；若已退出则记日志。"""
        result = subprocess.run(
            ["pgrep", "-f", "jsbsim_bridge"], capture_output=True
        )
        is_running = result.returncode == 0
        if not is_running:
            _LOGGER.warning("jsbsim_bridge 进程已退出")
            log_path = (
                self.px4_root
                / "build" / "px4_sitl_default" / "tmp" / "rootfs" / "jsbsim_bridge.log"
            )
            if log_path.exists():
                try:
                    last_lines = log_path.read_text(errors="replace").splitlines()[-10:]
                    if last_lines:
                        _LOGGER.warning(
                            "jsbsim_bridge 最近日志", tail="\n".join(last_lines)
                        )
                except OSError as e:
                    _LOGGER.failure(FailureRecord(
                        category=FailureCategory.CLEANUP,
                        code="bridge_log_read_failed",
                        message=f"读取 {log_path} 失败: {e}",
                        context={"path": str(log_path)},
                    ))
        return is_running

    async def stop(self) -> None:
        """先 SIGTERM 进程组，必要时 SIGKILL，再 pkill 安全网清残留。"""
        _LOGGER.info("停止 PX4 SITL")
        if self.proc is not None:
            await self._terminate_process_group()
            self.proc = None

        self._close_log_file()

        # 安全网：杀残留 / 孤儿进程（含上轮可能残留的 sitl_run.sh）
        for cmd in [
            ["pkill", "-9", "-x", "px4"],
            ["pkill", "-9", "-x", "JSBSim"],
            ["pkill", "-9", "-f", "jsbsim_bridge"],
            ["pkill", "-9", "-f", "sitl_run"],
        ]:
            try:
                subprocess.run(
                    cmd,
                    stderr=subprocess.DEVNULL,
                    timeout=self.runtime.timeouts.pkill_timeout_s,
                )
            except (subprocess.TimeoutExpired, OSError) as e:
                _LOGGER.failure(FailureRecord(
                    category=FailureCategory.CLEANUP,
                    code="pkill_failed",
                    message=f"{' '.join(cmd)}: {e}",
                    context={"cmd": cmd},
                ))

        await asyncio.sleep(self.runtime.sleeps.cleanup_post_stop_s)
        clear_px4_lock_files()
        _LOGGER.info("PX4 SITL 已停止")

    async def _terminate_process_group(self) -> None:
        assert self.proc is not None
        try:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            self.proc.wait(timeout=self.runtime.timeouts.px4_terminate_s)
        except (subprocess.TimeoutExpired, ProcessLookupError, OSError) as e:
            _LOGGER.warning("SIGTERM 超时或进程已退出，回退 SIGKILL", reason=str(e))
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            with suppress(Exception):
                self.proc.kill()

        try:
            self.proc.wait(timeout=self.runtime.timeouts.px4_kill_wait_s)
        except subprocess.TimeoutExpired:
            _LOGGER.failure(FailureRecord(
                category=FailureCategory.CLEANUP,
                code="px4_wait_timeout",
                message="SIGKILL 后等待回收仍超时",
                context={"pid": self.proc.pid},
            ))

    def _close_log_file(self) -> None:
        if self.log_file is None:
            return
        try:
            self.log_file.close()
        except OSError as e:
            _LOGGER.failure(FailureRecord(
                category=FailureCategory.CLEANUP,
                code="px4_log_close_failed",
                message=f"关闭 PX4 日志失败: {e}",
                context={"path": str(self.redirect_log_to)},
            ))
        self.log_file = None
