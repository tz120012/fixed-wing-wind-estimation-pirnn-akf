"""``mavsdk_server`` 子进程的生命周期封装。

替换原 ``generate_dataset.start_mavsdk_server`` / ``check_mavsdk_server_alive`` 与
``DatasetGenerator.check_and_restart_mavsdk_server`` 中三处散落的清理逻辑。

主要改动：
    1. 显式持有 ``self.proc``、``self.log_file``，``__aexit__`` 关闭句柄；
    2. 启动失败抛 ``MavsdkServerDead`` 而非返回 ``None``（让上层显式决定回退到 SDK 自启）；
    3. 静默 ``except: pass`` 改为 ``logger.failure``。
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import IO, Optional

from ..config.runtime import RuntimeConfig
from ..errors import MavsdkServerDead
from ..obs import FailureCategory, FailureRecord, get_logger


_LOGGER = get_logger("infra.mavsdk_server")


def _build_proxy_clean_env() -> dict:
    """构造一个移除所有 HTTP/HTTPS/gRPC 代理变量的子进程环境。

    mavsdk_server 内部用 gRPC C-core，若继承到 ``HTTP(S)_PROXY`` 等变量，
    可能在某些 gRPC 版本下把本地 ``udpin://`` / 50051 走代理，引发
    ``Socket closed`` / ``Connection refused``。这里显式置空 +
    显式 NO_PROXY，与父进程 generate_dataset.py 头部的清理保持一致。
    """
    env = os.environ.copy()
    for var in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
        "grpc_proxy",
    ):
        env[var] = ""
    env["no_proxy"] = "127.0.0.1,localhost,::1,0.0.0.0"
    env["NO_PROXY"] = env["no_proxy"]
    env["no_grpc_proxy"] = "*"
    env["GRPC_VERBOSITY"] = env.get("GRPC_VERBOSITY", "ERROR")
    env["GRPC_TRACE"] = env.get("GRPC_TRACE", "")
    return env


def find_mavsdk_server_executable() -> Optional[str]:
    """返回 ``mavsdk`` 包内 ``mavsdk_server`` 可执行文件路径。

    与原 ``get_mavsdk_server_path`` 同语义；找不到时返回 ``None``。
    """
    try:
        import mavsdk
    except ImportError:
        return None
    bin_dir = Path(mavsdk.__file__).resolve().parent / "bin"
    exe = bin_dir / "mavsdk_server"
    if exe.is_file() and os.access(exe, os.X_OK):
        return str(exe)
    return None


class MavsdkServerProcess:
    """``mavsdk_server`` 子进程 async context manager。

    Usage::

        async with MavsdkServerProcess(runtime=rc, log_dir=Path("logs")) as srv:
            if srv.is_running():
                fc = FlightController(mavsdk_server_address="127.0.0.1")
            ...
    """

    def __init__(
        self,
        runtime: RuntimeConfig,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.runtime = runtime
        self.log_dir = Path(log_dir) if log_dir is not None else Path("logs")
        self.proc: Optional[subprocess.Popen] = None
        self.log_file: Optional[IO] = None
        self.executable_path: Optional[str] = None

    async def __aenter__(self) -> "MavsdkServerProcess":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop()

    async def start(self) -> None:
        """启动 ``mavsdk_server``。若可执行文件不存在，则保持 ``proc=None``，
        由上层决定回退路径（不抛异常）。
        """
        self.executable_path = find_mavsdk_server_executable()
        if not self.executable_path:
            _LOGGER.warning("未找到 mavsdk_server 可执行文件，调用方应回退到 SDK 自启")
            return

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.log_dir / "mavsdk_server.log"
            self.log_file = open(log_path, "a")
            self.proc = subprocess.Popen(
                [
                    self.executable_path,
                    f"udpin://0.0.0.0:{self.runtime.ports.mavsdk_udp}",
                    "-p",
                    str(self.runtime.ports.mavsdk_grpc),
                ],
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                env=_build_proxy_clean_env(),
            )
        except OSError as e:
            self._close_log_file()
            raise MavsdkServerDead(
                f"启动 mavsdk_server 失败: {e}",
                context={"executable": self.executable_path},
            ) from e

        # 等待端口就绪
        await asyncio.sleep(self.runtime.sleeps.pkill_grace_s + 1)

        if self.proc.poll() is None:
            _LOGGER.info(
                "mavsdk_server 已启动",
                pid=self.proc.pid,
                udp=self.runtime.ports.mavsdk_udp,
                grpc=self.runtime.ports.mavsdk_grpc,
                log=str(self.log_dir / "mavsdk_server.log"),
            )
        else:
            exit_code = self.proc.returncode
            self._close_log_file()
            self.proc = None
            raise MavsdkServerDead(
                "mavsdk_server 启动后立即退出",
                context={"exit_code": exit_code},
            )

    def is_running(self) -> bool:
        """进程是否仍在运行。"""
        if self.proc is None:
            return False
        return self.proc.poll() is None

    async def stop(self) -> None:
        """优雅停止：``terminate`` → ``wait`` → 失败则 ``kill``。"""
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=self.runtime.timeouts.px4_kill_wait_s)
            except (subprocess.TimeoutExpired, OSError) as e:
                _LOGGER.warning("mavsdk_server SIGTERM 失败，回退 SIGKILL", reason=str(e))
                with suppress(Exception):
                    self.proc.kill()
                with suppress(Exception):
                    self.proc.wait(timeout=self.runtime.timeouts.px4_kill_wait_s)
        self.proc = None

        # pkill 兜底（清理上轮残留的 mavsdk_server）
        try:
            subprocess.run(
                ["pkill", "-f", "mavsdk_server"],
                stderr=subprocess.DEVNULL,
                timeout=self.runtime.timeouts.pkill_timeout_s,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            _LOGGER.failure(FailureRecord(
                category=FailureCategory.CLEANUP,
                code="mavsdk_pkill_failed",
                message=f"pkill mavsdk_server: {e}",
                context={},
            ))

        self._close_log_file()

    def _close_log_file(self) -> None:
        if self.log_file is None:
            return
        try:
            self.log_file.close()
        except OSError as e:
            _LOGGER.failure(FailureRecord(
                category=FailureCategory.CLEANUP,
                code="mavsdk_log_close_failed",
                message=f"关闭 mavsdk_server 日志失败: {e}",
                context={},
            ))
        self.log_file = None

    async def restart(self) -> None:
        """先停后启，用于 ``check_and_restart_mavsdk_server`` 的语义。"""
        await self.stop()
        await asyncio.sleep(self.runtime.sleeps.pkill_grace_s)
        await self.start()
