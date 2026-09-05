"""结构化日志：控制台 + JSONL 双写。

提供:
    - :class:`StructuredLogger`：以 logger 名称分桶，发出 ``info``/``warning``/``error``
      到 stdout，同时把每条事件以 JSON Lines 格式追加到 ``logs/runtime.jsonl``。
    - :meth:`StructuredLogger.failure` 专门写 ``FailureRecord`` 到
      ``logs/segment_failures.jsonl``，便于后续统计。

设计原则：
    1. 不引入第三方日志框架（``logging`` 标准库足够）。
    2. 控制台输出保持人读格式；机器分析用 JSONL。
    3. 写文件失败时回退到 stderr，绝不让日志故障拖垮主流程。

使用样例：
    >>> configure_logging(runtime_cfg)
    >>> log = get_logger(__name__)
    >>> log.info("PX4 启动", pid=12345, log_path="/tmp/px4.log")
    >>> log.failure(FailureRecord(category=FailureCategory.PROCESS, ...))
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .failure import FailureRecord


_DEFAULT_RUNTIME_JSONL = "logs/runtime.jsonl"
_DEFAULT_FAILURE_JSONL = "logs/segment_failures.jsonl"


class _JsonlSink:
    """线程安全的追加写 JSON Lines。失败时降级到 stderr。"""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            sys.stderr.write(f"[obs.logger] 无法创建日志目录 {self._path.parent}: {e}\n")

    def write(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as e:
                sys.stderr.write(
                    f"[obs.logger] JSONL 写入失败 ({self._path}): {e}\n  原始记录: {line}\n"
                )


class StructuredLogger:
    """命名 logger，发出控制台行同时写 JSONL。

    Parameters
    ----------
    name
        通常传 ``__name__``。出现在控制台输出和 JSONL 的 ``logger`` 字段中。
    runtime_sink
        可选 JSONL sink，用于写"事件流"（info/warning/error）。
    failure_sink
        可选 JSONL sink，用于写 :class:`FailureRecord`。
    """

    def __init__(
        self,
        name: str,
        runtime_sink: Optional[_JsonlSink] = None,
        failure_sink: Optional[_JsonlSink] = None,
        console_level: int = logging.INFO,
    ) -> None:
        self.name = name
        self._runtime_sink = runtime_sink
        self._failure_sink = failure_sink
        self._console_level = console_level

    def _emit(self, level: int, level_name: str, message: str, **fields: Any) -> None:
        if level >= self._console_level:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            extras = " ".join(f"{k}={v}" for k, v in fields.items()) if fields else ""
            line = f"[{ts}] [{level_name}] [{self.name}] {message}"
            if extras:
                line += f"  {extras}"
            print(line, file=sys.stderr if level >= logging.WARNING else sys.stdout)
        if self._runtime_sink is not None:
            self._runtime_sink.write({
                "ts": time.time(),
                "level": level_name,
                "logger": self.name,
                "message": message,
                "fields": fields,
            })

    def debug(self, message: str, **fields: Any) -> None:
        self._emit(logging.DEBUG, "DEBUG", message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._emit(logging.INFO, "INFO", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._emit(logging.WARNING, "WARNING", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._emit(logging.ERROR, "ERROR", message, **fields)

    def failure(self, record: FailureRecord, message: Optional[str] = None) -> None:
        """写一条 FailureRecord：控制台显示概要，JSONL 写完整结构。"""
        msg = message or f"[{record.category.value}/{record.code}] {record.message}"
        self.warning(msg, **{k: v for k, v in record.context.items() if not isinstance(v, (dict, list))})
        if self._failure_sink is not None:
            self._failure_sink.write({
                "logger": self.name,
                **record.to_dict(),
            })


_global_runtime_sink: Optional[_JsonlSink] = None
_global_failure_sink: Optional[_JsonlSink] = None
_global_console_level: int = logging.INFO
_global_lock = threading.Lock()


def configure_logging(
    runtime_jsonl: str = _DEFAULT_RUNTIME_JSONL,
    failure_jsonl: str = _DEFAULT_FAILURE_JSONL,
    console_level: str = "INFO",
    base_dir: Optional[str | os.PathLike[str]] = None,
) -> None:
    """初始化全局 sink。在程序入口处调用一次。

    ``base_dir`` 默认是 ``cwd``；若指定则相对路径以它为根。
    """
    global _global_runtime_sink, _global_failure_sink, _global_console_level
    base = Path(base_dir) if base_dir else Path.cwd()
    runtime_path = base / runtime_jsonl if not Path(runtime_jsonl).is_absolute() else Path(runtime_jsonl)
    failure_path = base / failure_jsonl if not Path(failure_jsonl).is_absolute() else Path(failure_jsonl)
    with _global_lock:
        _global_runtime_sink = _JsonlSink(str(runtime_path))
        _global_failure_sink = _JsonlSink(str(failure_path))
        _global_console_level = getattr(logging, console_level.upper(), logging.INFO)


def get_logger(name: str) -> StructuredLogger:
    """获取一个命名 logger。若未 ``configure_logging``，仅控制台输出。"""
    return StructuredLogger(
        name=name,
        runtime_sink=_global_runtime_sink,
        failure_sink=_global_failure_sink,
        console_level=_global_console_level,
    )
