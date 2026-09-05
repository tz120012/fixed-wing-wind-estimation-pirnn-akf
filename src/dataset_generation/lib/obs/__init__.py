"""可观测性：结构化日志 + 失败记录。"""

from .failure import FailureRecord, FailureCategory, classify_exception
from .logger import StructuredLogger, get_logger, configure_logging

__all__ = [
    "FailureRecord",
    "FailureCategory",
    "classify_exception",
    "StructuredLogger",
    "get_logger",
    "configure_logging",
]
