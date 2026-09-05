from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseBackend(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @property
    @abstractmethod
    def device_name(self) -> str:
        pass

    @abstractmethod
    def load(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def infer(
        self,
        x_seq_normalized: np.ndarray,
        prev_log_q: Optional[Any] = None,
        prev_log_r: Optional[Any] = None,
        ema_alpha: float = 0.1,
        clamp: bool = True,
    ) -> Dict[str, Any]:
        pass

    def close(self) -> None:
        return None
