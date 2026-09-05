from typing import Any, Dict, Optional

import numpy as np

from .base_backend import BaseBackend


class AscendBackend(BaseBackend):
    @property
    def device_name(self) -> str:
        return 'ascend'

    def load(self) -> Dict[str, Any]:
        raise NotImplementedError('AscendBackend 尚未在本仓库实现，请先使用 backend=torch。')

    def infer(
        self,
        x_seq_normalized: np.ndarray,
        prev_log_q: Optional[Any] = None,
        prev_log_r: Optional[Any] = None,
        ema_alpha: float = 0.1,
        clamp: bool = True,
    ) -> Dict[str, Any]:
        raise NotImplementedError('AscendBackend 尚未在本仓库实现，请先使用 backend=torch。')
