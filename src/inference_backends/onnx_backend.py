"""ONNX Runtime backend for Raspberry Pi deployment."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

from .base_backend import BaseBackend


class OnnxBackend(BaseBackend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session = None
        self.input_name = None
        self.output_names = None
        qr_range = self.config.get("physics", {}).get("qr_scale_range", [0.3, 4.0])
        self.scale_lo = float(qr_range[0])
        self.scale_hi = float(qr_range[1])
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

    @property
    def device_name(self) -> str:
        return "onnxruntime-cpu"

    def _resolve_path(self, value: str) -> str:
        if os.path.isabs(value):
            return value
        return os.path.join(self.project_root, value.lstrip("../"))

    def load(self) -> Dict[str, Any]:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX backend requires onnxruntime (or onnxruntime-gpu)"
            ) from exc
        configured = self.config.get("deployment", {}).get("model_path_onnx")
        if not configured:
            raise ValueError("deployment.model_path_onnx is required")
        model_path = self._resolve_path(str(configured))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(
            self.config.get("deployment", {}).get("onnx_intra_op_threads", 4)
        )
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]
        expected = ["wind_estimate", "q_scale", "r_scale", "angles", "confidence"]
        if self.output_names != expected:
            raise RuntimeError(
                f"Unexpected ONNX outputs {self.output_names}; expected {expected}"
            )
        return {
            "model_path": model_path,
            "backend": "onnxruntime",
            "device": self.device_name,
        }

    @staticmethod
    def _as_numpy(value: Optional[Any]) -> Optional[np.ndarray]:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def infer(
        self,
        x_seq_normalized: np.ndarray,
        prev_log_q: Optional[Any] = None,
        prev_log_r: Optional[Any] = None,
        ema_alpha: float = 0.1,
        clamp: bool = True,
    ) -> Dict[str, Any]:
        if self.session is None or self.input_name is None:
            raise RuntimeError("Backend not loaded. Call load() first.")
        x = np.asarray(x_seq_normalized, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        if x.ndim != 3:
            raise ValueError(f"Expected 2-D or 3-D input, got shape {x.shape}")
        wind, q_scale, r_scale, angles, confidence = self.session.run(
            self.output_names, {self.input_name: x}
        )
        log_q = np.log(np.maximum(q_scale, 1e-8))
        log_r = np.log(np.maximum(r_scale, 1e-8))
        prev_q = self._as_numpy(prev_log_q)
        prev_r = self._as_numpy(prev_log_r)
        if prev_q is not None:
            log_q = (1.0 - ema_alpha) * prev_q + ema_alpha * log_q
        if prev_r is not None:
            log_r = (1.0 - ema_alpha) * prev_r + ema_alpha * log_r
        q_smooth = np.exp(log_q)
        r_smooth = np.exp(log_r)
        if clamp:
            q_smooth = np.clip(q_smooth, self.scale_lo, self.scale_hi)
            r_smooth = np.clip(r_smooth, self.scale_lo, self.scale_hi)
        return {
            "wind_estimate": wind[0],
            "q_scale": q_smooth[0],
            "r_scale": r_smooth[0],
            "angles": angles[0],
            "confidence": confidence[0],
            "log_q_scale": log_q,
            "log_r_scale": log_r,
        }

    def close(self) -> None:
        self.session = None
