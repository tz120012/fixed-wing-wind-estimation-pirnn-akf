"""
TorchScript 离线图推理后端
通过 torch.jit.trace 将模型 forward() 编译为静态计算图，
消除 Python 解释器开销，适用于 CUDA/CPU 推理对比。

注意：
  - trace 的是 forward(x, return_dict=False)，返回 (wind, q_scale, r_scale)
  - EMA 平滑在 Python 层补回（与 TorchBackend 一致）
"""

import importlib.util
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from .base_backend import BaseBackend


class TorchScriptBackend(BaseBackend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        deploy_cfg = self.config.get('deployment', {})
        use_gpu = bool(deploy_cfg.get('use_gpu', True))
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.scripted_model = None
        self.alpha_lo: float = 0.3
        self.alpha_hi: float = 4.0
        self.src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = os.path.dirname(self.src_dir)

    @property
    def device_name(self) -> str:
        return str(self.device)

    def _load_model_class(self):
        model_file = os.path.join(self.src_dir, '2_pigru_module.py')
        spec = importlib.util.spec_from_file_location('model_definition', model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.PIGRU

    def _resolve_path(self, path_value: str) -> str:
        if os.path.isabs(path_value):
            return path_value
        return os.path.join(self.project_root, path_value.lstrip('../'))

    def _resolve_model_path(self) -> str:
        deploy_cfg = self.config.get('deployment', {})
        explicit = deploy_cfg.get('model_path_torch') or deploy_cfg.get('model_path_torchscript')
        if explicit:
            return self._resolve_path(explicit)
        model_save_path = self._resolve_path(self.config['training']['model_save_path'])
        # 找最新 train_ 子目录
        train_dirs = sorted(
            [d for d in os.listdir(model_save_path)
             if d.startswith('train_') and os.path.isdir(os.path.join(model_save_path, d))],
            reverse=True
        )
        if train_dirs:
            candidate = os.path.join(model_save_path, train_dirs[0], 'best_model.pth')
            if os.path.exists(candidate):
                return candidate
        return os.path.join(model_save_path, 'best_model.pth')

    def load(self) -> Dict[str, Any]:
        model_path = self._resolve_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path}')

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        PIGRU = self._load_model_class()
        physics_cfg = self.config.get('physics', {})
        qr_range = tuple(physics_cfg.get('qr_scale_range', physics_cfg.get('alpha_beta_range', [0.3, 4.0])))
        self.alpha_lo, self.alpha_hi = float(qr_range[0]), float(qr_range[1])

        model = PIGRU(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=0.0,
            enable_wind_head=True,
            angle_limit_deg=physics_cfg.get('angle_limit_deg', 5.0),
            s_tas_range=tuple(physics_cfg.get('s_tas_range', [0.9, 1.1])),
            qr_scale_range=qr_range,
        ).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # trace forward(x) -> 返回字典模式
        seq_len = int(self.config['data']['sequence_length'])
        input_size = int(self.config['model']['input_size'])
        dummy = torch.randn(1, seq_len, input_size, device=self.device)

        # forward 返回 dict，torch.jit.trace 不支持 dict 输出，
        # 改用 return_dict=False 的元组模式
        class _ForwardWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                out = self.m(x, return_dict=True)
                return (
                    out['wind_estimate'],
                    out['q_scale'],
                    out['r_scale'],
                    out['angles'],
                    out['confidence'],
                )

        wrapper = _ForwardWrapper(model).to(self.device)
        wrapper.eval()
        with torch.no_grad():
            self.scripted_model = torch.jit.trace(wrapper, dummy)
            self.scripted_model = torch.jit.optimize_for_inference(self.scripted_model)

        print(f'[TorchScriptBackend] 模型已 trace 并优化，设备: {self.device}')
        return {
            'model_path': model_path,
            'backend': 'torchscript',
            'device': str(self.device),
        }

    def infer(
        self,
        x_seq_normalized: np.ndarray,
        prev_log_q: Optional[Any] = None,
        prev_log_r: Optional[Any] = None,
        ema_alpha: float = 0.1,
        clamp: bool = True,
    ) -> Dict[str, Any]:
        if self.scripted_model is None:
            raise RuntimeError('Backend not loaded. Call load() first.')

        x_np = np.asarray(x_seq_normalized, dtype=np.float32)
        if x_np.ndim == 2:
            x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(self.device)
        else:
            x_tensor = torch.from_numpy(x_np).to(self.device)

        with torch.no_grad():
            wind, q_scale, r_scale, angles, confidence = self.scripted_model(x_tensor)

        # EMA 平滑（与 TorchBackend 保持一致）
        log_aq = torch.log(q_scale + 1e-8)
        log_b = torch.log(r_scale + 1e-8)
        if prev_log_q is not None:
            prev_t = prev_log_q if torch.is_tensor(prev_log_q) else torch.tensor(prev_log_q)
            log_aq = (1 - ema_alpha) * prev_t.to(self.device) + ema_alpha * log_aq
        if prev_log_r is not None:
            prev_t = prev_log_r if torch.is_tensor(prev_log_r) else torch.tensor(prev_log_r)
            log_b = (1 - ema_alpha) * prev_t.to(self.device) + ema_alpha * log_b

        q_scale_smooth = torch.exp(log_aq)
        r_scale_smooth = torch.exp(log_b)
        if clamp:
            q_scale_smooth = torch.clamp(q_scale_smooth, self.alpha_lo, self.alpha_hi)
            r_scale_smooth = torch.clamp(r_scale_smooth, self.alpha_lo, self.alpha_hi)

        return {
            'wind_estimate': wind.cpu().numpy()[0],
            'q_scale': q_scale_smooth.cpu().numpy()[0],
            'r_scale': r_scale_smooth.cpu().numpy()[0],
            'angles': angles.cpu().numpy()[0],
            'confidence': confidence.cpu().numpy()[0],
            'log_q_scale': log_aq.detach().cpu(),
            'log_r_scale': log_b.detach().cpu(),
        }

    def close(self) -> None:
        self.scripted_model = None
