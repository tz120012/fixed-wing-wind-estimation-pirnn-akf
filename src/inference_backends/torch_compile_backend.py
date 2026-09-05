"""
torch.compile 编译后端（PyTorch 2.x Dynamo）
通过 torch.compile 对整个推理函数进行 JIT 编译，
利用 Triton/inductor 自动生成融合 CUDA kernel，消除动态图开销。

特点：
  - 首次调用有编译预热开销（~10-30s），之后每步都是编译好的图
  - 与 TorchScript 不同，支持 Python 动态控制流（含 EMA 分支）
  - 使用 mode='reduce-overhead' 专门针对 batch=1 小模型优化
"""

import importlib.util
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from .base_backend import BaseBackend


class TorchCompileBackend(BaseBackend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        deploy_cfg = self.config.get('deployment', {})
        use_gpu = bool(deploy_cfg.get('use_gpu', True))
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = None
        self._compiled_infer = None
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
        explicit = deploy_cfg.get('model_path_torch')
        if explicit:
            return self._resolve_path(explicit)
        model_save_path = self._resolve_path(self.config['training']['model_save_path'])
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

        # 用 reduce-overhead 模式：专门针对 batch=1 的小模型/固定形状输入
        # 会启用 CUDA graph capture，极大减少 kernel launch 开销
        print(f'[TorchCompileBackend] 正在编译模型（首次编译约需 10-30s）...')
        self.model = torch.compile(model, mode='reduce-overhead', fullgraph=False)

        # 触发一次编译（warmup 时也会触发，但提前触发可以分离编译时间）
        seq_len = int(self.config['data']['sequence_length'])
        input_size = int(self.config['model']['input_size'])
        dummy = torch.randn(1, seq_len, input_size, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy, return_dict=True)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        print(f'[TorchCompileBackend] 编译完成，设备: {self.device}')

        return {
            'model_path': model_path,
            'backend': 'torch_compile',
            'device': str(self.device),
            'compile_mode': 'reduce-overhead',
        }

    def infer(
        self,
        x_seq_normalized: np.ndarray,
        prev_log_q: Optional[Any] = None,
        prev_log_r: Optional[Any] = None,
        ema_alpha: float = 0.1,
        clamp: bool = True,
    ) -> Dict[str, Any]:
        if self.model is None:
            raise RuntimeError('Backend not loaded. Call load() first.')

        x_np = np.asarray(x_seq_normalized, dtype=np.float32)
        if x_np.ndim == 2:
            x_tensor = torch.from_numpy(x_np).unsqueeze(0).to(self.device)
        else:
            x_tensor = torch.from_numpy(x_np).to(self.device)

        with torch.no_grad():
            out = self.model(x_tensor, return_dict=True)

        q_scale = out['q_scale']
        r_scale = out['r_scale']
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

        confidence = out.get('confidence')
        return {
            'wind_estimate': out['wind_estimate'].cpu().numpy()[0],
            'q_scale': q_scale_smooth.cpu().numpy()[0],
            'r_scale': r_scale_smooth.cpu().numpy()[0],
            'angles': out['angles'].cpu().numpy()[0],
            'confidence': confidence.cpu().numpy()[0] if confidence is not None else None,
            'log_q_scale': log_aq.detach().cpu(),
            'log_r_scale': log_b.detach().cpu(),
        }

    def close(self) -> None:
        self.model = None
