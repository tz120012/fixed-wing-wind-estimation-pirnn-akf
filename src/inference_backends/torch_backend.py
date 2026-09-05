import importlib.util
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from .base_backend import BaseBackend


class TorchBackend(BaseBackend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        deploy_cfg = self.config.get('deployment', {})
        use_gpu = bool(deploy_cfg.get('use_gpu', True))
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        self.model = None
        self.src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = os.path.dirname(self.src_dir)

    @property
    def device_name(self) -> str:
        return str(self.device)

    def _load_model_class(self):
        model_file = os.path.join(self.src_dir, '2_pigru_module.py')
        spec = importlib.util.spec_from_file_location('model_definition', model_file)
        if not spec or not spec.loader:
            raise ImportError('Cannot load 2_pigru_module.py')
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
        return os.path.join(model_save_path, 'best_model.pth')

    def load(self) -> Dict[str, Any]:
        model_path = self._resolve_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'模型文件不存在: {model_path}')

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        PIGRU = self._load_model_class()

        # ── yaw-invariant 模型必须在构造时注册 track 系归一化 buffer，否则前向会跳过
        #    track 系变换（模型权重是在 track 系下训练的），导致输出完全错误。
        #    checkpoint 的 state_dict 已自带 _X_mean/_X_scale/_y_mean/_y_scale 及
        #    _track_* 常数，据此重建 norm_params，无需额外的 norm_params.pkl。──
        state_dict = checkpoint['model_state_dict']
        model_cfg = self.config.get('model', {})
        yaw_invariant = ('_track_along_mean' in state_dict) or bool(
            model_cfg.get('yaw_invariant', False))
        norm_params = None
        if yaw_invariant:
            required = ['_X_mean', '_X_scale', '_y_mean', '_y_scale']
            if all(k in state_dict for k in required):
                norm_params = {
                    'X_mean': state_dict['_X_mean'].detach().cpu().numpy(),
                    'X_scale': state_dict['_X_scale'].detach().cpu().numpy(),
                    'y_mean': state_dict['_y_mean'].detach().cpu().numpy(),
                    'y_scale': state_dict['_y_scale'].detach().cpu().numpy(),
                }
            else:
                raise RuntimeError(
                    'checkpoint 标记为 yaw_invariant 但缺少 _X_mean/_X_scale/_y_mean/'
                    '_y_scale buffer，无法重建 track 系归一化常数。')

        self.model = PIGRU(
            input_size=model_cfg['input_size'],
            hidden_size=model_cfg['hidden_size'],
            num_layers=model_cfg['num_layers'],
            dropout=0.0,
            enable_wind_head=True,
            enable_noise_heads=bool(model_cfg.get('enable_noise_heads', True)),
            enable_angles_head=bool(model_cfg.get('enable_angles_head', True)),
            enable_confidence_head=bool(model_cfg.get('enable_confidence_head', True)),
            angle_limit_deg=self.config.get('physics', {}).get('angle_limit_deg', 5.0),
            s_tas_range=tuple(self.config.get('physics', {}).get('s_tas_range', [0.9, 1.1])),
            qr_scale_range=tuple(
                self.config.get('physics', {}).get(
                    'qr_scale_range',
                    self.config.get('physics', {}).get('alpha_beta_range', [0.3, 4.0])
                )
            ),
            yaw_invariant=yaw_invariant,
            norm_params=norm_params,
        ).to(self.device)

        # buffer 已按 checkpoint 结构注册，可严格加载；个别旧 checkpoint 若仍有细微
        # 键差异则回退非严格加载并告警。
        try:
            self.model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            missing = self.model.load_state_dict(state_dict, strict=False)
            print(f'[TorchBackend] strict load 失败，回退 strict=False: {e}\n'
                  f'  missing={missing.missing_keys} unexpected={missing.unexpected_keys}')
        self.model.eval()

        model_info = self.model.get_model_info()
        return {
            'model_path': model_path,
            'model_info': model_info,
            'checkpoint_epoch': checkpoint.get('epoch'),
            'best_val_loss': checkpoint.get('best_val_loss'),
            'best_val_rmse': checkpoint.get('best_val_rmse'),
            'best_val_wind_mag_error': checkpoint.get('best_val_wind_mag_error'),
            'selection_metric': checkpoint.get('selection_metric'),
        }

    def _to_device_tensor_or_none(self, value: Optional[Any]):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.to(self.device)
        return torch.tensor(value, dtype=torch.float32, device=self.device)

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
        elif x_np.ndim == 3:
            x_tensor = torch.from_numpy(x_np).to(self.device)
        else:
            raise ValueError(f'输入维度不正确，期望2D或3D，实际: {x_np.ndim}D')

        prev_log_q_t = self._to_device_tensor_or_none(prev_log_q)
        prev_log_r_t = self._to_device_tensor_or_none(prev_log_r)

        with torch.no_grad():
            out = self.model.predict_online(
                x_tensor,
                prev_log_q=prev_log_q_t,
                prev_log_r=prev_log_r_t,
                ema_alpha=ema_alpha,
                clamp=clamp,
            )

        return {
            'wind_estimate': out['wind_estimate'].detach().cpu().numpy()[0],
            'q_scale': out['q_scale'].detach().cpu().numpy()[0],
            'r_scale': out['r_scale'].detach().cpu().numpy()[0],
            'angles': out['angles'].detach().cpu().numpy()[0],
            'confidence': out.get('confidence', None).detach().cpu().numpy()[0] if out.get('confidence', None) is not None else None,
            'log_q_scale': out.get('log_q_scale', None).detach().cpu() if out.get('log_q_scale', None) is not None else None,
            'log_r_scale': out.get('log_r_scale', None).detach().cpu() if out.get('log_r_scale', None) is not None else None,
        }
