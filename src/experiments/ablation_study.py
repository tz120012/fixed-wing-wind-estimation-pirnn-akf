"""
消融实验脚本
论文 Figure 6: OOD泛化测试与λ_phy权重消融

功能：
  1. 测试不同物理损失权重(λ_phy)的训练效果
  2. 评估OOD(Out-of-Distribution)泛化能力
  3. 生成消融实验对比图表
"""

import numpy as np
import pandas as pd
import pickle
import yaml
import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt


try:
    import torch
except ModuleNotFoundError:
    torch = None


# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(np.square(diff))))


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return ''

    headers = [str(col) for col in df.columns]
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('|' + '|'.join(['---'] * len(headers)) + '|')

    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                cells.append(f'{val:.6g}')
            else:
                cells.append(str(val))
        lines.append('| ' + ' | '.join(cells) + ' |')

    return '\n'.join(lines)


class AblationStudy:
    """消融实验框架（多 OOD 场景版）"""

    DEFAULT_OOD_SCENES = [
        'ood_0p5x',
        'ood_0p75x',
        'ood_1p0x',
        'ood_1p25x',
        'ood_1p5x',
        'ood_low_speed',
        'ood_turb_heavy'
    ]
    
    def __init__(self,
                 config_path: str = None,
                 ood_scenes: Optional[List[str]] = None,
                 output_dir: Optional[str] = None,
                 load_norm_params: bool = True):
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'config.yaml')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.config_path = config_path
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.ood_scenes = ood_scenes if ood_scenes else self.DEFAULT_OOD_SCENES.copy()

        if output_dir is None:
            self.output_dir = os.path.join(project_root, 'data', 'ablation', f'ablation_{self.timestamp}')
        else:
            self.output_dir = self._resolve_path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # 设备
        self.device = torch.device('cuda' if torch is not None and torch.cuda.is_available() else 'cpu') if torch is not None else 'cpu'


        # 加载归一化参数（仅评估/训练需要）
        if load_norm_params:
            self.load_normalization_params()

        
        # 消融实验结果
        self.ablation_results = {}
        self.ood_results = {}
        self.lambda_model_registry = {}
        
        print(f"\n【消融实验框架】")
        print(f"  设备: {self.device}")
        print(f"  输出目录: {self.output_dir}")
        print(f"  OOD场景: {', '.join(self.ood_scenes)}")

    def _resolve_path(self, maybe_rel_path: str) -> str:
        if os.path.isabs(maybe_rel_path):
            return maybe_rel_path
        return os.path.join(project_root, maybe_rel_path.lstrip('../'))

    @staticmethod
    def _require_torch():
        if torch is None:
            raise ModuleNotFoundError('当前环境未安装 `torch`。训练/评估模式需要 PyTorch；仅重绘图表可使用 `--redraw-only`.')

    def load_existing_results(self, result_dir: Optional[str] = None):

        """从已有结果目录加载 CSV，用于仅重绘图表。"""
        if result_dir is not None:
            self.output_dir = self._resolve_path(result_dir)
        if not os.path.isdir(self.output_dir):
            raise FileNotFoundError(f"结果目录不存在: {self.output_dir}")

        file_map = {
            'network_multi_ood': 'ablation_network_multi_ood.csv',
            'system_multi_ood': 'ablation_system_multi_ood.csv',
            'covariance_multi_ood': 'ablation_covariance_multi_ood.csv'
        }

        inferred_scenes = []
        loaded_any = False
        for key, filename in file_map.items():
            path = os.path.join(self.output_dir, filename)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            self.ablation_results[key] = df.to_dict('records')
            loaded_any = True

            if key == 'network_multi_ood':
                inferred_scenes.extend([
                    col[:-5] for col in df.columns
                    if col.startswith('ood_') and col.endswith('_rmse')
                ])
            else:
                inferred_scenes.extend([
                    col[4:-5] for col in df.columns
                    if col.startswith('sys_ood_') and col.endswith('_rmse')
                ])
            print(f"  ✓ 已加载结果: {path}")

        if not loaded_any:
            raise FileNotFoundError(f"目录中未找到消融 CSV 结果: {self.output_dir}")

        dedup_scenes = []
        for scene in inferred_scenes:
            if scene not in dedup_scenes:
                dedup_scenes.append(scene)
        if dedup_scenes:
            self.ood_scenes = dedup_scenes
            print(f"  ✓ 从结果中识别 OOD 场景: {', '.join(self.ood_scenes)}")

    def load_train_val_data(self,

                            max_train_samples: Optional[int] = None,
                            max_val_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载训练与验证数据（用于批量训练）"""
        data_dir = self._resolve_path(self.config['data']['processed_dir'])
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

        if max_train_samples is not None and max_train_samples > 0 and len(X_train) > max_train_samples:
            X_train = X_train[:max_train_samples]
            y_train = y_train[:max_train_samples]
        if max_val_samples is not None and max_val_samples > 0 and len(X_val) > max_val_samples:
            X_val = X_val[:max_val_samples]
            y_val = y_val[:max_val_samples]
        return X_train, y_train, X_val, y_val

    def _write_temp_config(self,
                           lambda_phy: float,
                           num_epochs: Optional[int] = None,
                           force_num_workers: Optional[int] = None) -> str:
        """按λ写临时配置文件，供训练器加载"""
        cfg = yaml.safe_load(yaml.safe_dump(self.config))
        cfg['training']['lambda_physics'] = float(lambda_phy)
        if num_epochs is not None:
            cfg['training']['num_epochs'] = int(num_epochs)
        if force_num_workers is not None:
            cfg['training']['num_workers'] = int(force_num_workers)

        tmp_path = os.path.join(self.output_dir, f'tmp_config_lambda_{lambda_phy}.yaml')
        with open(tmp_path, 'w') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        return tmp_path

    def _latest_model_dir(self, prefix: str) -> Optional[str]:
        model_save_path = self._resolve_path(self.config['training']['model_save_path'])
        if not os.path.exists(model_save_path):
            return None
        candidates = [
            d for d in os.listdir(model_save_path)
            if d.startswith(prefix) and os.path.isdir(os.path.join(model_save_path, d))
        ]
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return os.path.join(model_save_path, candidates[0])

    def train_model_for_lambda(self,
                               lambda_val: float,
                               X_train: np.ndarray,
                               y_train: np.ndarray,
                               X_val: np.ndarray,
                               y_val: np.ndarray,
                               num_epochs: Optional[int] = None,
                               force_num_workers: Optional[int] = None,
                               prefer_pigru_for_zero: bool = False) -> Optional[Dict]:
        """训练单个 λ 的模型并返回模型信息"""
        if lambda_val == 0.0 and not prefer_pigru_for_zero:
            model_type = 'vanilla_gru'
            prefix = 'vanilla_gru_'
            module_file = os.path.join(project_root, 'src', '3b_train_vanilla_gru.py')
            class_name = 'VanillaGRUTrainer'
        else:
            model_type = 'pigru'
            prefix = 'train_'
            module_file = os.path.join(project_root, 'src', '3_train_pigru.py')
            class_name = 'Trainer'

        before_dir = self._latest_model_dir(prefix)
        temp_cfg = self._write_temp_config(
            lambda_val,
            num_epochs=num_epochs,
            force_num_workers=force_num_workers
        )

        print(f"\n【训练 λ={lambda_val}】")
        print(f"  模型类型: {model_type}")
        print(f"  配置文件: {temp_cfg}")

        import importlib.util
        spec = importlib.util.spec_from_file_location(f"trainer_lambda_{lambda_val}", module_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        TrainerClass = getattr(module, class_name)

        trainer = TrainerClass(config_path=temp_cfg)
        trainer.train(X_train, y_train, X_val, y_val)

        after_dir = trainer.model_save_dir if hasattr(trainer, 'model_save_dir') else self._latest_model_dir(prefix)
        if after_dir is None or not os.path.exists(os.path.join(after_dir, 'best_model.pth')):
            print(f"  ⚠️ λ={lambda_val} 训练完成但未找到可用 best_model.pth")
            return None

        if before_dir is not None and os.path.abspath(before_dir) == os.path.abspath(after_dir):
            print(f"  ⚠️ λ={lambda_val} 训练后目录未变化，请确认训练是否真正执行")

        model_info = {
            'lambda_phy': float(lambda_val),
            'model_type': model_type,
            'model_dir': after_dir,
            'model_path': os.path.join(after_dir, 'best_model.pth')
        }
        self.lambda_model_registry[float(lambda_val)] = model_info
        return model_info

    def train_models_for_lambdas(self,
                                 lambda_values: List[float],
                                 num_epochs: Optional[int] = None,
                                 skip_existing: bool = True,
                                 max_train_samples: Optional[int] = None,
                                 max_val_samples: Optional[int] = None,
                                 prefer_pigru_for_zero: bool = False) -> Dict[float, Dict]:
        """批量训练多个 λ 的模型"""
        print("\n" + "="*70)
        print("批量训练 λ 消融模型")
        print("="*70)

        X_train, y_train, X_val, y_val = self.load_train_val_data(
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples
        )
        print(f"\n【训练数据规模】")
        print(f"  train: X={X_train.shape}, y={y_train.shape}")
        print(f"  val  : X={X_val.shape}, y={y_val.shape}")

        force_num_workers = 0 if (max_train_samples is not None or max_val_samples is not None) else None

        registry = {}
        for lambda_val in lambda_values:
            lambda_key = float(lambda_val)

            existing = self.find_pigru_model_for_lambda(lambda_key) if (prefer_pigru_for_zero and lambda_key == 0.0) else self.find_model_for_lambda(lambda_key)
            if skip_existing and existing is not None:
                print(f"\n【λ={lambda_key}】发现现有模型，跳过训练")
                registry[lambda_key] = existing
                self.lambda_model_registry[lambda_key] = existing
                continue

            try:
                info = self.train_model_for_lambda(
                    lambda_key,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    num_epochs=num_epochs,
                    force_num_workers=force_num_workers,
                    prefer_pigru_for_zero=prefer_pigru_for_zero
                )
                if info is not None:
                    registry[lambda_key] = info
            except Exception as e:
                print(f"\n⚠️ λ={lambda_key} 训练失败: {e}")

        registry_path = os.path.join(self.output_dir, 'lambda_model_registry.csv')
        if registry:
            pd.DataFrame(list(registry.values())).to_csv(registry_path, index=False)
            print(f"\n  ✓ 模型注册表已保存: {registry_path}")
        return registry

    def find_model_for_lambda(self, lambda_val: float) -> Optional[Dict]:
        """按 λ 查找最佳可用模型（A 实验：λ=0 优先 Vanilla，其余优先 PI-GRU）"""
        lambda_key = float(lambda_val)
        cached = self.lambda_model_registry.get(lambda_key)
        if cached is not None and os.path.exists(cached.get('model_path', '')):
            return cached

        if lambda_key == 0.0:
            model_save_path = self._resolve_path(self.config['training']['model_save_path'])
            if os.path.exists(model_save_path):
                vanilla_dirs = [
                    d for d in os.listdir(model_save_path)
                    if d.startswith('vanilla_gru_') and os.path.isdir(os.path.join(model_save_path, d))
                ]
                vanilla_dirs.sort(reverse=True)
                for d in vanilla_dirs:
                    model_dir = os.path.join(model_save_path, d)
                    model_path = os.path.join(model_dir, 'best_model.pth')
                    if os.path.exists(model_path):
                        info = {
                            'lambda_phy': lambda_key,
                            'model_type': 'vanilla_gru',
                            'model_dir': model_dir,
                            'model_path': model_path
                        }
                        self.lambda_model_registry[lambda_key] = info
                        return info

        pigru_info = self.find_pigru_model_for_lambda(lambda_key)
        if pigru_info is not None:
            return pigru_info

        return None

    def find_pigru_model_for_lambda(self, lambda_val: float, tol: float = 1e-8) -> Optional[Dict]:
        """按 λ 查找 PI-GRU 模型（系统级实验使用，不会退化到 Vanilla GRU）"""
        self._require_torch()
        lambda_key = float(lambda_val)

        cached = self.lambda_model_registry.get(lambda_key)
        if cached is not None and cached.get('model_type') == 'pigru' and os.path.exists(cached.get('model_path', '')):
            return cached

        model_save_path = self._resolve_path(self.config['training']['model_save_path'])
        if not os.path.exists(model_save_path):
            return None

        model_dirs = [
            d for d in os.listdir(model_save_path)
            if d.startswith('train_') and os.path.isdir(os.path.join(model_save_path, d))
        ]
        model_dirs.sort(reverse=True)

        for d in model_dirs:
            model_dir = os.path.join(model_save_path, d)
            model_path = os.path.join(model_dir, 'best_model.pth')
            if not os.path.exists(model_path):
                continue

            try:
                ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
                ck_cfg = ckpt.get('config', {})
                ck_lambda = ck_cfg.get('training', {}).get('lambda_physics', None)
                if ck_lambda is None:
                    continue
                if abs(float(ck_lambda) - lambda_key) <= tol:
                    info = {
                        'lambda_phy': lambda_key,
                        'model_type': 'pigru',
                        'model_dir': model_dir,
                        'model_path': model_path
                    }
                    self.lambda_model_registry[lambda_key] = info
                    return info
            except Exception:
                continue

        return None
    
    def load_normalization_params(self):
        """加载归一化参数"""
        model_save_path = self.config['training']['model_save_path']
        if not os.path.isabs(model_save_path):
            model_save_path = os.path.join(project_root, model_save_path.lstrip('../'))
        
        norm_path = os.path.join(model_save_path, 'norm_params.pkl')
        
        with open(norm_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']
        
        self.wind_mean = self.scaler_y.mean_[0:3]
        self.wind_std = self.scaler_y.scale_[0:3]
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载同分布测试数据（ID）"""
        data_dir = self.config['data']['processed_dir']
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(project_root, data_dir.lstrip('../'))
        
        # 使用同分布测试集 test_id
        X_test = np.load(os.path.join(data_dir, 'X_test_id.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test_id.npy'))
        
        return X_test, y_test
    
    def load_ood_suite(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """加载多场景 OOD 测试集（X/y 一一对应）"""
        data_dir = self._resolve_path(self.config['data']['processed_dir'])
        suite = {}

        for scene in self.ood_scenes:
            x_path = os.path.join(data_dir, f'X_{scene}.npy')
            y_path = os.path.join(data_dir, f'y_{scene}.npy')
            if not os.path.exists(x_path) or not os.path.exists(y_path):
                print(f"  ⚠️ 跳过场景 {scene}: 缺少 {x_path} 或 {y_path}")
                continue
            suite[scene] = (np.load(x_path), np.load(y_path))

        if not suite:
            raise FileNotFoundError(f"未找到可用 OOD 场景数据，目录: {data_dir}")

        return suite

    @staticmethod
    def _aggregate_ood_metrics(scene_metrics: Dict[str, Dict]) -> Tuple[float, float]:
        vals = [m['total_rmse'] for m in scene_metrics.values() if m is not None and not np.isnan(m['total_rmse'])]
        if not vals:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.max(vals))
    
    def evaluate_model(self, model_path: str, X_test: np.ndarray, 
                       y_test: np.ndarray, model_type: str = 'pigru') -> Dict:
        """
        评估单个模型
        
        Args:
            model_path: 模型路径
            X_test: 测试输入
            y_test: 测试标签
            model_type: 模型类型 ('pigru' 或 'vanilla_gru')
        
        Returns:
            评估指标字典
        """
        self._require_torch()

        # 动态导入模型

        if model_type == 'pigru':
            import importlib.util
            model_file = os.path.join(project_root, 'src', '2_pigru_module.py')
            spec = importlib.util.spec_from_file_location("model_def", model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            ModelClass = model_module.PIGRU
        else:
            import importlib.util
            model_file = os.path.join(project_root, 'src', '2b_vanilla_gru.py')
            spec = importlib.util.spec_from_file_location("vanilla_gru", model_file)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            ModelClass = model_module.VanillaGRU
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        model = ModelClass(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=0.0
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 预测
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        wind_pred_norm = []
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), 256):
                batch = X_tensor[i:i+256]
                if model_type == 'pigru':
                    out = model(batch, return_dict=True)
                    wind = out['wind_estimate']
                else:
                    wind = model(batch, return_dict=False)
                wind_pred_norm.append(wind.cpu().numpy())
        
        wind_pred_norm = np.vstack(wind_pred_norm)
        wind_pred = wind_pred_norm * self.wind_std + self.wind_mean
        
        # 真值
        y_denorm = self.scaler_y.inverse_transform(y_test)
        wind_true = y_denorm[:, 0:3]
        
        # 计算指标
        metrics = {}
        for i, name in enumerate(['north', 'east', 'down']):
            metrics[f'{name}_rmse'] = _rmse(wind_true[:, i], wind_pred[:, i])

        metrics['total_rmse'] = _rmse(wind_true, wind_pred)
        metrics['vector_error'] = np.mean(np.linalg.norm(wind_pred - wind_true, axis=1))

        
        return metrics

    def _calc_metrics_from_norm(self, wind_pred_norm: np.ndarray, y_test_norm: np.ndarray) -> Dict:
        """由归一化预测与标签统一计算指标"""
        wind_pred = wind_pred_norm * self.wind_std + self.wind_mean
        y_denorm = self.scaler_y.inverse_transform(y_test_norm)
        wind_true = y_denorm[:, 0:3]

        metrics = {}
        for i, name in enumerate(['north', 'east', 'down']):
            metrics[f'{name}_rmse'] = _rmse(wind_true[:, i], wind_pred[:, i])
        metrics['total_rmse'] = _rmse(wind_true, wind_pred)
        metrics['vector_error'] = np.mean(np.linalg.norm(wind_pred - wind_true, axis=1))
        return metrics

    def _load_pirnn_akf_class(self):
        self._require_torch()
        import importlib.util

        module_file = os.path.join(project_root, 'src', '5_pigru_akf_fusion.py')
        spec = importlib.util.spec_from_file_location('pirnn_akf_fusion_for_ablation', module_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.PIRNN_AKF

    def evaluate_system_model(self, model_path: str, X_test: np.ndarray, y_test: np.ndarray,
                               enable_q_scale: bool = True, enable_r_scale: bool = True) -> Dict:
        """完整 PIRNN-AKF 系统评估"""
        PIRNN_AKF = self._load_pirnn_akf_class()
        estimator = PIRNN_AKF(
            config_path=self.config_path,
            model_path=model_path,
            enable_q_scale=enable_q_scale,
            enable_r_scale=enable_r_scale
        )
        estimator.reset()
        wind_pred_norm, _ = estimator.estimate_batch(X_test)
        return self._calc_metrics_from_norm(wind_pred_norm, y_test)
    
    def run_lambda_ablation(self, lambda_values: List[float] = [0.0, 0.1, 0.5, 1.0, 2.0]):
        """实验 A：裸网络 λ 消融（多 OOD 场景）"""
        print("\n" + "="*70)
        print("实验 A: λ_phy 裸网络消融（多 OOD 场景）")
        print("="*70)

        X_id, y_id = self.load_test_data()
        ood_suite = self.load_ood_suite()

        results = []
        for lambda_val in lambda_values:
            print(f"\n【λ_phy = {lambda_val}】")
            model_info = self.find_model_for_lambda(float(lambda_val))
            if model_info is None:
                print(f"  ⚠️ 未找到λ={lambda_val}的模型，跳过")
                continue

            model_path = model_info['model_path']
            model_type = model_info['model_type']
            if not os.path.exists(model_path):
                print(f"  ⚠️ 模型文件不存在: {model_path}")
                continue

            try:
                id_metrics = self.evaluate_model(model_path, X_id, y_id, model_type)
                row = {
                    'lambda_phy': float(lambda_val),
                    'model_type': model_type,
                    'model_dir': model_info['model_dir'],
                    'id_total_rmse': id_metrics['total_rmse'],
                    'id_vector_error': id_metrics['vector_error']
                }
                for scene in self.ood_scenes:
                    row[f'{scene}_rmse'] = np.nan

                scene_metrics = {}
                for scene, (X_ood, y_ood) in ood_suite.items():
                    m = self.evaluate_model(model_path, X_ood, y_ood, model_type)
                    scene_metrics[scene] = m
                    row[f'{scene}_rmse'] = m['total_rmse']

                ood_macro, ood_worst = self._aggregate_ood_metrics(scene_metrics)
                row['ood_macro'] = ood_macro
                row['ood_worst'] = ood_worst
                row['degradation_pct_macro'] = (ood_macro / id_metrics['total_rmse'] - 1.0) * 100 if id_metrics['total_rmse'] > 0 else np.nan
                results.append(row)

                print(f"  ID-RMSE: {id_metrics['total_rmse']:.3f} | OOD-macro: {ood_macro:.3f} | OOD-worst: {ood_worst:.3f}")
            except Exception as e:
                print(f"  ⚠️ 评估失败: {e}")

        self.ablation_results['network_multi_ood'] = results
        if results:
            df = pd.DataFrame(results)
            out_csv = os.path.join(self.output_dir, 'ablation_network_multi_ood.csv')
            df.to_csv(out_csv, index=False)
            print(f"\n  ✓ 实验A结果已保存: {out_csv}")
        return results
    
    def run_system_lambda_ablation(self, lambda_values: List[float] = [0.0, 0.1, 0.5, 1.0, 2.0]):
        """实验 B：系统级 λ 消融（PIRNN-AKF，多 OOD 场景）"""
        print("\n" + "="*70)
        print("实验 B: 系统级 λ 消融（多 OOD 场景）")
        print("="*70)

        X_id, y_id = self.load_test_data()
        ood_suite = self.load_ood_suite()
        results = []

        for lambda_val in lambda_values:
            print(f"\n【系统级 λ={lambda_val}】")
            model_info = self.find_pigru_model_for_lambda(float(lambda_val))
            if model_info is None:
                fallback = self.find_model_for_lambda(float(lambda_val))
                if fallback is not None and fallback.get('model_type') == 'vanilla_gru':
                    print(f"  ⚠️ λ={lambda_val} 仅找到 Vanilla GRU；系统级实验需要 PI-GRU(同 λ) checkpoint，跳过")
                else:
                    print(f"  ⚠️ 未找到 λ={lambda_val} 的 PI-GRU 模型，跳过")
                continue

            row = {
                'lambda_phy': float(lambda_val),
                'model_type': model_info['model_type'],
                'model_dir': model_info['model_dir']
            }
            for scene in self.ood_scenes:
                row[f'sys_{scene}_rmse'] = np.nan

            id_metrics = self.evaluate_system_model(
                model_path=model_info['model_path'],
                X_test=X_id,
                y_test=y_id,
            )
            row['sys_id_rmse'] = id_metrics['total_rmse']

            scene_metrics = {}
            for scene, (X_ood, y_ood) in ood_suite.items():
                m = self.evaluate_system_model(
                    model_path=model_info['model_path'],
                    X_test=X_ood,
                    y_test=y_ood,
                )
                scene_metrics[scene] = m
                row[f'sys_{scene}_rmse'] = m['total_rmse']

            macro_rmse, worst_rmse = self._aggregate_ood_metrics(scene_metrics)
            row['sys_ood_macro'] = macro_rmse
            row['sys_ood_worst'] = worst_rmse
            row['sys_degradation_pct_macro'] = (macro_rmse / row['sys_id_rmse'] - 1.0) * 100 if row['sys_id_rmse'] > 0 else np.nan
            results.append(row)

            print(f"  SYS-ID: {row['sys_id_rmse']:.3f} | SYS-OOD-macro: {macro_rmse:.3f} | SYS-OOD-worst: {worst_rmse:.3f}")

        self.ablation_results['system_multi_ood'] = results
        if results:
            df = pd.DataFrame(results)
            out_csv = os.path.join(self.output_dir, 'ablation_system_multi_ood.csv')
            df.to_csv(out_csv, index=False)
            print(f"\n  ✓ 实验B结果已保存: {out_csv}")
        return results

    def run_covariance_ablation(self, lambda_fixed: float = 0.1):
        """实验 C：协方差头拆解（固定 λ，系统级，多 OOD 场景）"""
        print("\n" + "="*70)
        print(f"实验 C: 协方差头拆解（λ={lambda_fixed}）")
        print("="*70)

        model_info = self.find_pigru_model_for_lambda(float(lambda_fixed))
        if model_info is None:
            fallback = self.find_model_for_lambda(float(lambda_fixed))
            if fallback is not None and fallback.get('model_type') == 'vanilla_gru':
                raise ValueError(f"协方差拆解实验需要 PI-GRU(λ={lambda_fixed})，当前仅找到 Vanilla GRU")
            raise FileNotFoundError(f"未找到 λ={lambda_fixed} 的 PI-GRU 模型")

        X_id, y_id = self.load_test_data()
        ood_suite = self.load_ood_suite()

        configs = [
            ('C1_fixed', False, False),
            ('C2_q_scale_only', True, False),
            ('C3_r_scale_only', False, True),
            ('C4_full', True, True)
        ]

        results = []
        for name, ena_q, ena_r in configs:
            print(f"\n【{name}】 q_scale={ena_q} r_scale={ena_r}")
            row = {
                'config': name,
                'lambda_phy': float(lambda_fixed),
                'model_type': model_info['model_type'],
                'model_dir': model_info['model_dir']
            }
            for scene in self.ood_scenes:
                row[f'sys_{scene}_rmse'] = np.nan

            id_metrics = self.evaluate_system_model(
                model_path=model_info['model_path'],
                X_test=X_id,
                y_test=y_id,
                enable_q_scale=ena_q,
                enable_r_scale=ena_r
            )
            row['sys_id_rmse'] = id_metrics['total_rmse']

            scene_metrics = {}
            for scene, (X_ood, y_ood) in ood_suite.items():
                m = self.evaluate_system_model(
                    model_path=model_info['model_path'],
                    X_test=X_ood,
                    y_test=y_ood,
                    enable_q_scale=ena_q,
                    enable_r_scale=ena_r
                )
                scene_metrics[scene] = m
                row[f'sys_{scene}_rmse'] = m['total_rmse']

            macro_rmse, worst_rmse = self._aggregate_ood_metrics(scene_metrics)
            row['sys_ood_macro'] = macro_rmse
            row['sys_ood_worst'] = worst_rmse
            row['sys_degradation_pct_macro'] = (macro_rmse / row['sys_id_rmse'] - 1.0) * 100 if row['sys_id_rmse'] > 0 else np.nan
            results.append(row)

            print(f"  SYS-ID: {row['sys_id_rmse']:.3f} | SYS-OOD-macro: {macro_rmse:.3f} | SYS-OOD-worst: {worst_rmse:.3f}")

        self.ablation_results['covariance_multi_ood'] = results
        if results:
            df = pd.DataFrame(results)
            out_csv = os.path.join(self.output_dir, 'ablation_covariance_multi_ood.csv')
            df.to_csv(out_csv, index=False)
            print(f"\n  ✓ 实验C结果已保存: {out_csv}")
        return results

    def generate_lambda_ood_summary(self) -> pd.DataFrame:
        """兼容接口：导出实验 A 多 OOD 汇总"""
        rows = self.ablation_results.get('network_multi_ood', [])
        df = pd.DataFrame(rows)
        if len(df) > 0:
            summary_csv = os.path.join(self.output_dir, 'lambda_id_ood_summary.csv')
            df.to_csv(summary_csv, index=False)
            print(f"\n  ✓ λ-ID/OOD汇总已保存: {summary_csv}")

            report_md = os.path.join(self.output_dir, 'lambda_id_ood_summary.md')
            with open(report_md, 'w', encoding='utf-8') as f:
                f.write("# λ 消融 ID/OOD 汇总（多 OOD 场景）\n\n")
                f.write(_df_to_markdown(df))
                f.write("\n")
            print(f"  ✓ λ-ID/OOD Markdown汇总已保存: {report_md}")
        return df
    
    def _apply_publication_style(self):
        """统一论文风格绘图参数"""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 9,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'axes.linewidth': 0.8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'legend.frameon': False,
            'grid.linewidth': 0.6,
            'lines.linewidth': 1.8,
            'lines.markersize': 5,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.02,
            'figure.dpi': 150
        })

    def _save_figure(self, fig, file_stem: str):
        """统一保存高质量图像"""
        for ext in ('pdf', 'svg'):
            out_path = os.path.join(self.output_dir, f'{file_stem}.{ext}')
            fig.savefig(out_path, format=ext, facecolor='white')
            print(f"  ✓ 图已保存: {out_path}")

        png_path = os.path.join(self.output_dir, f'{file_stem}.png')
        fig.savefig(png_path, format='png', dpi=600, facecolor='white')
        print(f"  ✓ 图已保存: {png_path}")

    def _scene_display_labels(self) -> List[str]:
        label_map = {
            'ood_0p5x': '0.5×',
            'ood_0p75x': '0.75×',
            'ood_1p0x': '1.0×',
            'ood_1p25x': '1.25×',
            'ood_1p5x': '1.5×',
            'ood_low_speed': 'low-speed',
            'ood_turb_heavy': 'heavy-turb'
        }
        return [label_map.get(scene, scene) for scene in self.ood_scenes]

    def _covariance_display_labels(self) -> Dict[str, str]:
        return {
            'C1_fixed': 'C1 Fixed',
            'C2_q_scale_only': 'C2 q_scale-only',
            'C3_r_scale_only': 'C3 r_scale-only',
            'C4_full': 'C4 Full'
        }

    def _prepare_covariance_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # 兼容历史结果中的旧命名（alphaQ/beta）
        rename_map = {
            'C2_alphaQ_only': 'C2_q_scale_only',
            'C3_beta_only': 'C3_r_scale_only'
        }
        order = ['C1_fixed', 'C2_q_scale_only', 'C3_r_scale_only', 'C4_full']
        df = df.copy()
        df['config'] = df['config'].replace(rename_map)
        df['config'] = pd.Categorical(df['config'], categories=order, ordered=True)
        return df.sort_values(by='config').reset_index(drop=True)

    @staticmethod
    def _clean_small_delta(value_cm: float, tol_cm: float = 0.005) -> float:
        if np.isnan(value_cm):
            return value_cm
        return 0.0 if abs(value_cm) < tol_cm else value_cm

    def _format_delta_label(self, value_cm: float) -> str:
        if np.isnan(value_cm):
            return 'NA'
        value_cm = self._clean_small_delta(value_cm)
        return f'{value_cm:+.2f}'

    def _build_heatmap_matrix(self, df: pd.DataFrame, metric_cols: List[str]):
        if df.empty or not metric_cols:
            return None, None, None

        df = df.sort_values(by='lambda_phy').reset_index(drop=True)
        matrix = df[metric_cols].to_numpy(dtype=float).T
        if matrix.size == 0 or np.all(np.isnan(matrix)):
            return None, None, None
        x_labels = [f'{v:g}' for v in df['lambda_phy'].tolist()]
        return df, matrix, x_labels

    def _annotate_heatmap(self, ax, matrix: np.ndarray, vmin: float, vmax: float):
        threshold = (vmin + vmax) / 2.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    text = 'NA'
                    color = 'black'
                else:
                    text = f'{val:.2f}'
                    color = 'white' if val >= threshold else 'black'
                ax.text(j, i, text, ha='center', va='center', fontsize=7, color=color)

    def _plot_lambda_panel(self, ax, df: pd.DataFrame, id_col: str, macro_col: str, worst_col: str,
                           title: str, show_ylabel: bool = False):
        if df.empty:
            return

        df = df.sort_values(by='lambda_phy')
        x_vals = df['lambda_phy'].to_numpy(dtype=float)
        ax.plot(x_vals, df[id_col].to_numpy(dtype=float), marker='o', color='#1d4f91', label='ID')
        ax.plot(x_vals, df[macro_col].to_numpy(dtype=float), marker='s', color='#e17c05', label='OOD-macro')
        ax.plot(x_vals, df[worst_col].to_numpy(dtype=float), marker='^', color='#b22222', label='OOD-worst')

        best_idx = df[macro_col].astype(float).idxmin()
        best_lambda = df.loc[best_idx, 'lambda_phy']
        best_macro = df.loc[best_idx, macro_col]
        ax.scatter([best_lambda], [best_macro], s=34, color='black', zorder=5)
        ax.annotate(
            f'λ*={best_lambda:g}',
            xy=(best_lambda, best_macro),
            xytext=(5, -12),
            textcoords='offset points',
            fontsize=6.8,
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.92)
        )

        ax.set_title(title, pad=4)
        ax.set_xlabel('Physics loss weight λ')
        ax.set_ylabel('RMSE (m/s)' if show_ylabel else '')
        ax.grid(True, alpha=0.22)
        ax.set_xticks(x_vals)
        ax.margins(x=0.05)

    def _plot_ood_heatmap(self, df: pd.DataFrame, metric_cols: List[str], title: str, file_stem: str,
                          vmin: float, vmax: float):
        built = self._build_heatmap_matrix(df, metric_cols)
        if built[0] is None:
            return

        df, matrix, x_labels = built
        fig, ax = plt.subplots(figsize=(7.0, 3.45))
        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax)

        ax.set_title(title, pad=4)
        ax.set_xlabel('Physics loss weight λ')
        ax.set_ylabel('OOD scenario')
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)
        ax.set_yticks(np.arange(len(self.ood_scenes)))
        ax.set_yticklabels(self._scene_display_labels())
        self._annotate_heatmap(ax, matrix, vmin=vmin, vmax=vmax)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label('RMSE (m/s)')
        fig.tight_layout()
        self._save_figure(fig, file_stem)
        plt.close(fig)

    def _plot_covariance_delta_panel(self, ax, df: pd.DataFrame, show_ylabels: bool = False):
        baseline = df.iloc[0]
        y_pos = np.arange(len(df))
        labels = self._covariance_display_labels()
        delta_cols = [
            ('sys_ood_macro', 'Δ macro', '#e17c05'),
            ('sys_ood_worst', 'Δ worst', '#b22222')
        ]
        delta_offsets = [-0.16, 0.16]
        all_delta = []

        for (col, label, color), offset in zip(delta_cols, delta_offsets):
            raw_delta = (df[col].to_numpy(dtype=float) - float(baseline[col])) * 100.0
            delta_cm = np.array([self._clean_small_delta(val) for val in raw_delta], dtype=float)
            all_delta.extend(delta_cm.tolist())
            ax.barh(y_pos + offset, delta_cm, height=0.28, color=color, alpha=0.88, label=label)

        max_abs = max(max(abs(v) for v in all_delta if not np.isnan(v)), 0.04)
        x_pad = max_abs * 0.24
        ax.set_xlim(-max_abs - x_pad, max_abs + 1.8 * x_pad)

        for (col, _, _), offset in zip(delta_cols, delta_offsets):
            delta_cm = (df[col].to_numpy(dtype=float) - float(baseline[col])) * 100.0
            delta_cm = np.array([self._clean_small_delta(val) for val in delta_cm], dtype=float)
            label_pad = max_abs * 0.08
            for idx, val in enumerate(delta_cm):
                ha = 'left' if val >= 0 else 'right'
                x_text = val + label_pad if val >= 0 else val - label_pad
                ax.text(x_text, y_pos[idx] + offset, self._format_delta_label(val), va='center', ha=ha, fontsize=6.8)

        ax.axvline(0.0, color='black', linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([labels.get(v, str(v)) for v in df['config'].astype(str)] if show_ylabels else [])
        ax.invert_yaxis()
        ax.set_xlabel('ΔRMSE vs C1 (cm/s)')
        ax.set_title('(b) OOD delta vs C1', pad=4)
        ax.grid(True, axis='x', alpha=0.22)

    def _plot_covariance_summary(self, df: pd.DataFrame, file_stem: str):
        if df.empty:
            return

        df = self._prepare_covariance_df(df)
        labels = self._covariance_display_labels()
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={'width_ratios': [1.2, 1.0]}, sharey=True)
        y_pos = np.arange(len(df))
        metrics = [
            ('sys_id_rmse', 'ID', '#1d4f91'),
            ('sys_ood_macro', 'OOD-macro', '#e17c05'),
            ('sys_ood_worst', 'OOD-worst', '#b22222')
        ]
        offsets = [-0.18, 0.0, 0.18]

        for (col, label, color), offset in zip(metrics, offsets):
            axes[0].plot(df[col].to_numpy(dtype=float), y_pos + offset, marker='o', color=color, label=label)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels([labels.get(v, str(v)) for v in df['config'].astype(str)])
        axes[0].invert_yaxis()
        axes[0].set_xlabel('RMSE (m/s)')
        axes[0].set_title('(a) Absolute RMSE', pad=4)
        axes[0].grid(True, axis='x', alpha=0.22)
        axes[0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=3, columnspacing=1.2, handletextpad=0.5)

        self._plot_covariance_delta_panel(axes[1], df, show_ylabels=False)
        axes[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2, columnspacing=1.2, handletextpad=0.5)

        fig.tight_layout(rect=[0, 0, 1, 0.96])

        self._save_figure(fig, file_stem)
        plt.close(fig)

    def _plot_covariance_multi_ood(self, df: pd.DataFrame, file_stem: str):
        if df.empty:
            return

        df = self._prepare_covariance_df(df)
        metric_cols = [f'sys_{scene}_rmse' for scene in self.ood_scenes if f'sys_{scene}_rmse' in df.columns]
        if not metric_cols:
            return

        matrix = df[metric_cols].to_numpy(dtype=float).T
        if matrix.size == 0 or np.all(np.isnan(matrix)):
            return

        config_short = {
            'C1_fixed': 'C1',
            'C2_q_scale_only': 'C2',
            'C3_r_scale_only': 'C3',
            'C4_full': 'C4'
        }
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), gridspec_kw={'width_ratios': [1.25, 1.0]})
        vmin = float(np.nanmin(matrix))
        vmax = float(np.nanmax(matrix))
        im = axes[0].imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=vmin, vmax=vmax)
        axes[0].set_title('(a) OOD RMSE by covariance head', pad=4)
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('OOD scenario')
        axes[0].set_xticks(np.arange(len(df)))
        axes[0].set_xticklabels([config_short.get(v, str(v)) for v in df['config'].astype(str)])
        axes[0].set_yticks(np.arange(len(self.ood_scenes)))
        axes[0].set_yticklabels(self._scene_display_labels())
        self._annotate_heatmap(axes[0], matrix, vmin=vmin, vmax=vmax)
        cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
        cbar.set_label('RMSE (m/s)')

        self._plot_covariance_delta_panel(axes[1], df, show_ylabels=False)
        axes[1].legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=2, columnspacing=1.2, handletextpad=0.5)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_figure(fig, file_stem)
        plt.close(fig)

    def generate_multi_ood_figures(self):
        """生成论文风格多 OOD 消融图（A/B/C）"""
        print("\n" + "="*70)
        print("生成多 OOD 消融图")
        print("="*70)
        self._apply_publication_style()

        network_df = pd.DataFrame(self.ablation_results.get('network_multi_ood', []))
        system_df = pd.DataFrame(self.ablation_results.get('system_multi_ood', []))
        covariance_df = pd.DataFrame(self.ablation_results.get('covariance_multi_ood', []))

        if not network_df.empty and not system_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=True)
            self._plot_lambda_panel(axes[0], network_df, 'id_total_rmse', 'ood_macro', 'ood_worst', '(a) Network-level', show_ylabel=True)
            self._plot_lambda_panel(axes[1], system_df, 'sys_id_rmse', 'sys_ood_macro', 'sys_ood_worst', '(b) System-level', show_ylabel=False)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.02))
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            self._save_figure(fig, 'figure_ablation_lambda_tradeoff')
            plt.close(fig)

        network_metric_cols = [f'{scene}_rmse' for scene in self.ood_scenes if f'{scene}_rmse' in network_df.columns] if not network_df.empty else []
        system_metric_cols = [f'sys_{scene}_rmse' for scene in self.ood_scenes if f'sys_{scene}_rmse' in system_df.columns] if not system_df.empty else []
        network_built = self._build_heatmap_matrix(network_df, network_metric_cols) if network_metric_cols else (None, None, None)
        system_built = self._build_heatmap_matrix(system_df, system_metric_cols) if system_metric_cols else (None, None, None)
        heatmap_values = []
        for built in (network_built, system_built):
            matrix = built[1]
            if matrix is not None:
                heatmap_values.append(matrix)

        if heatmap_values:
            combined = np.concatenate([m.ravel() for m in heatmap_values])
            combined = combined[~np.isnan(combined)]
            global_vmin = float(np.min(combined))
            global_vmax = float(np.max(combined))

            if network_metric_cols:
                self._plot_ood_heatmap(
                    network_df,
                    network_metric_cols,
                    'Network-level OOD',
                    'figure_ablation_network_multi_ood',
                    vmin=global_vmin,
                    vmax=global_vmax
                )

            if system_metric_cols:
                self._plot_ood_heatmap(
                    system_df,
                    system_metric_cols,
                    'System-level OOD',
                    'figure_ablation_system_multi_ood',
                    vmin=global_vmin,
                    vmax=global_vmax
                )

        if not covariance_df.empty:
            self._plot_covariance_multi_ood(covariance_df, 'figure_ablation_covariance_multi_ood')
            self._plot_covariance_summary(covariance_df, 'figure_ablation_covariance_summary')


    
    def save_all_results(self):
        """保存所有消融实验结果"""
        results_path = os.path.join(self.output_dir, 'ablation_results.pkl')
        
        with open(results_path, 'wb') as f:
            pickle.dump({
                'ablation_results': self.ablation_results,
                'ood_results': self.ood_results,
                'timestamp': self.timestamp
            }, f)
        
        print(f"\n  ✓ 完整结果已保存: {results_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='消融实验：多 OOD 场景（network/system/covariance）')
    parser.add_argument(
        '--exp',
        type=str,
        default='all',
        choices=['all', 'network', 'system', 'covariance'],
        help='选择实验类型'
    )
    parser.add_argument(
        '--lambda-values',
        type=str,
        default='0.0,0.1,0.5,1.0,2.0',
        help='逗号分隔的 λ_phy 列表，例如 0.0,0.1,0.5,1.0,2.0'
    )
    parser.add_argument('--lambda-fixed', type=float, default=0.1, help='协方差拆解实验使用的固定 λ')
    parser.add_argument(
        '--ood-scenes',
        type=str,
        default='',
        help='逗号分隔 OOD 场景，留空表示默认7场景'
    )
    parser.add_argument('--auto-train', action='store_true', help='自动批量训练各 λ 模型')
    parser.add_argument('--force-retrain', action='store_true', help='即使已有模型也重新训练')
    parser.add_argument('--num-epochs', type=int, default=None, help='覆盖训练轮数（调试加速）')
    parser.add_argument('--max-train-samples', type=int, default=None, help='训练子集样本数（快速预扫描）')
    parser.add_argument('--max-val-samples', type=int, default=None, help='验证子集样本数（快速预扫描）')
    parser.add_argument(
        '--zero-lambda-model',
        type=str,
        default='auto',
        choices=['auto', 'vanilla', 'pigru'],
        help='λ=0 训练类型：auto(网络实验用vanilla，系统实验用pigru)'
    )
    parser.add_argument('--no-figures', action='store_true', help='不生成图像文件')
    parser.add_argument('--redraw-only', action='store_true', help='仅从已有结果目录重绘图表，不重新评估')
    parser.add_argument('--result-dir', type=str, default='', help='已有消融结果目录（用于 --redraw-only）')
    return parser.parse_args()



def main():
    print("="*70)
    print(" 消融实验 (Ablation Study) - 多 OOD 场景")
    print("="*70)
    args = parse_args()
    lambda_values = [float(v.strip()) for v in args.lambda_values.split(',') if v.strip()]
    ood_scenes = [v.strip() for v in args.ood_scenes.split(',') if v.strip()] if args.ood_scenes else None

    try:
        if args.redraw_only:
            if not args.result_dir:
                raise ValueError('--redraw-only 模式必须提供 --result-dir')
            study = AblationStudy(
                ood_scenes=ood_scenes,
                output_dir=args.result_dir,
                load_norm_params=False
            )
            study.load_existing_results(args.result_dir)
            if not args.no_figures:
                study.generate_multi_ood_figures()

            study.save_all_results()
        else:
            study = AblationStudy(ood_scenes=ood_scenes)

            if args.auto_train:
                if args.zero_lambda_model == 'auto':
                    prefer_pigru_for_zero = args.exp in ('system', 'all')
                else:
                    prefer_pigru_for_zero = (args.zero_lambda_model == 'pigru')

                study.train_models_for_lambdas(
                    lambda_values=lambda_values,
                    num_epochs=args.num_epochs,
                    skip_existing=not args.force_retrain,
                    max_train_samples=args.max_train_samples,
                    max_val_samples=args.max_val_samples,
                    prefer_pigru_for_zero=prefer_pigru_for_zero
                )

            if args.exp in ('all', 'network'):
                study.run_lambda_ablation(lambda_values=lambda_values)
                study.generate_lambda_ood_summary()

            if args.exp in ('all', 'system'):
                study.run_system_lambda_ablation(lambda_values=lambda_values)

            if args.exp in ('all', 'covariance'):
                study.run_covariance_ablation(lambda_fixed=args.lambda_fixed)

            if not args.no_figures:
                study.generate_multi_ood_figures()

            study.save_all_results()


        print("\n" + "="*70)
        print("✅ 消融实验完成！")
        print(f"输出目录: {study.output_dir}")
        print("="*70)

    except Exception as e:
        print(f"\n❌ 消融实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
