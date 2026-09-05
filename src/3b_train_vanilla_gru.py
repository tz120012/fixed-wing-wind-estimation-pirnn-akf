"""
Vanilla GRU 训练模块 - 论文对比基线
特点：
  - 不使用物理约束损失
  - 与 3_train_pigru.py 对齐方向/模值监督与组合选模逻辑
  - 用于验证物理项与结构项是否真正带来收益
"""

import os
import argparse
import numpy as np
import random
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pickle
from datetime import datetime
from torch.utils.tensorboard.writer import SummaryWriter

# 导入Vanilla GRU模型
import importlib.util
model_file = os.path.join(os.path.dirname(__file__), '2b_vanilla_gru.py')
spec = importlib.util.spec_from_file_location("vanilla_gru", model_file)
if spec and spec.loader:
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    VanillaGRU = model_module.VanillaGRU
else:
    raise ImportError("Cannot load 2b_vanilla_gru.py")


class VanillaGRUTrainer:
    """Vanilla GRU 训练器（公平 baseline：方向/模值监督 + 组合选模）"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.seed = int(self.config.get('training', {}).get('seed', 26))
        self.set_global_seed(self.seed)
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        print(f"随机种子: {self.seed}")
        if self.device.type == 'cuda':
            print(f"  GPU型号: {torch.cuda.get_device_name(0)}")
        
        self.model = VanillaGRU(
            input_size=self.config['model']['input_size'],
            hidden_size=self.config['model']['hidden_size'],
            num_layers=self.config['model']['num_layers'],
            dropout=self.config['model']['dropout'],
            rnn_type=self.config['model'].get('rnn_type', 'gru'),
        ).to(self.device)
        
        model_info = self.model.get_model_info()
        self.model_name = model_info['model_type']
        print(f"\n【{model_info['model_type']} 模型配置】")
        print(f"  模型类型: {model_info['model_type']}")
        print(f"  输入维度: {model_info['input_size']}")
        print(f"  隐藏层维度: {model_info['hidden_size']}")
        print(f"  循环骨干: {model_info['rnn_type'].upper()}")
        print(f"  循环层数: {model_info['num_layers']}")
        print(f"  总参数量: {model_info['total_params']:,}")
        print(f"  物理损失: {'启用' if model_info['has_physics_loss'] else '禁用 ⭐'}")
        
        self.load_normalization_params()

        self.lambda_wind = float(self.config['training'].get('lambda_wind', 1.0))
        self.lambda_dir = float(self.config['training'].get('lambda_dir', 0.3))
        self.lambda_mag = float(self.config['training'].get('lambda_mag', 0.2))
        self.direction_loss_min_horizontal_wind = float(
            self.config['training'].get('direction_loss_min_horizontal_wind', 0.5)
        )
        self.selection_alpha = float(self.config['training'].get('selection_alpha', 0.5))
        self.selection_beta = float(self.config['training'].get('selection_beta', 0.5))
        self.selection_metric_name = self.config['training'].get('selection_metric', 'composite_score')
        if self.selection_metric_name == 'total':
            self.selection_metric_name = 'loss'

        self.config['training']['lambda_wind'] = float(self.lambda_wind)
        self.config['training']['lambda_dir'] = float(self.lambda_dir)
        self.config['training']['lambda_mag'] = float(self.lambda_mag)
        self.config['training']['direction_loss_min_horizontal_wind'] = float(self.direction_loss_min_horizontal_wind)
        self.config['training']['selection_alpha'] = float(self.selection_alpha)
        self.config['training']['selection_beta'] = float(self.selection_beta)
        self.config['training']['selection_metric'] = self.selection_metric_name

        _comp_w = self.config['training'].get('wind_component_weights', [1.0, 1.0, 0.1])
        self._wind_component_weights = torch.tensor(
            _comp_w, dtype=torch.float32, device=self.device
        )
        self.config['training']['wind_component_weights'] = [float(v) for v in _comp_w]

        dyn_w_cfg = self.config['training'].get('dynamic_sample_weight', {}) or {}
        self.dynamic_sample_weight_enabled = bool(dyn_w_cfg.get('enabled', True))
        self.dynamic_sample_weight_apply = set(
            (dyn_w_cfg.get('apply_to') or ['data', 'magnitude'])
        )
        clamp_range = dyn_w_cfg.get('clamp', [0.5, 5.0])
        self.dynamic_sample_weight_clamp = (
            float(clamp_range[0]) if clamp_range else 0.0,
            float(clamp_range[1]) if clamp_range and len(clamp_range) > 1 else 5.0,
        )
        self.config['training']['dynamic_sample_weight'] = {
            'enabled': self.dynamic_sample_weight_enabled,
            'apply_to': sorted(self.dynamic_sample_weight_apply),
            'clamp': list(self.dynamic_sample_weight_clamp),
        }
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training'].get('weight_decay', 0.0001)
        )
        
        scheduler_config = self.config.get('scheduler', {})
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 10),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
        
        self.early_stopping_patience = self.config['training']['early_stopping_patience']
        metric_labels = {
            'loss': '验证总损失',
            'rmse': '验证RMSE',
            'wind_mag_error': '验证风速大小MAE',
            'wind_mag_rmse': '验证风速大小RMSE',
            'wind_direction_error': '验证风向误差',
            'composite_score': '验证组合分数',
        }
        if self.selection_metric_name not in metric_labels:
            raise ValueError(f"不支持的 selection_metric: {self.selection_metric_name}")
        self.monitor_metric_name = self.selection_metric_name
        self.monitor_metric_label = metric_labels[self.monitor_metric_name]
        self.best_monitor_value = float('inf')
        self.best_val_loss = float('inf')
        self.best_val_rmse = float('inf')
        self.best_val_wind_mag_error = float('inf')
        self.best_val_wind_mag_rmse = float('inf')
        self.best_val_direction_error = float('inf')
        self.best_val_composite_score = float('inf')
        self.best_epochs = {
            'loss': None,
            'rmse': None,
            'wind_mag_error': None,
            'wind_mag_rmse': None,
            'wind_direction_error': None,
            'composite_score': None,
        }
        self.patience_counter = 0

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tensorboard_dir = self.config.get('logging', {}).get('tensorboard_dir', '../runs/')
        run_name = f"vanilla_gru_{self.timestamp}"
        self.writer = SummaryWriter(os.path.join(tensorboard_dir, run_name))
        
        base_save_path = self.config['training']['model_save_path']
        if not os.path.isabs(base_save_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            base_save_path = os.path.join(project_root, base_save_path.lstrip('../'))
        
        self.model_save_dir = os.path.join(base_save_path, f"vanilla_gru_{self.timestamp}")
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        print(f"\n【TensorBoard】")
        print(f"  日志目录: {os.path.join(tensorboard_dir, run_name)}")
        print(f"\n【模型保存目录】")
        print(f"  {self.model_save_dir}")
        print(f"\n【选模策略】")
        print(f"  主模型 `best_model.pth` 按 {self.monitor_metric_label} 保存")
        print("  额外保留 `best_rmse_model.pth`、`best_mag_model.pth`、`best_dir_model.pth`、`best_composite_model.pth`")
        
        self.history = {
            'train_loss': [],
            'train_data_loss': [],
            'train_dir_loss': [],
            'train_mag_loss': [],
            'val_loss': [],
            'val_data_loss': [],
            'val_dir_loss': [],
            'val_mag_loss': [],
            'learning_rate': [],
            'val_mae': [],
            'val_rmse': [],
            'val_wind_mag_error': [],
            'val_wind_mag_rmse': [],
            'val_wind_direction_error': [],
            'val_composite_score': []
        }

    @staticmethod
    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def load_normalization_params(self):
        """加载数据归一化参数"""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        candidate_dirs = []

        for path_value in [
            self.config.get('data', {}).get('processed_dir'),
            self.config.get('experiment', {}).get('processed_dir'),
            self.config['training'].get('model_save_path'),
        ]:
            if not path_value:
                continue
            if os.path.isabs(path_value):
                candidate_dirs.append(path_value)
            else:
                candidate_dirs.append(os.path.normpath(os.path.join(project_root, path_value.lstrip('../'))))

        norm_params_path = None
        for directory in candidate_dirs:
            candidate = os.path.join(directory, 'norm_params.pkl')
            if os.path.exists(candidate):
                norm_params_path = candidate
                break

        if norm_params_path is None:
            raise FileNotFoundError(
                f"归一化参数文件不存在，已尝试目录: {candidate_dirs}\n"
                f"请先运行 1_preprocessing_data.py"
            )
        
        with open(norm_params_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler_X = metadata['scaler_X']
        self.scaler_y = metadata['scaler_y']
        
        self.y_mean = torch.tensor(self.scaler_y.mean_, dtype=torch.float32).to(self.device)
        self.y_std = torch.tensor(self.scaler_y.scale_, dtype=torch.float32).to(self.device)
        
        self.wind_mean = self.y_mean[0:3]
        self.wind_std = self.y_std[0:3]
        
        print("\n【归一化参数加载成功】")
        print(f"  风速均值: [{self.wind_mean[0]:.2f}, {self.wind_mean[1]:.2f}, {self.wind_mean[2]:.2f}] m/s")
        print(f"  风速标准差: [{self.wind_std[0]:.2f}, {self.wind_std[1]:.2f}, {self.wind_std[2]:.2f}] m/s")

    def denormalize_wind(self, wind_estimate, y_batch):
        wind_truth_norm = y_batch[:, :3]
        wind_truth = wind_truth_norm * self.wind_std + self.wind_mean
        wind_estimate_real = wind_estimate * self.wind_std + self.wind_mean
        return wind_estimate_real, wind_truth

    def calculate_data_loss(self, wind_estimate, y_batch, sample_weight=None):
        wind_truth = y_batch[:, :3]
        comp_w = self._wind_component_weights.to(wind_estimate.device)
        per_dim = (wind_estimate - wind_truth).pow(2)
        per_sample = (per_dim * comp_w).sum(dim=1) / comp_w.sum().clamp(min=1e-6)
        if sample_weight is None or 'data' not in self.dynamic_sample_weight_apply:
            return per_sample.mean()
        w = sample_weight.view(-1).to(per_sample.dtype)
        return (per_sample * w).sum() / w.sum().clamp(min=1e-6)

    def calculate_direction_loss(self, wind_estimate, y_batch):
        wind_estimate_real, wind_truth = self.denormalize_wind(wind_estimate, y_batch)
        pred_horizontal = wind_estimate_real[:, :2]
        truth_horizontal = wind_truth[:, :2]
        truth_horizontal_mag = torch.norm(truth_horizontal, dim=1)
        valid_mask = truth_horizontal_mag >= max(self.direction_loss_min_horizontal_wind, 0.2)
        if not torch.any(valid_mask):
            return torch.zeros((), device=wind_estimate.device, dtype=wind_estimate.dtype)

        pred_horizontal = pred_horizontal[valid_mask]
        truth_horizontal = truth_horizontal[valid_mask]
        pred_horizontal_mag = torch.norm(pred_horizontal, dim=1)
        truth_horizontal_mag = torch.norm(truth_horizontal, dim=1)
        cosine = torch.sum(pred_horizontal * truth_horizontal, dim=1) / (
            pred_horizontal_mag.clamp(min=1e-6) * truth_horizontal_mag.clamp(min=1e-6)
        )
        cosine = torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6)
        return torch.mean(torch.acos(cosine) * (180.0 / torch.pi))

    def calculate_magnitude_loss(self, wind_estimate, y_batch, sample_weight=None):
        wind_estimate_real, wind_truth = self.denormalize_wind(wind_estimate, y_batch)
        pred_mag = torch.norm(wind_estimate_real, dim=1)
        truth_mag = torch.norm(wind_truth, dim=1)
        if sample_weight is None or 'magnitude' not in self.dynamic_sample_weight_apply:
            return F.smooth_l1_loss(pred_mag, truth_mag, beta=0.5)
        per_sample = F.smooth_l1_loss(pred_mag, truth_mag, beta=0.5, reduction='none')
        w = sample_weight.view(-1).to(per_sample.dtype)
        return (per_sample * w).sum() / w.sum().clamp(min=1e-6)

    def calculate_composite_score(self, rmse, wind_mag_rmse, wind_direction_error):
        return float(
            rmse
            + self.selection_alpha * wind_mag_rmse
            + self.selection_beta * (wind_direction_error / 180.0)
        )
    
    def calculate_metrics(self, wind_estimate, y_batch):
        wind_estimate_real, wind_truth = self.denormalize_wind(wind_estimate, y_batch)
        
        mae = torch.mean(torch.abs(wind_estimate_real - wind_truth)).item()
        rmse = torch.sqrt(torch.mean((wind_estimate_real - wind_truth) ** 2)).item()
        
        wind_mag_truth = torch.norm(wind_truth, dim=1)
        wind_mag_estimate = torch.norm(wind_estimate_real, dim=1)
        wind_mag_error = torch.mean(torch.abs(wind_mag_estimate - wind_mag_truth)).item()
        wind_mag_rmse = torch.sqrt(torch.mean((wind_mag_estimate - wind_mag_truth) ** 2)).item()

        truth_horizontal = wind_truth[:, :2]
        pred_horizontal = wind_estimate_real[:, :2]
        truth_horizontal_mag = torch.norm(truth_horizontal, dim=1)
        pred_horizontal_mag = torch.norm(pred_horizontal, dim=1)
        valid_mask = (
            (truth_horizontal_mag >= self.direction_loss_min_horizontal_wind)
            & (pred_horizontal_mag >= 1e-4)
        )
        if torch.any(valid_mask):
            pred_horizontal_valid = pred_horizontal[valid_mask]
            truth_horizontal_valid = truth_horizontal[valid_mask]
            pred_horizontal_mag = torch.norm(pred_horizontal_valid, dim=1)
            truth_horizontal_mag = torch.norm(truth_horizontal_valid, dim=1)
            cosine = torch.sum(pred_horizontal_valid * truth_horizontal_valid, dim=1) / (
                pred_horizontal_mag * truth_horizontal_mag + 1e-6
            )
            cosine = torch.clamp(cosine, -1.0, 1.0)
            wind_direction_error = torch.rad2deg(torch.acos(cosine)).mean().item()
        else:
            wind_direction_error = 0.0

        composite_score = self.calculate_composite_score(
            rmse=rmse,
            wind_mag_rmse=wind_mag_rmse,
            wind_direction_error=wind_direction_error,
        )
        
        return {
            'mae': mae,
            'rmse': rmse,
            'wind_mag_error': wind_mag_error,
            'wind_mag_rmse': wind_mag_rmse,
            'wind_direction_error': wind_direction_error,
            'composite_score': composite_score,
        }

    def train_epoch(self, train_loader, epoch):
        """训练一个epoch（无物理项，但保留方向/模值监督）"""
        self.model.train()
        epoch_total = 0.0
        epoch_data = 0.0
        epoch_dir = 0.0
        epoch_mag = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]', leave=False)
        
        for batch in pbar:
            if len(batch) == 3:
                X_batch, y_batch, w_batch = batch
                w_batch = w_batch.to(self.device)
            else:
                X_batch, y_batch = batch
                w_batch = None
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            wind_estimate = self.model(X_batch, return_dict=False)
            data_loss = self.calculate_data_loss(wind_estimate, y_batch, sample_weight=w_batch)
            dir_loss = self.calculate_direction_loss(wind_estimate, y_batch)
            mag_loss = self.calculate_magnitude_loss(wind_estimate, y_batch, sample_weight=w_batch)
            loss = (
                self.lambda_wind * data_loss
                + self.lambda_dir * dir_loss
                + self.lambda_mag * mag_loss
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.config['training'].get('gradient_clip_norm', 1.0)
            )
            self.optimizer.step()
            
            epoch_total += loss.item()
            epoch_data += data_loss.item()
            epoch_dir += dir_loss.item()
            epoch_mag += mag_loss.item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Data': f'{data_loss.item():.4f}',
                'Dir': f'{dir_loss.item():.4f}',
                'Mag': f'{mag_loss.item():.4f}'
            })
        
        num_batches = len(train_loader)
        return {
            'total': epoch_total / num_batches,
            'data': epoch_data / num_batches,
            'dir': epoch_dir / num_batches,
            'mag': epoch_mag / num_batches,
        }

    def validate(self, val_loader):
        """验证一个epoch"""
        self.model.eval()
        epoch_total = 0.0
        epoch_data = 0.0
        epoch_dir = 0.0
        epoch_mag = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        total_wind_mag_error = 0.0
        total_wind_mag_rmse = 0.0
        total_wind_direction_error = 0.0
        total_composite_score = 0.0
        
        pbar = tqdm(val_loader, desc='Validation', leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                if len(batch) == 3:
                    X_batch, y_batch, _ = batch
                else:
                    X_batch, y_batch = batch
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                wind_estimate = self.model(X_batch, return_dict=False)
                data_loss = self.calculate_data_loss(wind_estimate, y_batch)
                dir_loss = self.calculate_direction_loss(wind_estimate, y_batch)
                mag_loss = self.calculate_magnitude_loss(wind_estimate, y_batch)
                loss = (
                    self.lambda_wind * data_loss
                    + self.lambda_dir * dir_loss
                    + self.lambda_mag * mag_loss
                )
                
                epoch_total += loss.item()
                epoch_data += data_loss.item()
                epoch_dir += dir_loss.item()
                epoch_mag += mag_loss.item()
                
                metrics = self.calculate_metrics(wind_estimate, y_batch)
                total_mae += metrics['mae']
                total_rmse += metrics['rmse']
                total_wind_mag_error += metrics['wind_mag_error']
                total_wind_mag_rmse += metrics['wind_mag_rmse']
                total_wind_direction_error += metrics['wind_direction_error']
                total_composite_score += metrics['composite_score']
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'RMSE': f'{metrics["rmse"]:.3f}',
                    'Dir': f'{metrics["wind_direction_error"]:.2f}'
                })
        
        num_batches = len(val_loader)
        return {
            'loss': epoch_total / num_batches,
            'data': epoch_data / num_batches,
            'dir': epoch_dir / num_batches,
            'mag': epoch_mag / num_batches,
            'mae': total_mae / num_batches,
            'rmse': total_rmse / num_batches,
            'wind_mag_error': total_wind_mag_error / num_batches,
            'wind_mag_rmse': total_wind_mag_rmse / num_batches,
            'wind_direction_error': total_wind_direction_error / num_batches,
            'composite_score': total_composite_score / num_batches,
        }

    def train(self, X_train, y_train, X_val, y_val, w_train=None, w_val=None):
        """完整训练流程"""
        print(f"\n{'='*70}")
        print(f"开始 {self.model_name} 训练（公平 baseline：方向/模值监督，无物理约束）")
        print('='*70)
        print(f"训练集: {X_train.shape[0]} 样本")
        print(f"验证集: {X_val.shape[0]} 样本")

        use_sample_weight = self.dynamic_sample_weight_enabled and w_train is not None
        if use_sample_weight:
            clamp_lo, clamp_hi = self.dynamic_sample_weight_clamp
            w_train_arr = np.clip(np.asarray(w_train, dtype=np.float32), clamp_lo, clamp_hi)
            n_dyn = int(np.sum(w_train_arr > 1.0 + 1e-6))
            print(f"【sample_weight】启用，train 权重: mean={w_train_arr.mean():.3f} "
                  f"max={w_train_arr.max():.3f} 动态样本占比={100.0*n_dyn/max(len(w_train_arr),1):.1f}% "
                  f"(clamp={self.dynamic_sample_weight_clamp})")
        else:
            w_train_arr = np.ones(X_train.shape[0], dtype=np.float32)
            print("【sample_weight】未启用或未提供 w_train.npy：使用均匀权重")

        # 验证集统一使用 1.0 权重，保持 val_loss 跨实验可比。
        w_val_arr = np.ones(X_val.shape[0], dtype=np.float32)
        
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
            torch.FloatTensor(w_train_arr)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
            torch.FloatTensor(w_val_arr)
        )
        
        num_workers = int(self.config['training'].get('num_workers', 0))
        pin_memory = True if self.device.type == 'cuda' else False
        if num_workers <= 0:
            print("【DataLoader】 使用单进程加载 (num_workers=0)，避免多进程 worker 清理告警")
        else:
            print(f"【DataLoader】 使用多进程加载 (num_workers={num_workers})")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        num_epochs = self.config['training']['num_epochs']
        
        for epoch in range(num_epochs):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print('='*70)
            
            train_loss = self.train_epoch(train_loader, epoch)
            val_results = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss['total'])
            self.history['train_data_loss'].append(train_loss['data'])
            self.history['train_dir_loss'].append(train_loss['dir'])
            self.history['train_mag_loss'].append(train_loss['mag'])
            self.history['val_loss'].append(val_results['loss'])
            self.history['val_data_loss'].append(val_results['data'])
            self.history['val_dir_loss'].append(val_results['dir'])
            self.history['val_mag_loss'].append(val_results['mag'])
            self.history['val_mae'].append(val_results['mae'])
            self.history['val_rmse'].append(val_results['rmse'])
            self.history['val_wind_mag_error'].append(val_results['wind_mag_error'])
            self.history['val_wind_mag_rmse'].append(val_results['wind_mag_rmse'])
            self.history['val_wind_direction_error'].append(val_results['wind_direction_error'])
            self.history['val_composite_score'].append(val_results['composite_score'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            self.writer.add_scalar('Loss/Total_train', train_loss['total'], epoch)
            self.writer.add_scalar('Loss/Total_val', val_results['loss'], epoch)
            self.writer.add_scalar('Loss/Data_train', train_loss['data'], epoch)
            self.writer.add_scalar('Loss/Data_val', val_results['data'], epoch)
            self.writer.add_scalar('Loss/Direction_train', train_loss['dir'], epoch)
            self.writer.add_scalar('Loss/Direction_val', val_results['dir'], epoch)
            self.writer.add_scalar('Loss/Magnitude_train', train_loss['mag'], epoch)
            self.writer.add_scalar('Loss/Magnitude_val', val_results['mag'], epoch)
            self.writer.add_scalar('Metrics/MAE', val_results['mae'], epoch)
            self.writer.add_scalar('Metrics/RMSE', val_results['rmse'], epoch)
            self.writer.add_scalar('Metrics/WindMagError', val_results['wind_mag_error'], epoch)
            self.writer.add_scalar('Metrics/WindMagRMSE', val_results['wind_mag_rmse'], epoch)
            self.writer.add_scalar('Metrics/WindDirectionError', val_results['wind_direction_error'], epoch)
            self.writer.add_scalar('Metrics/CompositeScore', val_results['composite_score'], epoch)
            self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
            
            print(f"\n【训练损失】 总={train_loss['total']:.4f}, 数据={train_loss['data']:.4f}, 方向={train_loss['dir']:.4f}, 模值={train_loss['mag']:.4f}")
            print(f"【验证损失】 总={val_results['loss']:.4f}, 数据={val_results['data']:.4f}, 方向={val_results['dir']:.4f}, 模值={val_results['mag']:.4f}")
            print(f"【验证指标】 MAE={val_results['mae']:.3f} m/s, RMSE={val_results['rmse']:.3f} m/s, "
                  f"风速大小MAE={val_results['wind_mag_error']:.3f} m/s, 风速大小RMSE={val_results['wind_mag_rmse']:.3f} m/s, "
                  f"风向误差={val_results['wind_direction_error']:.2f}°")
            print(f"【组合选模】 {self.monitor_metric_label}={val_results[self.monitor_metric_name]:.4f}, Composite={val_results['composite_score']:.4f}")
            print(f"【学习率】 {current_lr:.6f}")
            
            self.scheduler.step(val_results[self.monitor_metric_name])
            
            improved_primary = False
            primary_metric_value = val_results[self.monitor_metric_name]
            primary_save_name_map = {
                'loss': 'best_total_loss_model.pth',
                'rmse': 'best_rmse_model.pth',
                'wind_mag_error': 'best_mag_model.pth',
                'wind_mag_rmse': 'best_mag_rmse_model.pth',
                'wind_direction_error': 'best_dir_model.pth',
                'composite_score': 'best_composite_model.pth',
            }
            if primary_metric_value < self.best_monitor_value:
                self.best_monitor_value = primary_metric_value
                self.best_epochs[self.monitor_metric_name] = epoch + 1
                self.patience_counter = 0
                improved_primary = True
                self.save_model('best_model.pth')
                primary_save_name = primary_save_name_map[self.monitor_metric_name]
                if primary_save_name != 'best_model.pth':
                    self.save_model(primary_save_name)
                print(f"✅ 保存主最佳模型 best_model.pth ({self.monitor_metric_label}: {primary_metric_value:.4f})")

            if val_results['loss'] < self.best_val_loss:
                self.best_val_loss = val_results['loss']
                self.best_epochs['loss'] = epoch + 1
                self.save_model('best_total_loss_model.pth')
                print(f"💾 更新 best_total_loss_model.pth (验证损失: {val_results['loss']:.4f})")

            if val_results['rmse'] < self.best_val_rmse:
                self.best_val_rmse = val_results['rmse']
                self.best_epochs['rmse'] = epoch + 1
                self.save_model('best_rmse_model.pth')
                print(f"💾 更新 best_rmse_model.pth (RMSE: {val_results['rmse']:.4f})")

            if val_results['wind_mag_error'] < self.best_val_wind_mag_error:
                self.best_val_wind_mag_error = val_results['wind_mag_error']
                self.best_epochs['wind_mag_error'] = epoch + 1
                self.save_model('best_mag_model.pth')
                print(f"💾 更新 best_mag_model.pth (风速大小MAE: {val_results['wind_mag_error']:.4f})")

            if val_results['wind_mag_rmse'] < self.best_val_wind_mag_rmse:
                self.best_val_wind_mag_rmse = val_results['wind_mag_rmse']
                self.best_epochs['wind_mag_rmse'] = epoch + 1
                self.save_model('best_mag_rmse_model.pth')
                print(f"💾 更新 best_mag_rmse_model.pth (风速大小RMSE: {val_results['wind_mag_rmse']:.4f})")

            if val_results['wind_direction_error'] < self.best_val_direction_error:
                self.best_val_direction_error = val_results['wind_direction_error']
                self.best_epochs['wind_direction_error'] = epoch + 1
                self.save_model('best_dir_model.pth')
                print(f"💾 更新 best_dir_model.pth (风向误差: {val_results['wind_direction_error']:.4f}°)")

            if val_results['composite_score'] < self.best_val_composite_score:
                self.best_val_composite_score = val_results['composite_score']
                self.best_epochs['composite_score'] = epoch + 1
                self.save_model('best_composite_model.pth')
                print(f"💾 更新 best_composite_model.pth (组合分数: {val_results['composite_score']:.4f})")

            if not improved_primary:
                self.patience_counter += 1
                print(f"❌ {self.monitor_metric_label} 未改善 ({self.patience_counter}/{self.early_stopping_patience})")
            
            checkpoint_interval = self.config['training'].get('checkpoint_save_interval', 10)
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth')
            
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n⚠️ 早停触发，训练结束")
                break
        
        self.writer.close()
        self.plot_training_history()
        
        print(f"\n{'='*70}")
        print(f"✅ {self.model_name} 训练完成！")
        print('='*70)
        print(f"主选模指标最佳值 ({self.monitor_metric_label}): {self.best_monitor_value:.4f} @ Epoch {self.best_epochs[self.monitor_metric_name]}")
        print(f"最佳验证损失: {self.best_val_loss:.4f} @ Epoch {self.best_epochs['loss']}")
        print(f"最佳RMSE: {self.best_val_rmse:.4f} @ Epoch {self.best_epochs['rmse']}")
        print(f"最佳风速大小RMSE: {self.best_val_wind_mag_rmse:.4f} @ Epoch {self.best_epochs['wind_mag_rmse']}")
        print(f"最佳风向误差: {self.best_val_direction_error:.4f}° @ Epoch {self.best_epochs['wind_direction_error']}")
        print(f"模型保存路径: {self.model_save_dir}")

    def save_model(self, filename):
        """保存模型"""
        save_path = os.path.join(self.model_save_dir, filename)
        
        torch.save({
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_monitor_value': self.best_monitor_value,
            'best_val_loss': self.best_val_loss,
            'best_val_rmse': self.best_val_rmse,
            'best_val_wind_mag_error': self.best_val_wind_mag_error,
            'best_val_wind_mag_rmse': self.best_val_wind_mag_rmse,
            'best_val_direction_error': self.best_val_direction_error,
            'best_val_composite_score': self.best_val_composite_score,
            'best_epochs': self.best_epochs,
            'selection_metric': self.monitor_metric_name,
            'config': self.config,
            'history': self.history,
            'model_type': self.model.get_model_info()['model_type'],
            'rnn_type': self.model.get_model_info()['rnn_type'],
        }, save_path)

    def plot_training_history(self):
        """绘制训练历史"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        axes[0, 0].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Supervised Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Total Supervised Loss')
        
        axes[0, 1].plot(self.history['train_data_loss'], label='Data', linewidth=2)
        axes[0, 1].plot(self.history['train_dir_loss'], label='Direction', linewidth=2)
        axes[0, 1].plot(self.history['train_mag_loss'], label='Magnitude', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Train Loss Components')
        
        axes[1, 0].plot(self.history['val_mae'], label='MAE', linewidth=2, color='blue')
        axes[1, 0].plot(self.history['val_rmse'], label='RMSE', linewidth=2, color='red')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Error (m/s)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Validation Vector Error')
        
        axes[1, 1].plot(self.history['val_wind_mag_error'], label='Mag MAE', linewidth=2, color='green')
        axes[1, 1].plot(self.history['val_wind_mag_rmse'], label='Mag RMSE', linewidth=2, color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Magnitude Error (m/s)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Wind Magnitude Error')
        
        axes[2, 0].plot(self.history['val_wind_direction_error'], label='Direction Error', linewidth=2, color='orange')
        axes[2, 0].plot(self.history['val_composite_score'], label='Composite', linewidth=2, color='black')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Score')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_title('Direction / Composite Selection')
        
        axes[2, 1].plot(self.history['learning_rate'], color='brown', linewidth=2)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Learning Rate')
        axes[2, 1].set_yscale('log')
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_title('Learning Rate Schedule')
        
        plt.suptitle(f'{self.model_name} Training History (Fair Baseline)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.model_save_dir, 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 训练曲线已保存: {save_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vanilla GRU 训练脚本（支持独立配置文件）")
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="可选配置文件路径，默认使用 config/config.yaml"
    )
    args = parser.parse_args()

    print("="*70)
    print(" Vanilla GRU 训练模块 (Fair Baseline)")
    print(" 特点: 方向/模值监督 + 组合选模，无物理约束")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    config_path = args.config_path or os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    data_dir_cfg = (
        cfg.get('experiment', {}).get('processed_dir')
        or cfg.get('data', {}).get('processed_dir', '../data/dataset_new_processed')
    )
    if os.path.isabs(data_dir_cfg):
        data_dir = data_dir_cfg
    else:
        data_dir = os.path.normpath(os.path.join(project_root, data_dir_cfg.lstrip('../')))

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"未找到预处理数据目录: {data_dir}，请先运行 src/1_preprocessing_data.py")

    print(f"\n【配置文件】 {config_path}")
    print(f"【数据目录】 {data_dir}")
    print("\n正在加载数据...")
    
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    w_train_path = os.path.join(data_dir, 'w_train.npy')
    w_val_path = os.path.join(data_dir, 'w_val.npy')
    w_train = np.load(w_train_path) if os.path.exists(w_train_path) else None
    w_val = np.load(w_val_path) if os.path.exists(w_val_path) else None

    print(f"  训练集: X={X_train.shape}, y={y_train.shape}")
    print(f"  验证集: X={X_val.shape}, y={y_val.shape}")
    if w_train is not None:
        n_dyn = int(np.sum(w_train > 1.0 + 1e-6))
        print(f"  sample_weight: w_train={w_train.shape} mean={w_train.mean():.3f} "
              f"max={w_train.max():.3f} 动态样本占比={100.0*n_dyn/max(len(w_train),1):.1f}%")
    else:
        print("  sample_weight: w_train.npy 不存在 → 使用均匀权重")

    trainer = VanillaGRUTrainer(config_path=config_path)
    trainer.train(X_train, y_train, X_val, y_val, w_train=w_train, w_val=w_val)
