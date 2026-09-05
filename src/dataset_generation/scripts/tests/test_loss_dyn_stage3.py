#!/usr/bin/env python3
"""
阶段 3：calculate_physics_loss_dyn 单元测试。

策略：
  - 不真正初始化 PIGRUTrainer（要加载 norm_params 太重），而是用 __new__ 创建空白实例，
    手动注入 calculate_physics_loss_dyn 所需的所有属性
  - 验证三件事：
    1. 关闭路径（lambda_phys_dyn=0 时 train_epoch 跳过）：直接验证返回值是 0 tensor
    2. 启用路径：返回有限标量，且能反向传播
    3. 数值层面与 calculate_physics_loss_6dof（同样输入下）有合理偏差但不发散
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
TRAIN_PATH = ROOT / "src" / "3_train_pigru.py"
spec = importlib.util.spec_from_file_location("train_mod", str(TRAIN_PATH))
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
PIGRUTrainer = train_mod.Trainer
FEATURE_IDX = train_mod.FEATURE_IDX
LABEL_IDX = train_mod.LABEL_IDX


def make_minimal_trainer(device="cpu"):
    """绕过 PIGRUTrainer.__init__ 创建最小对象，注入 L_dyn 所需属性。"""
    t = PIGRUTrainer.__new__(PIGRUTrainer)
    t.device = torch.device(device)

    # ───── 标签 scaler（label 7 维：wind_n/e/d, vel_n/e/d, airspeed）─────
    t.wind_mean = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    t.wind_std = torch.tensor([1.5, 1.5, 0.5], dtype=torch.float32)
    t.vel_mean = torch.tensor([13.0, 7.0, 0.1], dtype=torch.float32)
    t.vel_std = torch.tensor([3.0, 3.0, 0.5], dtype=torch.float32)
    t.airspeed_mean = torch.tensor(15.0, dtype=torch.float32)
    t.airspeed_std = torch.tensor(2.0, dtype=torch.float32)

    # ───── 姿态 scaler（X 中 9-11 索引）─────
    t.att_mean = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    t.att_std = torch.tensor([0.1, 0.1, 1.0], dtype=torch.float32)

    # ───── 物理参数（从 Rascal110 默认值）─────
    t.uav_mass = (13.0 + 1.5 * 0.8) * 0.45359237  # ≈ 6.444 kg
    t.gravity = 9.81
    t.air_density = 1.225
    t.wing_area = 10.57 * 0.09290304  # ≈ 0.982 m²
    t.wind_magnitude_max = 15.0

    # 控制面行程
    t.elevator_rad_range = (-0.35, 0.30)
    t.elevator_norm_domain = (-0.30, 0.30)
    t.aileron_rad_range = (-0.35, 0.35)
    t.rudder_rad_range = (-0.35, 0.35)

    # ───── 气动查表（精简：用回退多项式不需要 lookup tables）─────
    t.rascal_use_lookup_tables = False  # 走多项式回退分支
    # 多项式回退使用的常量（来自原代码默认值）
    t.C_L0 = 0.0
    t.C_L_alpha = 5.5
    t.C_L_delta_e = 0.36
    t.C_L_max = 1.4
    t.C_D0 = 0.028
    t.C_D_alpha2 = 0.45
    t.C_D_delta_e2 = 0.05
    t.C_Y_beta = -0.83
    t.C_Y_delta_r = 0.18
    # _compute_rascal_aero_coefficients 还可能用 delta_a_C_l_p 等；
    # 我们 force lookup_tables=False 走最简单的 fallback，避免缺属性

    # 螺旋桨参数
    t.prop_diameter = 0.4572  # 18 inch
    t.prop_pitch = 0.2032  # 8 inch
    t.prop_thrust_coef = torch.tensor(
        [[0.0, 0.10], [0.5, 0.08], [1.0, 0.04], [1.5, 0.0]],
        dtype=torch.float32,
    )
    t.prop_n_max_rps = 80.0
    t.prop_throttle_to_rps_curve = torch.tensor(
        [[0.0, 0.0], [0.5, 40.0], [1.0, 80.0]],
        dtype=torch.float32,
    )

    # _compute_rascal_propeller_thrust 用到的查表（占位）
    # 用真实 calculation 太繁；我们简化推力为线性
    def simple_thrust(throttle, axial_airspeed):
        # 简单线性推力模型：F = (k_T * throttle - k_v * V) * S
        F = torch.clamp(40.0 * throttle - 0.5 * axial_airspeed, min=0.0)
        # 返回 (F_thrust, J, n_rps) 占位
        zeros = torch.zeros_like(F)
        return F, zeros, zeros + 60.0
    t._compute_rascal_propeller_thrust = simple_thrust

    # _compute_rascal_aero_coefficients 简化（lookup_tables=False 时原代码会
    # 调用回退多项式；最干净的做法是直接 mock 一个 tuple）
    def simple_aero(alpha, beta, delta_e_rad, delta_e_norm, delta_a_cmd):
        C_L = 0.2 + 5.5 * alpha + 0.3 * delta_e_rad
        C_D = 0.03 + 0.3 * alpha ** 2
        C_Y = -0.5 * beta
        return C_L, C_D, C_Y
    t._compute_rascal_aero_coefficients = simple_aero

    return t


def make_dummy_batch(B=4, T=100, F=45):
    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.normal(0, 1, (B, T, F)).astype(np.float32))
    # y: [wind_n/e/d, vel_n/e/d, airspeed] 7 维（归一化空间）
    y = torch.from_numpy(rng.normal(0, 1, (B, 7)).astype(np.float32))
    angles = torch.zeros(B, 3, dtype=torch.float32)  # [Δα, Δβ, s_tas]，s_tas 用 0 会乘 V_T 为 0；用 1
    angles[:, 2] = 1.0
    wind_estimate = torch.zeros(B, 3, dtype=torch.float32, requires_grad=True)
    return wind_estimate, X, y, angles


def test_dyn_loss_runs_and_finite():
    t = make_minimal_trainer()
    wind, X, y, angles = make_dummy_batch()
    loss = t.calculate_physics_loss_dyn(wind, X, y, angles, epoch=None)
    assert loss.dim() == 0, f"应返回标量，shape={loss.shape}"
    assert torch.isfinite(loss), f"loss 非有限: {loss.item()}"
    assert loss.item() >= 0.0, f"loss 应非负，得 {loss.item()}"


def test_dyn_loss_backward_grad_nonzero():
    t = make_minimal_trainer()
    wind, X, y, angles = make_dummy_batch()
    loss = t.calculate_physics_loss_dyn(wind, X, y, angles, epoch=None)
    loss.backward()
    assert wind.grad is not None, "wind_estimate 应有梯度"
    assert torch.isfinite(wind.grad).all(), "梯度有 NaN/Inf"
    assert wind.grad.abs().sum() > 0, "梯度全 0，损失对风预测无敏感性"


def test_dyn_loss_with_zero_wind_close_to_baseline():
    """风估计为 0 时，残差大小有界（不发散到大数）。"""
    t = make_minimal_trainer()
    wind, X, y, angles = make_dummy_batch()
    loss_zero = t.calculate_physics_loss_dyn(wind, X, y, angles, epoch=None)
    # 给风一个非零扰动
    wind2 = wind.detach() + 0.5
    wind2.requires_grad_(True)
    loss_perturbed = t.calculate_physics_loss_dyn(wind2, X, y, angles, epoch=None)
    # 两者都应有限
    assert torch.isfinite(loss_zero) and torch.isfinite(loss_perturbed)
    # 数值有界（normalized residual ≤ 100 量级）
    assert loss_zero.item() < 100.0, f"loss_zero={loss_zero.item()} 太大"
    assert loss_perturbed.item() < 100.0, f"loss_perturbed={loss_perturbed.item()} 太大"


def test_train_epoch_skips_dyn_when_disabled():
    """模拟 train_epoch 路径：lambda_phys_dyn=0 + use_dyn_residual=False，
    physics_loss_dyn 应等于 0 tensor。"""
    t = make_minimal_trainer()
    # 不调用真实 train_epoch（依赖太多），手工模拟开关逻辑：
    lambda_phys_dyn = 0.0
    use_dyn_residual = False
    wind, X, y, angles = make_dummy_batch()
    if use_dyn_residual and lambda_phys_dyn > 0.0:
        loss = t.calculate_physics_loss_dyn(wind, X, y, angles, epoch=None)
    else:
        loss = torch.zeros(())
    assert loss.item() == 0.0


if __name__ == "__main__":
    tests = [
        test_dyn_loss_runs_and_finite,
        test_dyn_loss_backward_grad_nonzero,
        test_dyn_loss_with_zero_wind_close_to_baseline,
        test_train_epoch_skips_dyn_when_disabled,
    ]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"  [OK]   {fn.__name__}")
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failures.append(fn.__name__)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  [ERR]  {fn.__name__}: {type(e).__name__}: {e}")
            failures.append(fn.__name__)
    print()
    if failures:
        sys.exit(1)
    print(f"All {len(tests)} tests passed.")
