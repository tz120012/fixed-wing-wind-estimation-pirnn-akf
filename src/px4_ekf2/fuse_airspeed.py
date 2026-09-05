# -----------------------------------------------------------------------------
# 由 SymForce 自动生成的 Python EKF 测量更新函数，实现空速观测的雅可比、卡尔曼增益与残差计算。
# -----------------------------------------------------------------------------
import math
import typing as T

import numpy


def fuse_airspeed(v_local, state, P, airspeed, R, epsilon):
    # type: (numpy.ndarray, numpy.ndarray, numpy.ndarray, float, float, float) -> T.Tuple[numpy.ndarray, numpy.ndarray, float, float]
    """
    空速融合函数 - 用于扩展卡尔曼滤波器(EKF)中融合空速传感器测量值

    此函数实现了风速估计器的测量更新步骤，将空速传感器的测量值融合到状态估计中。
    使用线性化的观测模型计算卡尔曼增益，并更新风速状态估计。

    符号函数: fuse_airspeed

    参数:
        v_local: Matrix31 - 局部坐标系下的速度向量 [vx, vy, vz] (m/s)
        state: Matrix31 - 当前状态向量 [wind_x, wind_y, wind_scale]
                         wind_x: X轴风速 (m/s)
                         wind_y: Y轴风速 (m/s)
                         wind_scale: 风速缩放因子
        P: Matrix33 - 状态协方差矩阵 (3x3)
        airspeed: Scalar - 测量的空速值 (m/s)
        R: Scalar - 测量噪声方差
        epsilon: Scalar - 数值稳定性的小常数，避免除零

    输出:
        H: Matrix13 - 观测矩阵 (1x3)，测量模型相对于状态的雅可比矩阵
        K: Matrix31 - 卡尔曼增益向量 (3x1)
        innov_var: Scalar - 新息方差
        innov: Scalar - 新息(innovation)，即测量残差
    """

    # 总操作数: 56

    # ========================================================================
    # 输入数组形状验证
    # ========================================================================

    # 确保 v_local 是列向量 (3, 1)
    if v_local.shape == (3,):
        v_local = v_local.reshape((3, 1))
    elif v_local.shape != (3, 1):
        raise IndexError(
            "v_local is expected to have shape (3, 1) or (3,); instead had shape {}".format(
                v_local.shape
            )
        )

    # 确保 state 是列向量 (3, 1)
    if state.shape == (3,):
        state = state.reshape((3, 1))
    elif state.shape != (3, 1):
        raise IndexError(
            "state is expected to have shape (3, 1) or (3,); instead had shape {}".format(
                state.shape
            )
        )

    # ========================================================================
    # 中间项计算 (11个中间变量)
    # ========================================================================

    # 计算相对风速分量 (飞行器速度减去风速)
    _tmp0 = -state[0, 0] + v_local[0, 0]  # 相对风速X分量: vx - wind_x
    _tmp1 = -state[1, 0] + v_local[1, 0]  # 相对风速Y分量: vy - wind_y

    # 计算相对风速的模（空速的预测值）
    # sqrt((vx-wind_x)^2 + (vy-wind_y)^2 + vz^2 + epsilon)
    _tmp2 = math.sqrt(_tmp0**2 + _tmp1**2 + epsilon + v_local[2, 0] ** 2)

    # 计算缩放因子除以空速预测值（用于雅可比矩阵计算）
    _tmp3 = state[2, 0] / _tmp2  # wind_scale / airspeed_predicted

    # 观测矩阵 H 的各个分量（负值）
    _tmp4 = _tmp0 * _tmp3  # ∂h/∂wind_x 的计算中间项
    _tmp5 = _tmp1 * _tmp3  # ∂h/∂wind_y 的计算中间项

    # 计算 P * H^T 的各个分量（用于卡尔曼增益计算）
    _tmp6 = -P[0, 0] * _tmp4  # P[0,0] * H[0]
    _tmp7 = -P[1, 1] * _tmp5  # P[1,1] * H[1]
    _tmp8 = P[2, 2] * _tmp2   # P[2,2] * H[2]

    # 计算新息协方差 S = H * P * H^T + R
    # 这是测量预测不确定性的度量
    _tmp9 = (
        R  # 测量噪声方差
        + _tmp2 * (-P[0, 2] * _tmp4 - P[1, 2] * _tmp5 + _tmp8)  # H[2] * (P * H^T)
        - _tmp4 * (-P[1, 0] * _tmp5 + P[2, 0] * _tmp2 + _tmp6)  # H[0] * (P * H^T)
        - _tmp5 * (-P[0, 1] * _tmp4 + P[2, 1] * _tmp2 + _tmp7)  # H[1] * (P * H^T)
    )

    # 计算新息协方差的逆，使用 max 确保数值稳定性
    _tmp10 = 1 / max(_tmp9, epsilon)

    # ========================================================================
    # 输出项
    # ========================================================================

    # 观测矩阵 H (1x3): 测量模型相对于状态的偏导数
    # H = ∂h/∂state，其中 h 是观测模型
    _H = numpy.zeros(3)
    _H[0] = -_tmp4  # ∂(airspeed)/∂(wind_x)
    _H[1] = -_tmp5  # ∂(airspeed)/∂(wind_y)
    _H[2] = _tmp2   # ∂(airspeed)/∂(wind_scale)

    # 卡尔曼增益 K (3x1): K = P * H^T * S^(-1)
    # 决定测量对状态更新的权重
    _K = numpy.zeros(3)
    _K[0] = _tmp10 * (-P[0, 1] * _tmp5 + P[0, 2] * _tmp2 + _tmp6)  # K[0] = (P * H^T)[0] / S
    _K[1] = _tmp10 * (-P[1, 0] * _tmp4 + P[1, 2] * _tmp2 + _tmp7)  # K[1] = (P * H^T)[1] / S
    _K[2] = _tmp10 * (-P[2, 0] * _tmp4 - P[2, 1] * _tmp5 + _tmp8)  # K[2] = (P * H^T)[2] / S

    # 新息方差: 测量预测的不确定性
    _innov_var = _tmp9

    # 新息(innovation): 测量值与预测值的差
    # innov = y - h(x) = airspeed_measured - airspeed_predicted
    _innov = -_tmp2 * state[2, 0] + airspeed

    return _H, _K, _innov_var, _innov
