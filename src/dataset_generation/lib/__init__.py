"""PIRNN-AKF 数据采集库代码包。

按职责分层：
  - config/      运行期与数据集配置加载
  - obs/         结构化日志与失败记录
  - errors       领域异常层级
  - infra/       PX4 / mavsdk_server / JSBSim 进程与资源生命周期 (Phase 2)
  - controllers/ MAVSDK 飞控封装 (Phase 2)
  - recorder/    DataLogger 与子订阅 (Phase 2)
  - planner/     纯函数：schedule / wind / segment 配置生成 (Phase 1)
  - pipeline/    Pipeline / Stage 编排 (Phase 4)
  - validation/  质量门 (Phase 3)
  - recovery/    缺失检测与补采调度 (Phase 5)

scripts/ 入口点保留向后兼容 shim，外部调用方无须修改。
"""

__all__ = []  # noqa: F841 - 占位，子包按需 re-export
