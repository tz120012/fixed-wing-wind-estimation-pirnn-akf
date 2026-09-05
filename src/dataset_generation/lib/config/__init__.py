"""配置加载（runtime.yaml 与 dataset_config.json）。"""

from .runtime import RuntimeConfig, load_runtime_config

__all__ = ["RuntimeConfig", "load_runtime_config"]
