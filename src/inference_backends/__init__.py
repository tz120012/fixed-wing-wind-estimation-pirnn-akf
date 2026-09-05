def create_backend(config):
	backend_name = str(config.get('deployment', {}).get('backend', 'torch')).lower()
	if backend_name == 'torch':
		from .torch_backend import TorchBackend
		return TorchBackend(config)
	if backend_name == 'ascend':
		from .ascend_backend import AscendBackend
		return AscendBackend(config)
	if backend_name == 'torchscript':
		from .torchscript_backend import TorchScriptBackend
		return TorchScriptBackend(config)
	if backend_name in ('torch_compile', 'compile'):
		from .torch_compile_backend import TorchCompileBackend
		return TorchCompileBackend(config)
	if backend_name in ('onnx', 'onnxruntime'):
		from .onnx_backend import OnnxBackend
		return OnnxBackend(config)
	raise ValueError(f"不支持的推理后端: {backend_name}")


def __getattr__(name):
	if name == 'OnnxBackend':
		from .onnx_backend import OnnxBackend
		return OnnxBackend
	if name == 'TorchBackend':
		from .torch_backend import TorchBackend
		return TorchBackend
	if name == 'AscendBackend':
		from .ascend_backend import AscendBackend
		return AscendBackend
	if name == 'TorchScriptBackend':
		from .torchscript_backend import TorchScriptBackend
		return TorchScriptBackend
	if name == 'TorchCompileBackend':
		from .torch_compile_backend import TorchCompileBackend
		return TorchCompileBackend
	raise AttributeError(name)


__all__ = [
	'TorchBackend', 'AscendBackend', 'TorchScriptBackend',
	'TorchCompileBackend', 'OnnxBackend', 'create_backend'
]
