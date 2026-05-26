from __future__ import annotations

from importlib import import_module
from typing import Dict, Tuple, Type

from frame2kg_train.backends.base import VLMBackend


_BACKEND_CLASSES: Dict[str, Tuple[str, str]] = {
    "fastvlm": ("frame2kg_train.backends.fastvlm", "FastVLMBackend"),
    "gemma4_vl": ("frame2kg_train.backends.gemma4_vl", "Gemma4VLBackend"),
    "internvl35_vl": ("frame2kg_train.backends.internvl35_vl", "InternVL35Backend"),
    "lfm25_vl": ("frame2kg_train.backends.lfm25_vl", "LFM25VLBackend"),
    "llama32_vision": ("frame2kg_train.backends.llama32_vision", "Llama32VisionBackend"),
    "qwen25_vl": ("frame2kg_train.backends.qwen25_vl", "Qwen25VLBackend"),
    "qwen3_vl": ("frame2kg_train.backends.qwen3_vl", "Qwen3VLBackend"),
    "qwen35_vl": ("frame2kg_train.backends.qwen25_vl", "Qwen25VLBackend"),
    "smolvlm2": ("frame2kg_train.backends.smolvlm2", "SmolVLM2Backend"),
}
_REGISTRY: Dict[str, VLMBackend] = {}


def get_backend(name: str) -> VLMBackend:
    if name not in _BACKEND_CLASSES:
        raise KeyError(f"Unknown backend: {name}")
    if name not in _REGISTRY:
        module_name, class_name = _BACKEND_CLASSES[name]
        backend_cls: Type[VLMBackend] = getattr(import_module(module_name), class_name)
        _REGISTRY[name] = backend_cls()
    return _REGISTRY[name]
