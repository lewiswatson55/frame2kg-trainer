from __future__ import annotations

from importlib import import_module

from .base import BackendArtifacts, VLMBackend


_BACKEND_EXPORTS = {
    "FastVLMBackend": ("frame2kg_train.backends.fastvlm", "FastVLMBackend"),
    "Gemma4VLBackend": ("frame2kg_train.backends.gemma4_vl", "Gemma4VLBackend"),
    "InternVL35Backend": ("frame2kg_train.backends.internvl35_vl", "InternVL35Backend"),
    "LFM25VLBackend": ("frame2kg_train.backends.lfm25_vl", "LFM25VLBackend"),
    "Llama32VisionBackend": ("frame2kg_train.backends.llama32_vision", "Llama32VisionBackend"),
    "Qwen25VLBackend": ("frame2kg_train.backends.qwen25_vl", "Qwen25VLBackend"),
    "Qwen3VLBackend": ("frame2kg_train.backends.qwen3_vl", "Qwen3VLBackend"),
    "SmolVLM2Backend": ("frame2kg_train.backends.smolvlm2", "SmolVLM2Backend"),
}

__all__ = [
    "VLMBackend",
    "BackendArtifacts",
    *_BACKEND_EXPORTS,
]


def __getattr__(name: str):
    if name not in _BACKEND_EXPORTS:
        raise AttributeError(name)
    module_name, class_name = _BACKEND_EXPORTS[name]
    value = getattr(import_module(module_name), class_name)
    globals()[name] = value
    return value
