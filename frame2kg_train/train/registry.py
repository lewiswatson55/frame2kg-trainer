from __future__ import annotations
from typing import Dict

from frame2kg_train.backends.base import VLMBackend
from frame2kg_train.backends.gemma4_vl import Gemma4VLBackend
from frame2kg_train.backends.qwen25_vl import Qwen25VLBackend
from frame2kg_train.backends.qwen3_vl import Qwen3VLBackend
from frame2kg_train.backends.smolvlm2 import SmolVLM2Backend


_REGISTRY: Dict[str, VLMBackend] = {
    "gemma4_vl": Gemma4VLBackend(),
    "qwen25_vl": Qwen25VLBackend(),
    "qwen3_vl": Qwen3VLBackend(),
    "qwen35_vl": Qwen25VLBackend(),
    "smolvlm2": SmolVLM2Backend(),
}


def get_backend(name: str) -> VLMBackend:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend: {name}")
    return _REGISTRY[name]
