from __future__ import annotations
from typing import Dict

from frame2kg_train.backends.base import VLMBackend
from frame2kg_train.backends.qwen25_vl import Qwen25VLBackend
from frame2kg_train.backends.blip2 import BLIP2Backend
from frame2kg_train.backends.florence2 import Florence2Backend
from frame2kg_train.backends.smolvlm2 import SmolVLM2Backend


_REGISTRY: Dict[str, VLMBackend] = {
    "qwen25_vl": Qwen25VLBackend(),
    "qwen3_vl": Qwen25VLBackend(),
    "qwen35_vl": Qwen25VLBackend(),
    "smolvlm2": SmolVLM2Backend(),
    "blip2": BLIP2Backend(),
    "florence2": Florence2Backend(),
}


def get_backend(name: str) -> VLMBackend:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown backend: {name}")
    return _REGISTRY[name]
