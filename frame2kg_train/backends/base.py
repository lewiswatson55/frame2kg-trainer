from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Protocol

import torch
from datasets import Dataset


class Collator(Protocol):
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        ...


@dataclass
class BackendArtifacts:
    model: Any
    tokenizer: Any
    processor: Any | None
    collator: Collator


class VLMBackend(Protocol):
    """
    Backend interface for a Vision–Language Model with:
    - model/processor/tokeniser loading (incl. quantisation if desired),
    - collator tailored to that model family,
    - default LoRA target modules (if applicable),
    - ability to generate for per‑sample eval in callbacks.
    """

    name: str

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        ...

    def default_lora_target_modules(self) -> List[str]:
        return []

    @torch.inference_mode()
    def generate_text(self, artifacts: BackendArtifacts, image, prompt: str, max_new_tokens: int) -> str:
        ...
