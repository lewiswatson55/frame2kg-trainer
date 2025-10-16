from __future__ import annotations
from typing import Any, Dict, List

import torch
from frame2kg.backends.base import VLMBackend, BackendArtifacts


class BLIP2Backend(VLMBackend):
    name: str = "blip2"

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        # Placeholder: wire up your chosen BLIP‑2 variant here
        raise NotImplementedError("Implement BLIP‑2 backend loading")

    def default_lora_target_modules(self) -> List[str]:
        return []

    def generate_text(self, artifacts: BackendArtifacts, image, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError
