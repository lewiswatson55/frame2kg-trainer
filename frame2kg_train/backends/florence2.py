from __future__ import annotations
from typing import Any, Dict, List

from frame2kg_train.backends.base import VLMBackend, BackendArtifacts


class Florence2Backend(VLMBackend):
    name: str = "florence2"

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        raise NotImplementedError("Implement Florence2 backend loading")

    def default_lora_target_modules(self) -> List[str]:
        return []

    def generate_text(self, artifacts: BackendArtifacts, image, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError
