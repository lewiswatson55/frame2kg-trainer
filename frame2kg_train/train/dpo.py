from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import Trainer


class DPOTrainer(Trainer):
    def __init__(self, *args, ref_model: Any, beta: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def _sequence_logps(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels.ne(-100)
        safe_labels = torch.where(mask, shift_labels, torch.zeros_like(shift_labels))
        token_logps = F.log_softmax(shift_logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logps = token_logps * mask
        return token_logps.sum(dim=-1)

    def _batch_logps(self, model: Any, prefix: str, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        model_inputs = {
            "input_ids": inputs[f"{prefix}_input_ids"],
            "attention_mask": inputs[f"{prefix}_attention_mask"],
            "labels": inputs[f"{prefix}_labels"],
        }
        if f"{prefix}_pixel_values" in inputs:
            model_inputs["pixel_values"] = inputs[f"{prefix}_pixel_values"]
        if f"{prefix}_image_grid_thw" in inputs:
            model_inputs["image_grid_thw"] = inputs[f"{prefix}_image_grid_thw"]
        outputs = model(**model_inputs)
        return self._sequence_logps(outputs.logits, model_inputs["labels"])

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        pi_chosen = self._batch_logps(model, "chosen", inputs)
        pi_rejected = self._batch_logps(model, "rejected", inputs)

        with torch.no_grad():
            ref_chosen = self._batch_logps(self.ref_model, "chosen", inputs)
            ref_rejected = self._batch_logps(self.ref_model, "rejected", inputs)

        pi_logratios = pi_chosen - pi_rejected
        ref_logratios = ref_chosen - ref_rejected
        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()

        if return_outputs:
            outputs = {
                "pi_chosen": pi_chosen.detach(),
                "pi_rejected": pi_rejected.detach(),
                "ref_chosen": ref_chosen.detach(),
                "ref_rejected": ref_rejected.detach(),
            }
            return loss, outputs
        return loss