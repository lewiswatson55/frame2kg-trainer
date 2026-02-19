from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from PIL import Image
from transformers import Qwen2_5_VLProcessor

SYSTEM_PROMPT = (
    'You are a VLM that outputs ONLY a single, strict JSON object with exactly the keys "nodes" and "edges". '
    'Use valid JSON: double quotes for all keys and string values, no single quotes, no trailing commas. '
    'Schema — "nodes": [{"id":"str","label":"str","location":"x1,y1,x2,y2,confidence","attributes":{...}}], '
    '"edges":[{"predicate":"str","source":"node.id","target":"node.id"}]. '
    'Output the JSON object only - no code fences, no role tags, no prefixes/suffixes, no prose. '
    'The first character must be "{", and the last must be "}".'
)

def _to_json_text(x):
    import json
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))

@dataclass
class QwenVLDataCollator:
    proc: Qwen2_5_VLProcessor

    def __post_init__(self):
        self.pad_id = self.proc.tokenizer.pad_token_id
        special_tokens = [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
        ]
        self.special_ids = set()
        for tok in special_tokens:
            tid = self.proc.tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid != self.proc.tokenizer.unk_token_id:
                self.special_ids.add(tid)

    def _to_pil(self, x):
        if isinstance(x, str):
            return Image.open(x).convert("RGB")
        return x

    def __call__(self, batch: List[Dict[str, Any]]):
        images = [self._to_pil(b["image"]) for b in batch]
        gold_graphs = [b["graph"] for b in batch]
        mode = batch[0].get("mode", "train")

        chats = []
        for img, g in zip(images, gold_graphs):
            chats.append([
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Extract data in JSON."}]},
                {"role": "assistant", "content": [{"type": "text", "text": _to_json_text(g)}]},
            ])

        full_texts = [self.proc.apply_chat_template(c, tokenize=False) for c in chats]
        prompts = [self.proc.apply_chat_template(c[:2], tokenize=False, add_generation_prompt=True) for c in chats]

        if mode == "train":
            inputs = self.proc(text=full_texts, images=[c[1]["content"][0]["image"] for c in chats], return_tensors="pt", padding=True)
            prompt_inputs = self.proc(text=prompts, images=[c[1]["content"][0]["image"] for c in chats], return_tensors="pt", padding=True)

            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            prompt_lens = (prompt_inputs["input_ids"] != self.pad_id).sum(dim=1)
            for i, cutoff in enumerate(prompt_lens.tolist()):
                labels[i, :cutoff] = -100
            labels[input_ids == self.pad_id] = -100
            if self.special_ids:
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for sid in self.special_ids:
                    mask |= input_ids == sid
                labels[mask] = -100
            inputs["labels"] = labels
            return inputs
        else:
            prompt_inputs = self.proc(text=prompts, images=[c[1]["content"][0]["image"] for c in chats], return_tensors="pt", padding=True)
            # Dummy labels to satisfy Trainer signature; metrics ignore these
            prompt_inputs["labels"] = torch.full_like(prompt_inputs["input_ids"], -100)
            return prompt_inputs


@dataclass
class SmolVLMDataCollator:
    proc: Any

    def __post_init__(self):
        tok = self.proc.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        self.special_ids = set(getattr(tok, "all_special_ids", []) or [])
        special_tokens = [
            "<image>",
            "<video>",
            "<end_of_utterance>",
            "<fake_token_around_image>",
            "<global-img>",
        ]
        for token in special_tokens:
            tid = tok.convert_tokens_to_ids(token)
            if tid is not None and tid != tok.unk_token_id:
                self.special_ids.add(tid)

    def _to_pil(self, x):
        if isinstance(x, str):
            return Image.open(x).convert("RGB")
        return x

    def __call__(self, batch: List[Dict[str, Any]]):
        images = [self._to_pil(b["image"]) for b in batch]
        batched_images = [[img] for img in images]
        gold_graphs = [b["graph"] for b in batch]
        mode = batch[0].get("mode", "train")

        chats = []
        for img, g in zip(images, gold_graphs):
            chats.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": "Extract data in JSON."},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": _to_json_text(g)}]},
                ]
            )

        full_texts = [self.proc.apply_chat_template(c, tokenize=False) for c in chats]
        prompts = [self.proc.apply_chat_template(c[:2], tokenize=False, add_generation_prompt=True) for c in chats]

        if mode == "train":
            inputs = self.proc(text=full_texts, images=batched_images, return_tensors="pt", padding=True)
            prompt_inputs = self.proc(text=prompts, images=batched_images, return_tensors="pt", padding=True)

            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            prompt_lens = (prompt_inputs["input_ids"] != self.pad_id).sum(dim=1)
            for i, cutoff in enumerate(prompt_lens.tolist()):
                labels[i, :cutoff] = -100
            labels[input_ids == self.pad_id] = -100
            if self.special_ids:
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for sid in self.special_ids:
                    mask |= input_ids == sid
                labels[mask] = -100
            inputs["labels"] = labels
            return inputs

        prompt_inputs = self.proc(text=prompts, images=batched_images, return_tensors="pt", padding=True)
        # Dummy labels to satisfy Trainer signature; metrics ignore these
        prompt_inputs["labels"] = torch.full_like(prompt_inputs["input_ids"], -100)
        return prompt_inputs

@dataclass
class QwenVLDPODataCollator:
    proc: Qwen2_5_VLProcessor

    def __post_init__(self):
        self.pad_id = self.proc.tokenizer.pad_token_id

    def _to_pil(self, x):
        if isinstance(x, str):
            return Image.open(x).convert("RGB")
        return x

    def _build_dialogue(self, image, user_prompt: str, assistant_text: str):
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": _to_json_text(assistant_text)}]},
        ]

    def _tokenize_with_prompt_mask(self, dialogs):
        full_texts = [self.proc.apply_chat_template(c, tokenize=False) for c in dialogs]
        prompts = [self.proc.apply_chat_template(c[:2], tokenize=False, add_generation_prompt=True) for c in dialogs]
        images = [c[1]["content"][0]["image"] for c in dialogs]

        inputs = self.proc(text=full_texts, images=images, return_tensors="pt", padding=True)
        prompt_inputs = self.proc(text=prompts, images=images, return_tensors="pt", padding=True)

        labels = inputs["input_ids"].clone()
        prompt_lens = (prompt_inputs["input_ids"] != self.pad_id).sum(dim=1)
        for i, cutoff in enumerate(prompt_lens.tolist()):
            labels[i, :cutoff] = -100
        labels[inputs["input_ids"] == self.pad_id] = -100
        inputs["labels"] = labels
        return inputs

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        images = [self._to_pil(b["image"]) for b in batch]
        prompts = [b.get("prompt", "Extract data in JSON.") for b in batch]

        chosen_dialogs = [self._build_dialogue(img, prompt, b["chosen"]) for img, prompt, b in zip(images, prompts, batch)]
        rejected_dialogs = [self._build_dialogue(img, prompt, b["rejected"]) for img, prompt, b in zip(images, prompts, batch)]

        chosen = self._tokenize_with_prompt_mask(chosen_dialogs)
        rejected = self._tokenize_with_prompt_mask(rejected_dialogs)

        out: Dict[str, torch.Tensor] = {}
        for key, value in chosen.items():
            out[f"chosen_{key}"] = value
        for key, value in rejected.items():
            out[f"rejected_{key}"] = value
        return out