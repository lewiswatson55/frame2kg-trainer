from __future__ import annotations
from dataclasses import dataclass
import json
from typing import Any, Dict, List

import torch
from PIL import Image

SYSTEM_PROMPT = (
    'You are a VLM that outputs ONLY a single, strict JSON object with exactly the keys "nodes" and "edges". '
    'Use valid JSON: double quotes for all keys and string values, no single quotes, no trailing commas. '
    'Schema — "nodes": [{"id":"str","label":"str","location":"x1,y1,x2,y2,confidence","attributes":{...}}], '
    '"edges":[{"predicate":"str","source":"node.id","target":"node.id"}]. '
    'Output the JSON object only - no code fences, no role tags, no prefixes/suffixes, no prose. '
    'The first character must be "{", and the last must be "}".'
)

def _apply_chat_template(proc: Any, chat: List[Dict[str, Any]], *, add_generation_prompt: bool, disable_thinking: bool) -> str:
    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return proc.apply_chat_template(chat, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return proc.apply_chat_template(chat, **kwargs)

def normalise_graph_key_order(value: Any = "dataset") -> str:
    order = str(value or "dataset").strip().lower()
    if order not in {"dataset", "nodes_first", "edges_first"}:
        raise ValueError("graph_key_order must be one of: dataset, nodes_first, edges_first")
    return order


def _reorder_graph_keys(graph: Any, graph_key_order: str) -> Any:
    order = normalise_graph_key_order(graph_key_order)
    if order == "dataset" or not isinstance(graph, dict):
        return graph

    preferred = ("nodes", "edges") if order == "nodes_first" else ("edges", "nodes")
    reordered: Dict[str, Any] = {}
    for key in preferred:
        if key in graph:
            reordered[key] = graph[key]
    for key, value in graph.items():
        if key not in reordered:
            reordered[key] = value
    return reordered


def graph_to_json_text(graph: Any, graph_key_order: str = "dataset") -> str:
    if isinstance(graph, str):
        if normalise_graph_key_order(graph_key_order) == "dataset":
            return graph
        try:
            graph = json.loads(graph)
        except Exception:
            return graph

    graph = _reorder_graph_keys(graph, graph_key_order)
    return json.dumps(graph, ensure_ascii=False, separators=(",", ":"))


def _non_pad_mask(inputs: Dict[str, torch.Tensor], input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        return attention_mask.to(device=input_ids.device, dtype=torch.bool)
    return input_ids != pad_id


def _mask_prompt_labels(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    inputs: Dict[str, torch.Tensor],
    prompt_inputs: Dict[str, torch.Tensor],
    pad_id: int,
) -> torch.Tensor:
    full_non_pad = _non_pad_mask(inputs, input_ids, pad_id)
    prompt_non_pad = _non_pad_mask(prompt_inputs, prompt_inputs["input_ids"], pad_id)
    prompt_lens = prompt_non_pad.sum(dim=1).to(device=input_ids.device)

    token_positions = full_non_pad.long().cumsum(dim=1)
    labels[full_non_pad & (token_positions <= prompt_lens[:, None])] = -100
    labels[~full_non_pad] = -100
    return labels


@dataclass
class QwenVLDataCollator:
    proc: Any
    graph_key_order: str = "dataset"
    disable_thinking: bool = True

    def __post_init__(self):
        self.graph_key_order = normalise_graph_key_order(self.graph_key_order)
        self.pad_id = self.proc.tokenizer.pad_token_id
        special_tokens = [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>",
            "<think>", "</think>",
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
                {"role": "assistant", "content": [{"type": "text", "text": graph_to_json_text(g, self.graph_key_order)}]},
            ])

        full_texts = [
            _apply_chat_template(
                self.proc,
                c,
                add_generation_prompt=False,
                disable_thinking=self.disable_thinking,
            )
            for c in chats
        ]
        prompts = [
            _apply_chat_template(
                self.proc,
                c[:2],
                add_generation_prompt=True,
                disable_thinking=self.disable_thinking,
            )
            for c in chats
        ]

        if mode == "train":
            inputs = self.proc(text=full_texts, images=[c[1]["content"][0]["image"] for c in chats], return_tensors="pt", padding=True)
            prompt_inputs = self.proc(text=prompts, images=[c[1]["content"][0]["image"] for c in chats], return_tensors="pt", padding=True)

            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            labels = _mask_prompt_labels(labels, input_ids, inputs, prompt_inputs, self.pad_id)
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
class LFMVLDataCollator:
    proc: Any
    graph_key_order: str = "dataset"
    disable_thinking: bool = False

    def __post_init__(self):
        self.graph_key_order = normalise_graph_key_order(self.graph_key_order)
        tok = self.proc.tokenizer
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
        self.special_ids = set(getattr(tok, "all_special_ids", []) or [])

        image_token_id = getattr(self.proc, "image_token_id", None)
        if image_token_id is not None:
            self.special_ids.add(int(image_token_id))

        special_tokens = [
            "<|startoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<image>",
            "<|image_start|>",
            "<|image_end|>",
            "<|img_thumbnail|>",
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
                    {"role": "assistant", "content": [{"type": "text", "text": graph_to_json_text(g, self.graph_key_order)}]},
                ]
            )

        full_texts = [self.proc.apply_chat_template(c, tokenize=False) for c in chats]
        prompts = [self.proc.apply_chat_template(c[:2], tokenize=False, add_generation_prompt=True) for c in chats]

        if mode == "train":
            inputs = self.proc(text=full_texts, images=batched_images, return_tensors="pt", padding=True)
            prompt_inputs = self.proc(text=prompts, images=batched_images, return_tensors="pt", padding=True)

            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            labels = _mask_prompt_labels(labels, input_ids, inputs, prompt_inputs, self.pad_id)
            if self.special_ids:
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for sid in self.special_ids:
                    mask |= input_ids == sid
                labels[mask] = -100
            inputs["labels"] = labels
            return inputs

        prompt_inputs = self.proc(text=prompts, images=batched_images, return_tensors="pt", padding=True)
        prompt_inputs["labels"] = torch.full_like(prompt_inputs["input_ids"], -100)
        return prompt_inputs


@dataclass
class SmolVLMDataCollator:
    proc: Any
    graph_key_order: str = "dataset"

    def __post_init__(self):
        self.graph_key_order = normalise_graph_key_order(self.graph_key_order)
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
                    {"role": "assistant", "content": [{"type": "text", "text": graph_to_json_text(g, self.graph_key_order)}]},
                ]
            )

        full_texts = [self.proc.apply_chat_template(c, tokenize=False) for c in chats]
        prompts = [self.proc.apply_chat_template(c[:2], tokenize=False, add_generation_prompt=True) for c in chats]

        if mode == "train":
            inputs = self.proc(text=full_texts, images=batched_images, return_tensors="pt", padding=True)
            prompt_inputs = self.proc(text=prompts, images=batched_images, return_tensors="pt", padding=True)

            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            labels = _mask_prompt_labels(labels, input_ids, inputs, prompt_inputs, self.pad_id)
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
