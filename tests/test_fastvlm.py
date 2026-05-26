from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
from PIL import Image

from frame2kg_train.backends.base import BackendArtifacts
from frame2kg_train.backends.fastvlm import FastVLMBackend
from frame2kg_train.backends.tokenizer_extension import TokenizerExtension, add_tokens_from_config
from frame2kg_train.data.collators import FastVLMDataCollator, graph_to_json_text
from frame2kg_train.data.graph_formats import COMPRESSED_GRAPH_TOKENS, compressed_graph_to_json
from frame2kg_train.train.callbacks import SaveAdaptersCallback
from frame2kg_train.train.registry import get_backend


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    unk_token_id = -1
    pad_token = "<|endoftext|>"
    eos_token = "<|endoftext|>"
    padding_side = "left"
    all_special_ids = [0, 2, 10, 11, 99]

    _special = {
        "<|im_start|>": 10,
        "<|im_end|>": 11,
        "<|endoftext|>": 2,
        "<image>": 99,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._special.get(token, self.unk_token_id)

    def encode_text(self, text: str) -> List[int]:
        return [1000 + ord(ch) for ch in text]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        assert add_special_tokens is False
        return self.encode_text(text)

    def decode_text(self, ids: List[int]) -> str:
        return "".join(chr(token_id - 1000) for token_id in ids if token_id >= 1000)

    def batch_decode(self, rows, **_kwargs):
        return [self.decode_text(row.tolist() if hasattr(row, "tolist") else list(row)) for row in rows]


class FakeProcessor:
    image_token = "<image>"

    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def _message_ids(self, message: Dict[str, Any]) -> List[int]:
        ids = [10]
        ids.extend(self.tokenizer.encode_text(message["role"]))
        ids.extend(self.tokenizer.encode_text("\n"))
        for content in message["content"]:
            if content["type"] == "image":
                ids.append(99)
        for content in message["content"]:
            if content["type"] == "text":
                ids.extend(self.tokenizer.encode_text("\n" + content["text"]))
        ids.append(11)
        return ids

    def _chat_ids(self, chat: List[Dict[str, Any]], add_generation_prompt: bool) -> List[int]:
        ids: List[int] = []
        for message in chat:
            ids.extend(self._message_ids(message))
        if add_generation_prompt:
            ids.append(10)
            ids.extend(self.tokenizer.encode_text("assistant\n"))
        return ids

    def apply_chat_template(self, chats, *, add_generation_prompt, tokenize, return_dict, padding, return_tensors):
        assert tokenize is True
        assert return_dict is True
        assert padding is True
        assert return_tensors == "pt"
        rows = [self._chat_ids(chat, add_generation_prompt) for chat in chats]
        max_len = max(len(row) for row in rows)
        padded = []
        masks = []
        for row in rows:
            pad_len = max_len - len(row)
            padded.append([self.tokenizer.pad_token_id] * pad_len + row)
            masks.append([0] * pad_len + [1] * len(row))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "pixel_values": torch.ones(len(rows), 3, 2, 2),
        }

    def __call__(self, *, text, images, return_tensors, padding):
        assert return_tensors == "pt"
        assert padding is True
        rows = [self.tokenizer.encode_text(item) for item in text]
        max_len = max(len(row) for row in rows)
        padded = []
        masks = []
        for row in rows:
            pad_len = max_len - len(row)
            padded.append([self.tokenizer.pad_token_id] * pad_len + row)
            masks.append([0] * pad_len + [1] * len(row))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "pixel_values": torch.ones(len(images), 3, 2, 2),
        }

    def batch_decode(self, rows, **kwargs):
        return self.tokenizer.batch_decode(rows, **kwargs)


class FakeGenerationModel:
    device = torch.device("cpu")

    def parameters(self):
        yield torch.nn.Parameter(torch.zeros(1))

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        generated = torch.tensor([[1000 + ord("{"), 1000 + ord("}")]], dtype=torch.long)
        return torch.cat([input_ids, generated.expand(input_ids.shape[0], -1)], dim=1)


def test_get_backend_fastvlm_resolves():
    backend = get_backend("fastvlm")
    assert backend.name == "fastvlm"


def test_transformers_fastvlm_import_smoke():
    pytest.importorskip("transformers")
    try:
        from transformers import FastVlmForConditionalGeneration  # noqa: F401
    except Exception as exc:
        pytest.skip(f"installed Transformers build does not expose native FastVLM: {exc}")


def test_fastvlm_collator_masks_only_assistant_json_tokens():
    proc = FakeProcessor()
    collator = FastVLMDataCollator(proc, graph_key_order="nodes_first")
    image = Image.new("RGB", (8, 8), "white")
    graphs = [
        {"edges": [], "nodes": [{"id": "n1", "label": "frame"}]},
        {"edges": [{"predicate": "on", "source": "n1", "target": "n2"}], "nodes": []},
    ]
    batch = [{"image": image, "graph": graph, "mode": "train"} for graph in graphs]

    inputs = collator(batch)

    assert {"input_ids", "attention_mask", "pixel_values", "labels"} <= set(inputs)
    assert inputs["input_ids"].shape == inputs["labels"].shape
    for row_index, graph in enumerate(graphs):
        trainable_mask = inputs["labels"][row_index] != -100
        assert trainable_mask.any()
        trainable_ids = inputs["input_ids"][row_index][trainable_mask].tolist()
        assert not ({0, 2, 10, 11, 99} & set(trainable_ids))
        assert proc.tokenizer.decode_text(trainable_ids) == graph_to_json_text(graph, "nodes_first")


def test_fastvlm_collator_can_train_compressed_graph_strings():
    proc = FakeProcessor()
    collator = FastVLMDataCollator(proc, target_format="compressed_tokens")
    image = Image.new("RGB", (8, 8), "white")
    graph = (
        "<|graph|><|nodes|><|node|><|id|>n1<|label|>frame"
        "<|bbox|>0 0 1 1<|conf|>0.9<|attrs|>size=small"
        "<|edges|><|end_graph|>"
    )

    inputs = collator([{"image": image, "graph": graph, "mode": "train"}])

    trainable_mask = inputs["labels"][0] != -100
    assert trainable_mask.any()
    trainable_ids = inputs["input_ids"][0][trainable_mask].tolist()
    assert proc.tokenizer.decode_text(trainable_ids) == graph


def test_compressed_graph_parser_returns_metric_shape():
    graph = (
        "<|graph|><|nodes|><|node|><|id|>person1<|label|>man"
        "<|bbox|>0.0 0.2 0.45 0.95<|conf|>0.9"
        "<|attrs|>appearance=white cotton<|attrs|>size=medium"
        "<|edges|><|edge|><|src|>person1<|pred|>left_of<|tgt|>desk"
        "<|end_graph|>"
    )

    parsed = compressed_graph_to_json(graph)

    assert parsed == {
        "nodes": [
            {
                "id": "person1",
                "label": "man",
                "location": "0.0 0.2 0.45 0.95",
                "confidence": "0.9",
                "attributes": {"appearance": "white cotton", "size": "medium"},
            }
        ],
        "edges": [{"source": "person1", "predicate": "left_of", "target": "desk"}],
    }


class FakeAddedTokenizer:
    unk_token_id = -1

    def __init__(self):
        self.vocab = {"base": 0}
        self.added_order: List[str] = []

    def __len__(self):
        return len(self.vocab)

    def _token_text(self, token) -> str:
        return str(getattr(token, "content", token))

    def add_tokens(self, tokens, special_tokens=False):
        assert special_tokens is False
        added = 0
        for token in tokens:
            token = self._token_text(token)
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.added_order.append(token)
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        assert add_special_tokens is False
        if text in self.vocab:
            return [self.vocab[text]]
        return [1000 + ord(ch) for ch in text]


def test_compressed_tokens_are_added_as_single_non_special_tokens():
    tokenizer = FakeAddedTokenizer()

    extension = add_tokens_from_config(tokenizer, {"target_format": "compressed_tokens"})

    assert extension.tokens == COMPRESSED_GRAPH_TOKENS
    assert extension.added_count == len(COMPRESSED_GRAPH_TOKENS)
    assert extension.token_ids == [tokenizer.convert_tokens_to_ids(token) for token in COMPRESSED_GRAPH_TOKENS]
    for token in COMPRESSED_GRAPH_TOKENS:
        assert tokenizer.encode(token, add_special_tokens=False) == [tokenizer.convert_tokens_to_ids(token)]


def test_adapter_checkpoint_can_include_processor_and_token_manifest(tmp_path, monkeypatch):
    class FakeModel:
        def save_pretrained(self, path):
            (Path(path) / "adapter_model.safetensors").write_text("adapter")

    class FakeProc:
        def save_pretrained(self, path):
            (Path(path) / "tokenizer_config.json").write_text("{}")

    class FakeArtifact:
        def __init__(self, *args, **kwargs):
            self.dirs = []

        def add_dir(self, path):
            self.dirs.append(path)

    class FakeSaveTokenizer:
        _frame2kg_tokenizer_extension = TokenizerExtension(
            target_format="compressed_tokens",
            tokens=["<|graph|>"],
            token_ids=[123],
            added_count=1,
        )

        def save_pretrained(self, path):
            (Path(path) / "tokenizer.json").write_text("{}")

    tokenizer = FakeSaveTokenizer()
    logged = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "wandb",
        SimpleNamespace(Artifact=FakeArtifact, log_artifact=lambda artifact, aliases=None: logged.append((artifact, aliases))),
    )

    callback = SaveAdaptersCallback(
        proc=FakeProc(),
        tokenizer=tokenizer,
        max_new_tokens=128,
        include_processor=True,
    )
    args = SimpleNamespace(output_dir=str(tmp_path), run_name="run")
    state = SimpleNamespace(global_step=7)

    callback.on_save(args, state, SimpleNamespace(), model=FakeModel())

    checkpoint_dir = tmp_path / "adapters-step-7"
    assert (checkpoint_dir / "adapter_model.safetensors").exists()
    assert (checkpoint_dir / "tokenizer_config.json").exists()
    assert (checkpoint_dir / "tokenizer.json").exists()
    assert (checkpoint_dir / "frame2kg_tokenizer_extension.json").exists()
    assert logged


def test_fastvlm_generation_strips_prompt_tokens():
    proc = FakeProcessor()
    artifacts = BackendArtifacts(
        model=FakeGenerationModel(),
        tokenizer=proc.tokenizer,
        processor=proc,
        collator=FastVLMDataCollator(proc),
    )
    image = Image.new("RGB", (8, 8), "white")

    output = FastVLMBackend().generate_text(artifacts, image, "formatted prompt", max_new_tokens=2)

    assert output == "{}"
