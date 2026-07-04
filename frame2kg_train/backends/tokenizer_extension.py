from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from frame2kg_train.data.graph_formats import (
    COMPRESSED_GRAPH_TOKENS,
    COMPRESSED_TOKENS_TARGET_FORMAT,
    normalise_target_format,
)


@dataclass
class TokenizerExtension:
    target_format: str
    tokens: List[str]
    token_ids: List[int]
    added_count: int

    @property
    def enabled(self) -> bool:
        return bool(self.tokens)


def _tokens_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    target_format = normalise_target_format(cfg.get("target_format", "json"))
    if "added_tokens" in cfg:
        configured = cfg["added_tokens"]
    elif "tokenizer_added_tokens" in cfg:
        configured = cfg["tokenizer_added_tokens"]
    else:
        configured = None
    if configured is None and target_format == COMPRESSED_TOKENS_TARGET_FORMAT:
        configured = COMPRESSED_GRAPH_TOKENS
    if configured is None:
        return []
    if isinstance(configured, str):
        return [configured]
    return [str(token) for token in configured]


def _as_added_tokens(tokens: Iterable[str]) -> List[Any]:
    try:
        from tokenizers import AddedToken
    except Exception:
        return list(tokens)
    return [
        AddedToken(
            token,
            single_word=False,
            lstrip=False,
            rstrip=False,
            normalized=False,
            special=False,
        )
        for token in tokens
    ]


def _encode_one(tokenizer: Any, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(ids)


def add_tokens_from_config(tokenizer: Any, cfg: Dict[str, Any]) -> TokenizerExtension:
    target_format = normalise_target_format(cfg.get("target_format", "json"))
    tokens = _tokens_from_cfg(cfg)
    if not tokens:
        extension = TokenizerExtension(target_format=target_format, tokens=[], token_ids=[], added_count=0)
        setattr(tokenizer, "_frame2kg_tokenizer_extension", extension)
        return extension

    before_size = len(tokenizer)
    try:
        added_count = int(tokenizer.add_tokens(_as_added_tokens(tokens), special_tokens=False))
    except TypeError:
        added_count = int(tokenizer.add_tokens(tokens))

    token_ids: List[int] = []
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        encoded = _encode_one(tokenizer, token)
        if len(encoded) != 1 or int(encoded[0]) != int(token_id):
            raise RuntimeError(
                f"Added token {token!r} must encode to exactly one token id; "
                f"convert_tokens_to_ids={token_id!r}, encode={encoded!r}"
            )
        token_ids.append(int(token_id))

    extension = TokenizerExtension(
        target_format=target_format,
        tokens=tokens,
        token_ids=token_ids,
        added_count=added_count,
    )
    setattr(tokenizer, "_frame2kg_tokenizer_extension", extension)
    print(
        f"[tokenizer] target_format={target_format} vocab_before={before_size} "
        f"vocab_after={len(tokenizer)} configured_tokens={len(tokens)} newly_added={added_count}"
    )
    return extension


def resize_model_embeddings_for_tokenizer(
    model: Any,
    tokenizer: Any,
    extension: TokenizerExtension,
) -> None:
    if not extension.enabled:
        return
    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if input_embeddings is None:
        raise RuntimeError("Cannot resize token embeddings because model has no input embedding layer")
    old_size = int(input_embeddings.weight.shape[0])
    new_size = int(len(tokenizer))
    if old_size != new_size:
        model.resize_token_embeddings(new_size)
        print(f"[tokenizer] resized model token embeddings from {old_size} to {new_size}")


def _module_name_for_object(model: Any, target: Any) -> str | None:
    for name, module in model.named_modules():
        if module is target:
            return name
    return None


def _same_weight(left: Any, right: Any) -> bool:
    if left is right:
        return True
    left_weight = getattr(left, "weight", None)
    right_weight = getattr(right, "weight", None)
    if left_weight is None or right_weight is None:
        return False
    try:
        return left_weight.data_ptr() == right_weight.data_ptr()
    except Exception:
        return False


def _trainable_token_indices_for_model(model: Any, token_ids: List[int]) -> List[int] | Dict[str, List[int]]:
    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if output_embeddings is None or input_embeddings is None or _same_weight(input_embeddings, output_embeddings):
        return token_ids

    input_name = _module_name_for_object(model, input_embeddings)
    output_name = _module_name_for_object(model, output_embeddings)
    if input_name is None or output_name is None:
        raise RuntimeError(
            "The model has untied input/output embeddings, but one of their module names could not be resolved. "
            "Cannot save an unmerged LoRA with newly-added output tokens safely."
        )
    return {input_name: token_ids, output_name: token_ids}


def lora_config_kwargs(
    lora_config_cls: Any,
    model: Any,
    lora_cfg: Dict[str, Any],
    target_modules: Any,
    extension: TokenizerExtension,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "r": int(lora_cfg.get("r", 8)),
        "lora_alpha": int(lora_cfg.get("alpha", 16)),
        "lora_dropout": float(lora_cfg.get("dropout", 0.05)),
        "bias": "none",
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }
    modules_to_save = lora_cfg.get("modules_to_save")
    if modules_to_save:
        kwargs["modules_to_save"] = modules_to_save

    if extension.enabled:
        signature_target = getattr(lora_config_cls, "__init__", lora_config_cls)
        supported = set(inspect.signature(signature_target).parameters)
        if "trainable_token_indices" not in supported:
            raise RuntimeError(
                "Compressed-token LoRA requires a PEFT version whose LoraConfig supports "
                "trainable_token_indices so added token embeddings can be saved unmerged."
            )
        kwargs["trainable_token_indices"] = _trainable_token_indices_for_model(model, extension.token_ids)
        if "ensure_weight_tying" in supported:
            kwargs["ensure_weight_tying"] = True
    return kwargs


def save_tokenizer_extension_manifest(tokenizer: Any, output_dir: str) -> None:
    extension = getattr(tokenizer, "_frame2kg_tokenizer_extension", None)
    if not extension or not getattr(extension, "enabled", False):
        return
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "frame2kg_tokenizer_extension.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "target_format": extension.target_format,
                "tokens": extension.tokens,
                "token_ids": extension.token_ids,
                "added_count": extension.added_count,
                "single_token_ids_verified": True,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
