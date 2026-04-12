from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from frame2kg_train.backends.base import BackendArtifacts, VLMBackend
from frame2kg_train.data.collators import Gemma4DataCollator


_SUPPORTED_MODELS = {
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
}


@dataclass
class Gemma4VLBackend(VLMBackend):
    name: str = "gemma4_vl"

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        model_id = cfg.get("model_id", "google/gemma-4-E2B-it")
        if model_id not in _SUPPORTED_MODELS:
            supported = ", ".join(sorted(_SUPPORTED_MODELS))
            raise ValueError(f"gemma4_vl currently supports only: {supported}")

        lora_cfg: Dict[str, Any] = cfg.get("lora", {})
        load_in_4bit = bool(cfg.get("load_in_4bit", torch.cuda.is_available()))
        attn_impl = cfg.get("attn_implementation", None)
        disable_thinking = bool(cfg.get("disable_thinking", True))

        try:
            proc = AutoProcessor.from_pretrained(model_id)
        except (KeyError, ValueError) as e:
            if "gemma4" in str(e).lower():
                raise RuntimeError(
                    "This transformers version does not recognize Gemma 4 yet. "
                    "Install transformers>=5.5.3."
                ) from e
            raise
        if proc.tokenizer.pad_token is None:
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
        proc.tokenizer.padding_side = "left"

        model_kwargs: Dict[str, Any] = {}
        if attn_impl:
            model_kwargs["attn_implementation"] = str(attn_impl)

        if load_in_4bit and torch.cuda.is_available():
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs.update(
                {
                    "device_map": "auto",
                    "quantization_config": bnb_cfg,
                    "torch_dtype": torch.bfloat16,
                }
            )
        else:
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available()
                else (torch.float16 if torch.backends.mps.is_available() else torch.float32)
            )
            model_kwargs["torch_dtype"] = dtype

        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        except ValueError as e:
            msg = str(e)
            if "model type `gemma4`" in msg and "does not recognize this architecture" in msg:
                raise RuntimeError(
                    "This transformers version does not recognize Gemma 4 yet. "
                    "Install transformers>=5.5.3."
                ) from e
            raise
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)

        for attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
            token_id = getattr(proc.tokenizer, attr, None)
            if token_id is not None:
                setattr(model.generation_config, attr, token_id)
                setattr(model.config, attr, token_id)
        model.generation_config.do_sample = False
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        if lora_cfg:
            if load_in_4bit and torch.cuda.is_available():
                model = prepare_model_for_kbit_training(model)
            elif hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            peft = LoraConfig(
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(lora_cfg.get("alpha", 16)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                bias="none",
                target_modules=lora_cfg.get("target_modules", "all-linear"),
                modules_to_save=["lm_head", "embed_tokens"],
                ensure_weight_tying=True,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft)

        collator = Gemma4DataCollator(proc, disable_thinking=disable_thinking)
        return BackendArtifacts(model=model, tokenizer=proc.tokenizer, processor=proc, collator=collator)

    @torch.inference_mode()
    def generate_text(self, artifacts: BackendArtifacts, image, prompt: str, max_new_tokens: int) -> str:
        proc = artifacts.processor
        assert proc is not None, "Gemma 4 requires a processor"
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        inputs = proc(text=[prompt], images=[[image]], return_tensors="pt", padding=True)
        inputs = {k: v.to(artifacts.model.device) for k, v in inputs.items() if hasattr(v, "to")}
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0

        eos_token_ids = []
        for token_id in (
            getattr(proc.tokenizer, "eos_token_id", None),
            proc.tokenizer.convert_tokens_to_ids("<turn|>"),
        ):
            if token_id is not None and token_id != proc.tokenizer.unk_token_id and token_id not in eos_token_ids:
                eos_token_ids.append(token_id)

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if eos_token_ids:
            generate_kwargs["eos_token_id"] = eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0]

        gen = artifacts.model.generate(**inputs, **generate_kwargs)
        gen_only = gen[:, prompt_len:] if prompt_len and gen.shape[1] > prompt_len else gen

        raw = proc.tokenizer.batch_decode(
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0].strip()
        if hasattr(proc, "parse_response"):
            try:
                parsed = proc.parse_response(raw)
            except Exception:
                return raw
            if isinstance(parsed, dict):
                content = parsed.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
        return raw
