from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from frame2kg_train.backends.base import BackendArtifacts, VLMBackend
from frame2kg_train.backends.tokenizer_extension import (
    add_tokens_from_config,
    lora_config_kwargs,
    resize_model_embeddings_for_tokenizer,
)
from frame2kg_train.data.collators import SmolVLMDataCollator


@dataclass
class SmolVLM2Backend(VLMBackend):
    name: str = "smolvlm2"

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        model_id = cfg.get("model_id", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        lora_cfg: Dict[str, Any] = cfg.get("lora", {})
        load_in_4bit = bool(cfg.get("load_in_4bit", torch.cuda.is_available()))
        attn_impl = cfg.get("attn_implementation", None)

        proc = AutoProcessor.from_pretrained(model_id)
        if proc.tokenizer.pad_token is None:
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
        proc.tokenizer.padding_side = "left"
        tokenizer_extension = add_tokens_from_config(proc.tokenizer, cfg)

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
                    "quantization_config": bnb_cfg,
                    "device_map": "auto",
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
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        resize_model_embeddings_for_tokenizer(model, proc.tokenizer, tokenizer_extension)

        pad_id = proc.tokenizer.pad_token_id
        model.generation_config.pad_token_id = pad_id
        model.config.pad_token_id = pad_id
        model.generation_config.do_sample = False
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        if lora_cfg:
            if load_in_4bit and torch.cuda.is_available():
                model = prepare_model_for_kbit_training(model)
            elif hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            peft = LoraConfig(
                **lora_config_kwargs(
                    LoraConfig,
                    model,
                    lora_cfg,
                    lora_cfg.get("target_modules", self.default_lora_target_modules()),
                    tokenizer_extension,
                )
            )
            model = get_peft_model(model, peft)

        collator = SmolVLMDataCollator(proc)
        return BackendArtifacts(model=model, tokenizer=proc.tokenizer, processor=proc, collator=collator)

    def default_lora_target_modules(self) -> List[str]:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "down_proj",
            "up_proj",
        ]

    @torch.inference_mode()
    def generate_text(self, artifacts: BackendArtifacts, image, prompt: str, max_new_tokens: int) -> str:
        proc = artifacts.processor
        assert proc is not None, "SmolVLM2 requires a processor"
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        inputs = proc(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(artifacts.model.device) for k, v in inputs.items() if hasattr(v, "to")}
        prompt_len = inputs["input_ids"].shape[1]

        gen = artifacts.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_only = gen[:, prompt_len:] if gen.shape[1] > prompt_len else gen
        return proc.tokenizer.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
