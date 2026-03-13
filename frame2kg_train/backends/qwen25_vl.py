from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)
from PIL import Image

from frame2kg_train.backends.base import VLMBackend, BackendArtifacts
from frame2kg_train.data.collators import QwenVLDataCollator


@dataclass
class Qwen25VLBackend(VLMBackend):
    name: str = "qwen25_vl"

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        model_id = cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        min_pixels = int(cfg.get("min_pixels", 256 * 28 * 28))
        max_pixels = int(cfg.get("max_pixels", 1120 * 28 * 28))
        lora_cfg: Dict[str, Any] = cfg.get("lora", {})
        load_in_4bit = bool(cfg.get("load_in_4bit", torch.cuda.is_available()))
        trust_remote_code = bool(cfg.get("trust_remote_code", False))
        attn_impl = cfg.get("attn_implementation", None)
        disable_thinking = bool(cfg.get("disable_thinking", True))

        # Processor
        try:
            proc = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                use_fast=True,
                trust_remote_code=trust_remote_code,
            )
        except TypeError:
            proc = AutoProcessor.from_pretrained(
                model_id,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                trust_remote_code=trust_remote_code,
            )
        if proc.tokenizer.pad_token is None:
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
        proc.tokenizer.padding_side = "left"

        # Model
        model_kwargs: Dict[str, Any] = {}
        if attn_impl:
            model_kwargs["attn_implementation"] = str(attn_impl)

        precision_path = "unknown"
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
            precision_path = "4-bit NF4 (compute_dtype=bfloat16)"
        else:
            dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
            model_kwargs["torch_dtype"] = dtype
            precision_path = f"full-precision torch_dtype={dtype}"

        print(
            f"[backend:{self.name}] model_id={model_id} "
            f"cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()} "
            f"load_in_4bit={load_in_4bit} selected_precision={precision_path}"
        )

        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        except ValueError as e:
            msg = str(e)
            if "model type `qwen3_5`" in msg and "does not recognize this architecture" in msg:
                raise RuntimeError(
                    "This transformers version does not recognize Qwen3.5 model_type `qwen3_5` yet. "
                    "Install a newer transformers build (Qwen docs currently reference main/dev builds) "
                    "or use a Qwen2.5/Qwen3-VL checkpoint."
                ) from e
            raise
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        model_param_dtype = next(model.parameters()).dtype
        print(f"[backend:{self.name}] loaded model parameter dtype={model_param_dtype}")

        pad_id = proc.tokenizer.pad_token_id
        model.generation_config.pad_token_id = pad_id
        model.config.pad_token_id = pad_id
        model.generation_config.do_sample = False
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        # Optional LoRA
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
                target_modules=lora_cfg.get("target_modules", self.default_lora_target_modules()),
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft)

        collator = QwenVLDataCollator(proc, disable_thinking=disable_thinking)
        return BackendArtifacts(model=model, tokenizer=proc.tokenizer, processor=proc, collator=collator)

    def default_lora_target_modules(self) -> List[str]:
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj",
        ]

    @torch.inference_mode()
    def generate_text(self, artifacts: BackendArtifacts, image, prompt: str, max_new_tokens: int) -> str:
        proc = artifacts.processor
        assert proc is not None, "Qwen requires a processor"
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
