from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

try:
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
except Exception:  # pragma: no cover - depends on transformers version
    Qwen3VLForConditionalGeneration = None
    Qwen3VLProcessor = None

from frame2kg_train.backends.base import BackendArtifacts, VLMBackend
from frame2kg_train.data.collators import QwenVLDataCollator


@dataclass
class Qwen3VLBackend(VLMBackend):
    name: str = "qwen3_vl"

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        model_id = cfg.get("model_id", "Qwen/Qwen3-VL-2B-Instruct")
        min_pixels = int(cfg.get("min_pixels", 256 * 28 * 28))
        max_pixels = int(cfg.get("max_pixels", 1120 * 28 * 28))
        lora_cfg: Dict[str, Any] = cfg.get("lora", {})
        load_in_4bit = bool(cfg.get("load_in_4bit", torch.cuda.is_available()))
        attn_impl = cfg.get("attn_implementation", None)

        proc = self._load_processor(model_id=model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        if proc.tokenizer.pad_token is None:
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
        proc.tokenizer.padding_side = "left"

        model_kwargs: Dict[str, Any] = {}
        if attn_impl:
            model_kwargs["_attn_implementation"] = str(attn_impl)
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

        model = self._load_model(model_id=model_id, model_kwargs=model_kwargs)

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
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(lora_cfg.get("alpha", 16)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                bias="none",
                target_modules=lora_cfg.get("target_modules", self.default_lora_target_modules()),
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft)

        collator = QwenVLDataCollator(proc)
        return BackendArtifacts(model=model, tokenizer=proc.tokenizer, processor=proc, collator=collator)

    def _load_processor(self, model_id: str, min_pixels: int, max_pixels: int):
        kwargs_variants = (
            {"min_pixels": min_pixels, "max_pixels": max_pixels, "use_fast": True},
            {"min_pixels": min_pixels, "max_pixels": max_pixels},
            {},
        )

        if Qwen3VLProcessor is not None:
            for proc_kwargs in kwargs_variants:
                try:
                    return Qwen3VLProcessor.from_pretrained(model_id, **proc_kwargs)
                except TypeError:
                    continue

        for proc_kwargs in kwargs_variants:
            try:
                return AutoProcessor.from_pretrained(model_id, **proc_kwargs)
            except TypeError:
                continue

        return AutoProcessor.from_pretrained(model_id)

    def _load_model(self, model_id: str, model_kwargs: Dict[str, Any]):
        specific_error: Exception | None = None
        if Qwen3VLForConditionalGeneration is not None:
            try:
                return Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
            except TypeError:
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs.pop("_attn_implementation", None)
                return Qwen3VLForConditionalGeneration.from_pretrained(model_id, **fallback_kwargs)
            except Exception as exc:
                specific_error = exc

        try:
            return AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        except TypeError:
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("_attn_implementation", None)
            return AutoModelForImageTextToText.from_pretrained(model_id, **fallback_kwargs)
        except Exception:
            if specific_error is not None:
                raise specific_error
            raise

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
        assert proc is not None, "Qwen3-VL requires a processor"
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        inputs = proc(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(artifacts.model.device) for k, v in inputs.items() if hasattr(v, "to")}
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0

        gen = artifacts.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_only = gen[:, prompt_len:] if prompt_len and gen.shape[1] > prompt_len else gen
        return proc.tokenizer.batch_decode(
            gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
