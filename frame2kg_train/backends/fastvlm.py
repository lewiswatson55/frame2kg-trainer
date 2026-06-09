from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from PIL import Image

from frame2kg_train.backends.base import BackendArtifacts, VLMBackend
from frame2kg_train.backends.tokenizer_extension import (
    add_tokens_from_config,
    lora_config_kwargs,
    resize_model_embeddings_for_tokenizer,
)
from frame2kg_train.data.collators import FastVLMDataCollator


@dataclass
class FastVLMBackend(VLMBackend):
    name: str = "fastvlm"

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        try:
            from transformers import AutoProcessor, BitsAndBytesConfig, FastVlmForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "FastVLM requires a Transformers build with native FastVlmForConditionalGeneration support. "
                "Install the repository pin from requirements.txt/pyproject.toml."
            ) from exc

        model_id = cfg.get("model_id", "KamilaMila/FastVLM-0.5B")
        lora_cfg: Dict[str, Any] = cfg.get("lora", {})
        load_in_4bit = bool(cfg.get("load_in_4bit", torch.cuda.is_available()))
        trust_remote_code = bool(cfg.get("trust_remote_code", False))
        attn_impl = cfg.get("attn_implementation", None)

        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        if proc.tokenizer.pad_token is None:
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
        proc.tokenizer.padding_side = "left"
        tokenizer_extension = add_tokens_from_config(proc.tokenizer, cfg)

        model_kwargs: Dict[str, Any] = {}
        decoder_attn_impl = self._decoder_attention_implementation(attn_impl)
        if decoder_attn_impl is not None:
            model_kwargs["attn_implementation"] = decoder_attn_impl

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
            dtype = (
                torch.bfloat16
                if torch.cuda.is_available()
                else (torch.float16 if torch.backends.mps.is_available() else torch.float32)
            )
            model_kwargs["torch_dtype"] = dtype
            precision_path = f"full-precision torch_dtype={dtype}"

        print(
            f"[backend:{self.name}] model_id={model_id} "
            f"cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()} "
            f"load_in_4bit={load_in_4bit} selected_precision={precision_path} "
            f"attn_implementation={decoder_attn_impl}"
        )

        try:
            model = FastVlmForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        except TypeError:
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("attn_implementation", None)
            model = FastVlmForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                **fallback_kwargs,
            )

        model_param_dtype = next(model.parameters()).dtype
        print(f"[backend:{self.name}] loaded model parameter dtype={model_param_dtype}")
        resize_model_embeddings_for_tokenizer(model, proc.tokenizer, tokenizer_extension)

        self._set_pad_token_id(model, proc.tokenizer.pad_token_id)
        model.generation_config.do_sample = False
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = False
        model.gradient_checkpointing_enable()

        if lora_cfg:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            target_modules = [str(name) for name in lora_cfg.get("target_modules", self.default_lora_target_modules())]
            resolved_target_modules = self._resolve_text_decoder_target_modules(model, target_modules)
            print(f"[backend:{self.name}] resolved text-decoder LoRA module paths={len(resolved_target_modules)}")

            if load_in_4bit and torch.cuda.is_available():
                model = prepare_model_for_kbit_training(model)
            elif hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

            peft = LoraConfig(
                **lora_config_kwargs(
                    LoraConfig,
                    model,
                    lora_cfg,
                    resolved_target_modules,
                    tokenizer_extension,
                )
            )
            model = get_peft_model(model, peft)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()

        collator = FastVLMDataCollator(proc)
        return BackendArtifacts(model=model, tokenizer=proc.tokenizer, processor=proc, collator=collator)

    def _decoder_attention_implementation(self, attn_impl: Any) -> Dict[str, str] | None:
        if not attn_impl:
            return None
        if isinstance(attn_impl, dict):
            extra_keys = sorted(set(attn_impl) - {"text_config"})
            if extra_keys:
                raise ValueError(
                    "FastVLM attention overrides must be decoder-scoped as "
                    f"attn_implementation.text_config; unsupported keys: {extra_keys}"
                )
            value = attn_impl.get("text_config")
            return {"text_config": str(value)} if value else None
        return {"text_config": str(attn_impl)}

    def _set_pad_token_id(self, model: Any, pad_id: int | None) -> None:
        if pad_id is None:
            return
        model.generation_config.pad_token_id = pad_id
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = pad_id
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "pad_token_id"):
            text_config.pad_token_id = pad_id

    def _resolve_text_decoder_target_modules(self, model: Any, target_modules: List[str]) -> List[str]:
        allowed = set(self.default_lora_target_modules())
        unsupported = sorted(set(target_modules) - allowed)
        if unsupported:
            raise RuntimeError(
                f"[backend:{self.name}] Unsupported FastVLM LoRA target names: {unsupported}. "
                "Use Qwen-style q/k/v/o plus gate/down/up projection names."
            )

        matches: Dict[str, List[str]] = {name: [] for name in target_modules}
        for module_name, _module in model.named_modules():
            if not module_name.startswith("model.language_model."):
                continue
            leaf = module_name.rsplit(".", 1)[-1]
            if leaf in matches:
                matches[leaf].append(module_name)

        missing = [name for name, module_names in matches.items() if not module_names]
        if missing:
            raise RuntimeError(
                f"[backend:{self.name}] FastVLM text-decoder LoRA targets were not found: {missing}. "
                "Expected Qwen-style modules under model.language_model."
            )

        counts = {name: len(module_names) for name, module_names in matches.items()}
        print(f"[backend:{self.name}] available text-decoder LoRA target counts={counts}")
        return [module_name for name in target_modules for module_name in matches[name]]

    def _model_device(self, model: Any) -> torch.device:
        device = getattr(model, "device", None)
        if isinstance(device, torch.device) and device.type != "meta":
            return device
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _move_inputs_to_device(self, batch: Any, model: Any) -> Dict[str, Any]:
        device = self._model_device(model)
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}

    def _decode(self, proc: Any, token_ids: torch.Tensor) -> str:
        if hasattr(proc, "batch_decode"):
            return proc.batch_decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        return proc.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

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
        assert proc is not None, "FastVLM requires a processor"
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        inputs = proc(text=[prompt], images=[image], return_tensors="pt", padding=True)
        inputs = self._move_inputs_to_device(inputs, artifacts.model)
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = artifacts.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        gen_only = generated_ids[:, prompt_len:] if generated_ids.shape[1] > prompt_len else generated_ids
        return self._decode(proc, gen_only)
