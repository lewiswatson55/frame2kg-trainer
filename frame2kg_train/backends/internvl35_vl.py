from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from frame2kg_train.backends.base import BackendArtifacts, VLMBackend
from frame2kg_train.backends.internvl35_targets import (
    TEXT_TARGETS,
    flatten_resolved_target_modules,
    group_exact_target_module_paths,
    is_allowed_text_decoder_target_path,
    resolve_text_decoder_target_modules,
)
from frame2kg_train.data.collators import InternVL35DataCollator


@dataclass
class InternVL35Backend(VLMBackend):
    name: str = "internvl35_vl"
    _processor_kwargs: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        model_id = cfg.get("model_id", "OpenGVLab/InternVL3_5-1B-HF")
        lora_cfg: Dict[str, Any] = cfg.get("lora", {})
        load_in_4bit = bool(cfg.get("load_in_4bit", torch.cuda.is_available()))
        trust_remote_code = bool(cfg.get("trust_remote_code", True))
        attn_impl = cfg.get("attn_implementation", None)
        disable_thinking = bool(cfg.get("disable_thinking", True))
        crop_to_patches = bool(cfg.get("crop_to_patches", True))
        min_patches = int(cfg.get("min_patches", 1))
        max_patches = int(cfg.get("max_patches", 12))
        self._processor_kwargs = {
            "crop_to_patches": crop_to_patches,
            "min_patches": min_patches,
            "max_patches": max_patches,
        }

        proc = self._load_processor(model_id=model_id, trust_remote_code=trust_remote_code)
        if proc.tokenizer.pad_token is None:
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
        proc.tokenizer.padding_side = "left"

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
                    "quantization_config": bnb_cfg,
                    "device_map": "auto",
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
            f"crop_to_patches={crop_to_patches} min_patches={min_patches} max_patches={max_patches}"
        )

        model = self._load_model(
            model_id=model_id,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
        )
        model_param_dtype = next(model.parameters()).dtype
        print(f"[backend:{self.name}] loaded model parameter dtype={model_param_dtype}")

        self._set_pad_token_id(model, proc.tokenizer.pad_token_id)
        model.generation_config.do_sample = False
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "use_cache"):
            text_config.use_cache = False
        model.gradient_checkpointing_enable()

        if lora_cfg:
            target_modules = [str(name) for name in lora_cfg.get("target_modules", self.default_lora_target_modules())]
            text_backbone_prefix, available_targets = resolve_text_decoder_target_modules(model, target_modules)
            self._log_text_backbone_prefix(text_backbone_prefix)
            self._validate_lora_target_modules(
                model_id=model_id,
                target_modules=target_modules,
                text_backbone_prefix=text_backbone_prefix,
                matches=available_targets,
            )
            self._log_target_module_summary("available", available_targets)
            resolved_target_modules = flatten_resolved_target_modules(target_modules, available_targets)
            print(f"[backend:{self.name}] resolved text-decoder LoRA module paths={len(resolved_target_modules)}")

            if load_in_4bit and torch.cuda.is_available():
                model = prepare_model_for_kbit_training(model)
            elif hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

            peft = LoraConfig(
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(lora_cfg.get("alpha", 16)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                bias="none",
                target_modules=resolved_target_modules,
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft)
            adapted_target_modules = self._collect_lora_wrapped_module_names(model, resolved_target_modules)
            self._validate_lora_wrapped_target_modules(
                model_id=model_id,
                resolved_target_modules=resolved_target_modules,
                adapted_target_modules=adapted_target_modules,
            )
            adapted_targets = group_exact_target_module_paths(adapted_target_modules, target_modules)
            self._log_target_module_summary("adapted", adapted_targets)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()

        collator = InternVL35DataCollator(
            proc,
            disable_thinking=disable_thinking,
            crop_to_patches=crop_to_patches,
            min_patches=min_patches,
            max_patches=max_patches,
        )
        return BackendArtifacts(model=model, tokenizer=proc.tokenizer, processor=proc, collator=collator)

    def _load_processor(self, model_id: str, trust_remote_code: bool):
        try:
            return AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except ValueError as e:
            msg = str(e).lower()
            if "internvl" in msg and ("does not recognize" in msg or "unrecognized" in msg):
                raise RuntimeError(
                    "This transformers version does not recognize InternVL yet. "
                    "Install an InternVL-capable Transformers build; InternVL3.5-HF model cards require transformers>=4.52.1."
                ) from e
            raise

    def _load_model(self, model_id: str, model_kwargs: Dict[str, Any], trust_remote_code: bool):
        try:
            return AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
        except ValueError as e:
            msg = str(e).lower()
            if "internvl" in msg and ("does not recognize" in msg or "unrecognized" in msg):
                raise RuntimeError(
                    "This transformers version does not recognize InternVL yet. "
                    "Install an InternVL-capable Transformers build; InternVL3.5-HF model cards require transformers>=4.52.1."
                ) from e
            raise
        except TypeError:
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("attn_implementation", None)
            return AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=trust_remote_code,
                **fallback_kwargs,
            )

    def _set_pad_token_id(self, model: Any, pad_id: int | None) -> None:
        if pad_id is None:
            return
        model.generation_config.pad_token_id = pad_id
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = pad_id
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "pad_token_id"):
            text_config.pad_token_id = pad_id

    def _collect_lora_wrapped_module_names(self, model: Any, resolved_target_modules: List[str]) -> List[str]:
        variant_to_original: Dict[str, str] = {}
        for module_name in resolved_target_modules:
            for candidate in self._candidate_wrapped_module_names(module_name):
                variant_to_original[candidate] = module_name

        matched = set()
        for module_name, module in model.named_modules():
            original_name = variant_to_original.get(module_name)
            if original_name is not None and hasattr(module, "base_layer"):
                matched.add(original_name)

        return [module_name for module_name in resolved_target_modules if module_name in matched]

    def _candidate_wrapped_module_names(self, module_name: str) -> List[str]:
        return [
            module_name,
            f"base_model.model.{module_name}",
        ]

    def _validate_lora_target_modules(
        self,
        *,
        model_id: str,
        target_modules: List[str],
        text_backbone_prefix: str,
        matches: Dict[str, List[str]],
    ) -> None:
        unsupported = sorted(set(target_modules) - TEXT_TARGETS)
        if unsupported:
            raise RuntimeError(
                f"[backend:{self.name}] Unsupported InternVL LoRA target names for model_id={model_id}: {unsupported}. "
                "Use q/k/v/o plus gate/down/up projection names."
            )

        missing = [name for name in target_modules if not matches.get(name)]
        if missing:
            raise RuntimeError(
                f"[backend:{self.name}] InternVL text-decoder LoRA targets were not found for model_id={model_id}: {missing}. "
                "The backend resolves qkvo projections under self_attn plus gate/up/down projections under mlp."
            )

        invalid_paths = [
            module_name
            for module_names in matches.values()
            for module_name in module_names
            if not is_allowed_text_decoder_target_path(module_name, text_backbone_prefix)
        ]
        if invalid_paths:
            raise RuntimeError(
                f"[backend:{self.name}] InternVL resolved non-text-decoder LoRA targets for model_id={model_id}: "
                f"{invalid_paths[:5]}"
            )

    def _validate_lora_wrapped_target_modules(
        self,
        *,
        model_id: str,
        resolved_target_modules: List[str],
        adapted_target_modules: List[str],
    ) -> None:
        adapted_set = set(adapted_target_modules)
        missing = [module_name for module_name in resolved_target_modules if module_name not in adapted_set]
        if missing:
            raise RuntimeError(
                f"[backend:{self.name}] InternVL text-decoder LoRA adapters were not applied for model_id={model_id}: "
                f"{missing[:5]}"
            )

    def _log_text_backbone_prefix(self, text_backbone_prefix: str) -> None:
        print(f"[backend:{self.name}] text backbone prefix={text_backbone_prefix}")

    def _log_target_module_summary(self, label: str, matches: Dict[str, List[str]]) -> None:
        counts = {name: len(module_names) for name, module_names in matches.items()}
        print(f"[backend:{self.name}] {label} text-decoder LoRA target counts={counts}")
        for name, module_names in matches.items():
            preview = ", ".join(module_names[:3])
            if len(module_names) > 3:
                preview = f"{preview}, ..."
            print(f"[backend:{self.name}] {label} {name}: {preview}")

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

    def _processor_call(self, proc: Any, text: str, image: Image.Image):
        kwargs: Dict[str, Any] = {
            "text": [text],
            "images": [image],
            "return_tensors": "pt",
            "padding": True,
            **self._processor_kwargs,
        }
        try:
            return proc(**kwargs)
        except TypeError:
            for key in ("crop_to_patches", "min_patches", "max_patches"):
                kwargs.pop(key, None)
            return proc(**kwargs)

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
        assert proc is not None, "InternVL3.5 requires a processor"
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        inputs = self._processor_call(proc, prompt, image)
        inputs = self._move_inputs_to_device(inputs, artifacts.model)
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0

        generated_ids = artifacts.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        gen_only = generated_ids[:, prompt_len:] if prompt_len and generated_ids.shape[1] > prompt_len else generated_ids
        return self._decode(proc, gen_only)
