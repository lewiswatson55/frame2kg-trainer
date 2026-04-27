from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

try:
    from transformers import MllamaForConditionalGeneration
except Exception as exc:  # pragma: no cover - depends on transformers/torchvision build
    MllamaForConditionalGeneration = None
    _MLLAMA_IMPORT_ERROR = exc
else:  # pragma: no cover - import availability is environment-specific
    _MLLAMA_IMPORT_ERROR = None

from frame2kg_train.backends.base import BackendArtifacts, VLMBackend
from frame2kg_train.backends.llama32_targets import (
    TEXT_TARGETS,
    flatten_resolved_target_modules,
    group_exact_target_module_paths,
    is_allowed_text_decoder_target_path,
    resolve_text_decoder_target_modules,
)
from frame2kg_train.data.collators import Llama32VisionDataCollator


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    texts: List[str] = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            texts.append(str(item["text"]))
    return "\n\n".join(texts)


@dataclass
class Llama32VisionBackend(VLMBackend):
    name: str = "llama32_vision"

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        model_id = cfg.get("model_id", "meta-llama/Llama-3.2-11B-Vision-Instruct")
        lora_cfg: Dict[str, Any] = cfg.get("lora", {})
        load_in_4bit = bool(cfg.get("load_in_4bit", torch.cuda.is_available()))
        attn_impl = cfg.get("attn_implementation", None)

        self._ensure_mllama_support(model_id)

        proc = self._load_processor(model_id=model_id)
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
            f"load_in_4bit={load_in_4bit} selected_precision={precision_path}"
        )

        model = self._load_model(model_id=model_id, model_kwargs=model_kwargs)
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

        collator = Llama32VisionDataCollator(proc)
        return BackendArtifacts(model=model, tokenizer=proc.tokenizer, processor=proc, collator=collator)

    def _ensure_mllama_support(self, model_id: str) -> None:
        if MllamaForConditionalGeneration is None:
            detail = f" Import failed with: {_MLLAMA_IMPORT_ERROR}" if _MLLAMA_IMPORT_ERROR is not None else ""
            raise RuntimeError(
                f"This transformers/torchvision build does not expose MllamaForConditionalGeneration needed for {model_id}."
                " Install a Llama 3.2 Vision-capable Transformers build (Meta/HF document support from transformers>=4.45.0)."
                f"{detail}"
            )

    def _load_processor(self, model_id: str):
        try:
            return AutoProcessor.from_pretrained(model_id)
        except ValueError as e:
            msg = str(e).lower()
            if "mllama" in msg and ("does not recognize" in msg or "unrecognized" in msg):
                raise RuntimeError(
                    "This transformers version does not recognize Mllama yet. "
                    "Install a Llama 3.2 Vision-capable Transformers build."
                ) from e
            raise

    def _load_model(self, model_id: str, model_kwargs: Dict[str, Any]):
        assert MllamaForConditionalGeneration is not None
        try:
            return MllamaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
        except ValueError as e:
            msg = str(e).lower()
            if "mllama" in msg and ("does not recognize" in msg or "unrecognized" in msg):
                raise RuntimeError(
                    "This transformers version does not recognize Mllama yet. "
                    "Install a Llama 3.2 Vision-capable Transformers build."
                ) from e
            raise
        except TypeError:
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("attn_implementation", None)
            return MllamaForConditionalGeneration.from_pretrained(model_id, **fallback_kwargs)

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
                f"[backend:{self.name}] Unsupported Mllama LoRA target names for model_id={model_id}: {unsupported}. "
                "Use q/k/v/o plus gate/down/up projection names."
            )

        missing = [name for name in target_modules if not matches.get(name)]
        if missing:
            raise RuntimeError(
                f"[backend:{self.name}] Mllama text-decoder LoRA targets were not found for model_id={model_id}: {missing}. "
                "The backend resolves qkvo projections under self_attn/cross_attn plus gate/up/down projections under mlp."
            )

        invalid_paths = [
            module_name
            for module_names in matches.values()
            for module_name in module_names
            if not is_allowed_text_decoder_target_path(module_name, text_backbone_prefix)
        ]
        if invalid_paths:
            raise RuntimeError(
                f"[backend:{self.name}] Mllama resolved non-text-decoder LoRA targets for model_id={model_id}: "
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
                f"[backend:{self.name}] Mllama text-decoder LoRA adapters were not applied for model_id={model_id}: "
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

    def _normalise_messages(self, messages: List[Dict[str, Any]]):
        normalised: List[Dict[str, Any]] = []
        images: List[Any] = []

        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content")

            if isinstance(content, str):
                normalised.append({"role": role, "content": [{"type": "text", "text": content}]})
                continue
            if not isinstance(content, list):
                normalised.append({"role": role, "content": [{"type": "text", "text": str(content or "")}]})
                continue

            normalised_content: List[Dict[str, Any]] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    image = item.get("image", item.get("url"))
                    if isinstance(image, str) and "url" not in item:
                        image = Image.open(image).convert("RGB")
                    if image is not None:
                        images.append(image)
                    normalised_content.append({"type": "image"})
                    continue
                if "text" in item:
                    normalised_content.append({"type": "text", "text": str(item["text"])})

            if not normalised_content and role in {"system", "assistant"}:
                normalised_content.append({"type": "text", "text": _extract_text_content(content)})
            normalised.append({"role": role, "content": normalised_content})

        return normalised, images

    def _apply_chat_template(self, proc: Any, messages: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
        return proc.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _processor_call(self, proc: Any, text: str, images: List[Any]):
        kwargs: Dict[str, Any] = {
            "text": [text],
            "return_tensors": "pt",
            "padding": True,
            "add_special_tokens": False,
        }
        if images:
            kwargs["images"] = [images]
        try:
            return proc(**kwargs)
        except TypeError:
            kwargs.pop("add_special_tokens", None)
            return proc(**kwargs)

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
    def generate_from_messages(
        self,
        artifacts: BackendArtifacts,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
    ) -> str:
        proc = artifacts.processor
        assert proc is not None, "Llama 3.2 Vision requires a processor"

        normalised_messages, images = self._normalise_messages(messages)
        prompt = self._apply_chat_template(proc, normalised_messages, add_generation_prompt=True)
        inputs = self._processor_call(proc, prompt, images)
        inputs = self._move_inputs_to_device(inputs, artifacts.model)
        prompt_len = inputs["input_ids"].shape[1]

        generated_ids = artifacts.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        gen_only = generated_ids[:, prompt_len:] if generated_ids.shape[1] > prompt_len else generated_ids
        return self._decode(proc, gen_only)

    @torch.inference_mode()
    def generate_text(self, artifacts: BackendArtifacts, image, prompt: str, max_new_tokens: int) -> str:
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        inputs = self._processor_call(artifacts.processor, prompt, [image])
        inputs = self._move_inputs_to_device(inputs, artifacts.model)
        prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0

        generated_ids = artifacts.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        gen_only = generated_ids[:, prompt_len:] if prompt_len and generated_ids.shape[1] > prompt_len else generated_ids
        return self._decode(artifacts.processor, gen_only)
