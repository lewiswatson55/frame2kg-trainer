from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.utils import is_torch_available

try:
    from transformers import AutoModelForMultimodalLM
except Exception:  # pragma: no cover - depends on transformers version
    AutoModelForMultimodalLM = None

try:
    from transformers import Gemma4ForConditionalGeneration, Gemma4Processor
except Exception:  # pragma: no cover - depends on transformers version
    Gemma4ForConditionalGeneration = None
    Gemma4Processor = None

from frame2kg_train.backends.base import BackendArtifacts, VLMBackend
from frame2kg_train.backends.tokenizer_extension import (
    add_tokens_from_config,
    lora_config_kwargs,
    resize_model_embeddings_for_tokenizer,
)
from frame2kg_train.data.collators import Gemma4VLDataCollator


@dataclass
class Gemma4VLBackend(VLMBackend):
    name: str = "gemma4_vl"

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        model_id = cfg.get("model_id", "google/gemma-4-E2B-it")
        lora_cfg: Dict[str, Any] = cfg.get("lora", {})
        load_in_4bit = bool(cfg.get("load_in_4bit", torch.cuda.is_available()))
        attn_impl = cfg.get("attn_implementation", None)
        disable_thinking = bool(cfg.get("disable_thinking", True))

        self._ensure_gemma4_support(model_id)

        proc = self._load_processor(model_id=model_id)
        if proc.tokenizer.pad_token is None:
            proc.tokenizer.pad_token = proc.tokenizer.eos_token
        proc.tokenizer.padding_side = "left"
        tokenizer_extension = add_tokens_from_config(proc.tokenizer, cfg)

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
            target_modules = [str(name) for name in lora_cfg.get("target_modules", self.default_lora_target_modules())]
            available_targets = self._collect_target_module_names(model, target_modules)
            self._validate_lora_target_modules(model_id=model_id, target_modules=target_modules, matches=available_targets)
            self._log_target_module_summary("available", available_targets)
            resolved_target_modules = [
                module_name
                for target_name in target_modules
                for module_name in available_targets.get(target_name, [])
            ]
            print(f"[backend:{self.name}] resolved text-only LoRA module paths={len(resolved_target_modules)}")

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
            adapted_targets = self._collect_lora_wrapped_module_names(model, target_modules)
            self._validate_lora_target_modules(model_id=model_id, target_modules=target_modules, matches=adapted_targets)
            self._log_target_module_summary("adapted", adapted_targets)
            if hasattr(model, "print_trainable_parameters"):
                model.print_trainable_parameters()

        collator = Gemma4VLDataCollator(proc, disable_thinking=disable_thinking)
        return BackendArtifacts(model=model, tokenizer=proc.tokenizer, processor=proc, collator=collator)

    def _ensure_gemma4_support(self, model_id: str) -> None:
        if not is_torch_available():
            raise RuntimeError("PyTorch is required to load Gemma 4.")
        if AutoModelForMultimodalLM is None or Gemma4ForConditionalGeneration is None or Gemma4Processor is None:
            raise RuntimeError(
                f"This transformers build does not expose Gemma 4 multimodal classes needed for {model_id}. "
                "Install a Gemma 4-capable build, e.g. transformers>=5.5.3."
            )

    def _load_processor(self, model_id: str):
        kwargs = {"padding_side": "left"}
        try:
            return AutoProcessor.from_pretrained(model_id, **kwargs)
        except TypeError:
            return AutoProcessor.from_pretrained(model_id)
        except ValueError as e:
            msg = str(e).lower()
            if "gemma4" in msg and ("does not recognize" in msg or "unrecognized" in msg):
                raise RuntimeError(
                    "This transformers version does not recognize Gemma 4 yet. "
                    "Install a Gemma 4-capable build, e.g. transformers>=5.5.3."
                ) from e
            raise

    def _load_model(self, model_id: str, model_kwargs: Dict[str, Any]):
        if AutoModelForMultimodalLM is None:
            raise RuntimeError(
                "AutoModelForMultimodalLM is unavailable in this transformers build. "
                "Install a Gemma 4-capable build, e.g. transformers>=5.5.3."
            )

        try:
            return AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
        except ValueError as e:
            msg = str(e).lower()
            if "gemma4" in msg and ("does not recognize" in msg or "unrecognized" in msg):
                raise RuntimeError(
                    "This transformers version does not recognize Gemma 4 yet. "
                    "Install a Gemma 4-capable build, e.g. transformers>=5.5.3."
                ) from e
            raise
        except TypeError:
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs.pop("attn_implementation", None)
            return AutoModelForMultimodalLM.from_pretrained(model_id, **fallback_kwargs)

    def _set_pad_token_id(self, model: Any, pad_id: int | None) -> None:
        if pad_id is None:
            return
        model.generation_config.pad_token_id = pad_id
        if hasattr(model.config, "pad_token_id"):
            model.config.pad_token_id = pad_id
        text_config = getattr(model.config, "text_config", None)
        if text_config is not None and hasattr(text_config, "pad_token_id"):
            text_config.pad_token_id = pad_id

    def _collect_target_module_names(self, model: Any, target_modules: List[str]) -> Dict[str, List[str]]:
        matches = {name: [] for name in target_modules}
        for module_name, module in model.named_modules():
            for target_name in target_modules:
                if (module_name == target_name or module_name.endswith(f".{target_name}")) and hasattr(module, "weight"):
                    matches[target_name].append(module_name)
        return matches

    def _collect_lora_wrapped_module_names(self, model: Any, target_modules: List[str]) -> Dict[str, List[str]]:
        matches = {name: [] for name in target_modules}
        for module_name, module in model.named_modules():
            for target_name in target_modules:
                if (module_name == target_name or module_name.endswith(f".{target_name}")) and hasattr(module, "base_layer"):
                    matches[target_name].append(module_name)
        return matches

    def _validate_lora_target_modules(
        self,
        *,
        model_id: str,
        target_modules: List[str],
        matches: Dict[str, List[str]],
    ) -> None:
        missing = [name for name in target_modules if not matches.get(name)]
        if missing:
            raise RuntimeError(
                f"[backend:{self.name}] Gemma 4 LoRA targets were not found for model_id={model_id}: {missing}. "
                "The backend expects Gemma 4 qkvo plus gate/up/down projection names."
            )

    def _log_target_module_summary(self, label: str, matches: Dict[str, List[str]]) -> None:
        counts = {name: len(module_names) for name, module_names in matches.items()}
        print(f"[backend:{self.name}] {label} LoRA target counts={counts}")
        for name, module_names in matches.items():
            preview = ", ".join(module_names[:3])
            if len(module_names) > 3:
                preview = f"{preview}, ..."
            print(f"[backend:{self.name}] {label} {name}: {preview}")

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
        assert proc is not None, "Gemma 4 requires a processor"
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
