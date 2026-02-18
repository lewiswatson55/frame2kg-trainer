from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from frame2kg_train.backends.base import BackendArtifacts, VLMBackend
from frame2kg_train.data.collators import MoondreamDataCollator


class MoondreamProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _extract_text(self, message: Dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    txt = part.get("text", "")
                    if txt:
                        chunks.append(str(txt))
            return "\n".join(chunks)
        return ""

    def apply_chat_template(
        self,
        conversation: List[Dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ):
        system_texts: List[str] = []
        user_texts: List[str] = []
        assistant_text: str | None = None

        for message in conversation:
            role = str(message.get("role", "")).lower()
            text = self._extract_text(message).strip()
            if not text:
                continue
            if role == "system":
                system_texts.append(text)
            elif role == "user":
                user_texts.append(text)
            elif role == "assistant":
                assistant_text = text

        question_parts = []
        if system_texts:
            question_parts.append("\n".join(system_texts))
        if user_texts:
            question_parts.append("\n".join(user_texts))
        question = "\n\n".join(question_parts).strip() or "Describe this image."

        prompt = f"<image>\n\nQuestion: {question}\n\nAnswer:"
        if assistant_text and not add_generation_prompt:
            prompt = f"{prompt}{assistant_text}"

        if tokenize:
            return self.tokenizer(prompt, return_tensors="pt")
        return prompt

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        self.tokenizer.save_pretrained(save_directory)
        config_path = os.path.join(save_directory, "processor_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"processor_class": "MoondreamProcessor"}, f, ensure_ascii=False, indent=2)


class Moondream2TrainWrapper(nn.Module):
    def __init__(self, core_model: nn.Module, tokenizer, model_id: str, revision: str | None):
        super().__init__()
        self.core_model = core_model
        self.text_model = getattr(core_model, "text_model", None)
        if self.text_model is None:
            raise RuntimeError(
                "Loaded Moondream model has no `text_model` attribute. "
                "Use revision=2024-08-26 for LoRA training in this backend."
            )
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.revision = revision
        self.config = getattr(core_model, "config", None)
        self.generation_config = getattr(self.text_model, "generation_config", None)
        for attr in ("is_loaded_in_4bit", "is_loaded_in_8bit", "hf_device_map"):
            if hasattr(core_model, attr):
                setattr(self, attr, getattr(core_model, attr))

        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        if bos_id is None and eos_id is None:
            raise RuntimeError("Tokenizer must define at least one of bos_token_id or eos_token_id.")
        self.bos_token_id = int(bos_id if bos_id is not None else eos_id)
        self.pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.bos_token_id)

    @property
    def device(self) -> torch.device:
        if hasattr(self.core_model, "device"):
            return self.core_model.device
        return next(self.parameters()).device

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Dict[str, Any] | None = None):
        if hasattr(self.text_model, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is None:
                self.text_model.gradient_checkpointing_enable()
            else:
                self.text_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )

    def gradient_checkpointing_disable(self):
        if hasattr(self.text_model, "gradient_checkpointing_disable"):
            self.text_model.gradient_checkpointing_disable()

    def enable_input_require_grads(self):
        if hasattr(self.text_model, "enable_input_require_grads"):
            self.text_model.enable_input_require_grads()

    def _token_ids(self, text: str) -> List[int]:
        if not text:
            return []
        encoded = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    @staticmethod
    def _split_image_token(prompt: str) -> tuple[str, str]:
        if "<image>" not in prompt:
            return "", prompt
        before, after = prompt.split("<image>", 1)
        return before, after

    def _encode_image(self, image: Image.Image | str):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        with torch.no_grad():
            image_embeds = self.core_model.encode_image(image)
        if not torch.is_tensor(image_embeds):
            raise RuntimeError(
                "This Moondream revision does not return tensor image embeddings. "
                "Use revision=2024-08-26 for LoRA training."
            )
        if image_embeds.ndim == 2:
            image_embeds = image_embeds.unsqueeze(0)
        if image_embeds.ndim != 3:
            raise RuntimeError(f"Unexpected image embedding shape: {tuple(image_embeds.shape)}")
        return image_embeds

    def _build_sample(
        self,
        image: Image.Image | str,
        prompt_text: str,
        answer_text: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_embeds = self._encode_image(image)

        embed_layer = self.text_model.get_input_embeddings()
        dev = embed_layer.weight.device
        dtype = embed_layer.weight.dtype
        image_embeds = image_embeds.to(device=dev, dtype=dtype)

        before_prompt, after_prompt = self._split_image_token(prompt_text)
        before_ids = self._token_ids(before_prompt)
        prompt_after_ids = self._token_ids(after_prompt)

        answer_ids: List[int] = []
        if answer_text is not None:
            answer_ids = self._token_ids(answer_text)
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and (not answer_ids or answer_ids[-1] != int(eos_id)):
                answer_ids.append(int(eos_id))

        after_ids = prompt_after_ids + answer_ids
        pieces = []

        bos = torch.tensor([[self.bos_token_id]], device=dev)
        pieces.append(embed_layer(bos))

        if before_ids:
            before = torch.tensor([before_ids], device=dev)
            pieces.append(embed_layer(before))

        pieces.append(image_embeds)

        if after_ids:
            after = torch.tensor([after_ids], device=dev)
            pieces.append(embed_layer(after))

        embeds = torch.cat(pieces, dim=1).squeeze(0)
        attn = torch.ones((embeds.shape[0],), dtype=torch.long, device=dev)
        labels = torch.full((embeds.shape[0],), -100, dtype=torch.long, device=dev)
        if answer_ids:
            labels[-len(answer_ids):] = torch.tensor(answer_ids, dtype=torch.long, device=dev)
        return embeds, attn, labels

    @staticmethod
    def _pad_token_batch(
        rows: Sequence[Sequence[int]],
        pad_id: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not rows:
            return torch.empty((0, 0), dtype=torch.long, device=device)
        max_len = max(len(r) for r in rows)
        out = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=device)
        for i, row in enumerate(rows):
            if row:
                out[i, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        return out

    @staticmethod
    def _strip_answer_prefix(text: str) -> str:
        marker = "Answer:"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return text.strip()

    def forward(
        self,
        images=None,
        prompt_texts=None,
        answer_texts=None,
        labels=None,
        **kwargs,
    ):
        allowed_text_kwargs = {
            "position_ids",
            "past_key_values",
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "cache_position",
            "num_logits_to_keep",
        }
        model_kwargs = {k: v for k, v in kwargs.items() if k in allowed_text_kwargs}

        if images is None or prompt_texts is None:
            return self.text_model(labels=labels, **model_kwargs)

        prompt_list = list(prompt_texts)
        image_list = list(images)
        if len(prompt_list) != len(image_list):
            raise ValueError("Batch mismatch: `images` and `prompt_texts` must have equal length.")

        if answer_texts is None:
            answer_list = [None] * len(prompt_list)
        else:
            answer_list = [str(x) if x is not None else None for x in list(answer_texts)]
            if len(answer_list) != len(prompt_list):
                raise ValueError("Batch mismatch: `answer_texts` must match `prompt_texts` length.")

        built = [self._build_sample(img, prm, ans) for img, prm, ans in zip(image_list, prompt_list, answer_list)]
        max_len = max(t[0].shape[0] for t in built)
        hidden = built[0][0].shape[1]
        dev = built[0][0].device
        dtype = built[0][0].dtype

        input_embeds = torch.zeros((len(built), max_len, hidden), dtype=dtype, device=dev)
        attn_mask = torch.zeros((len(built), max_len), dtype=torch.long, device=dev)
        label_tensor = torch.full((len(built), max_len), -100, dtype=torch.long, device=dev)

        for i, (emb, attn, lab) in enumerate(built):
            n = emb.shape[0]
            input_embeds[i, :n] = emb
            attn_mask[i, :n] = attn
            label_tensor[i, :n] = lab

        use_labels = any(ans is not None for ans in answer_list)
        return self.text_model(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            labels=label_tensor if use_labels else None,
            use_cache=False,
            **model_kwargs,
        )

    @torch.inference_mode()
    def generate(
        self,
        images=None,
        prompt_texts=None,
        answer_texts=None,
        max_new_tokens: int | None = None,
        **kwargs,
    ):
        if images is None or prompt_texts is None:
            if max_new_tokens is not None:
                kwargs["max_new_tokens"] = int(max_new_tokens)
            return self.text_model.generate(**kwargs)

        _ = answer_texts  # ignored for generation
        max_new = int(max_new_tokens or kwargs.get("max_new_tokens") or kwargs.get("max_length") or 256)
        prompts = list(prompt_texts)
        image_list = list(images)
        if len(prompts) != len(image_list):
            raise ValueError("Batch mismatch: `images` and `prompt_texts` must have equal length.")

        rows: List[List[int]] = []
        for img, prompt in zip(image_list, prompts):
            image_embeds = self._encode_image(img)
            text = self.core_model.generate(
                image_embeds=image_embeds,
                prompt=str(prompt),
                tokenizer=self.tokenizer,
                max_new_tokens=max_new,
                do_sample=False,
            )[0]
            text = self._strip_answer_prefix(str(text))
            tok_ids = self._token_ids(text)
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None and (not tok_ids or tok_ids[-1] != int(eos_id)):
                tok_ids.append(int(eos_id))
            rows.append(tok_ids)

        return self._pad_token_batch(rows=rows, pad_id=self.pad_token_id, device=self.device)

    @torch.inference_mode()
    def generate_text(self, image, prompt: str, max_new_tokens: int) -> str:
        image_embeds = self._encode_image(image)
        text = self.core_model.generate(
            image_embeds=image_embeds,
            prompt=prompt,
            tokenizer=self.tokenizer,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )[0]
        return self._strip_answer_prefix(str(text))

    def save_pretrained(self, save_directory: str, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.text_model.save_pretrained(save_directory, **kwargs)
        self.tokenizer.save_pretrained(save_directory)
        with open(os.path.join(save_directory, "moondream_backend.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"model_id": self.model_id, "revision": self.revision, "backend": "moondream2"},
                f,
                ensure_ascii=False,
                indent=2,
            )

    def merge_and_unload(self):
        if not hasattr(self.text_model, "merge_and_unload"):
            raise AttributeError("merge_and_unload is only available for LoRA-wrapped models.")
        merged_text = self.text_model.merge_and_unload()
        self.core_model.text_model = merged_text
        self.text_model = merged_text
        self.generation_config = getattr(self.text_model, "generation_config", None)
        return self


@dataclass
class Moondream2Backend(VLMBackend):
    name: str = "moondream2"

    def load(self, cfg: Dict[str, Any]) -> BackendArtifacts:
        model_id = cfg.get("model_id", "vikhyatk/moondream2")
        revision_raw = cfg.get("revision", "2024-08-26")
        if revision_raw is None:
            revision = "2024-08-26"
        elif isinstance(revision_raw, str):
            revision = revision_raw
        elif hasattr(revision_raw, "isoformat"):
            revision = str(revision_raw.isoformat())
        else:
            revision = str(revision_raw)
        trust_remote_code = bool(cfg.get("trust_remote_code", True))
        attn_impl = cfg.get("attn_implementation", None)
        lora_cfg: Dict[str, Any] = cfg.get("lora", {})
        load_in_4bit = bool(cfg.get("load_in_4bit", False))

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
        tokenizer.padding_side = "left"

        model_kwargs: Dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
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
            core_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            core_model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

        if not hasattr(core_model, "text_model"):
            raise RuntimeError(
                "Moondream backend currently supports trainable revision 2024-08-26 "
                "(models exposing `text_model`)."
            )

        for p in core_model.parameters():
            p.requires_grad = False

        text_model = core_model.text_model
        if hasattr(text_model.config, "use_cache"):
            text_model.config.use_cache = False
        if hasattr(core_model.config, "use_cache"):
            core_model.config.use_cache = False

        if lora_cfg:
            if load_in_4bit and torch.cuda.is_available():
                text_model = prepare_model_for_kbit_training(text_model)
            elif hasattr(text_model, "enable_input_require_grads"):
                text_model.enable_input_require_grads()

            peft_cfg = LoraConfig(
                r=int(lora_cfg.get("r", 8)),
                lora_alpha=int(lora_cfg.get("alpha", 16)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                bias="none",
                target_modules=lora_cfg.get("target_modules", self.default_lora_target_modules()),
                task_type="CAUSAL_LM",
            )
            text_model = get_peft_model(text_model, peft_cfg)
            core_model.text_model = text_model

        wrapper = Moondream2TrainWrapper(
            core_model=core_model,
            tokenizer=tokenizer,
            model_id=model_id,
            revision=revision,
        )
        wrapper.text_model = text_model
        wrapper.generation_config = getattr(text_model, "generation_config", None)
        if hasattr(text_model, "gradient_checkpointing_enable"):
            text_model.gradient_checkpointing_enable()

        processor = MoondreamProcessor(tokenizer)
        collator = MoondreamDataCollator(processor)
        return BackendArtifacts(model=wrapper, tokenizer=tokenizer, processor=processor, collator=collator)

    def default_lora_target_modules(self) -> List[str]:
        return ["Wqkv", "out_proj", "fc1", "fc2"]

    @torch.inference_mode()
    def generate_text(self, artifacts: BackendArtifacts, image, prompt: str, max_new_tokens: int) -> str:
        model = artifacts.model
        if hasattr(model, "generate_text"):
            return model.generate_text(image=image, prompt=prompt, max_new_tokens=max_new_tokens)
        raise RuntimeError("Moondream model wrapper does not expose generate_text().")
