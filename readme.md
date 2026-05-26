# Frame2KG Trainer

Scaffold for Frame→KG training with pluggable VLM backends.

## Install

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

## Usage

### Training

This scaffold includes a training entrypoint and example configs for Qwen (2.5/3/3.5-compatible backend), InternVL3.5, Gemma 4, Llama 3.2 Vision, LFM2.5-VL, SmolVLM2, and Transformers-native FastVLM conversion checkpoints with LoRA. You must have access to the referenced model checkpoints.

```bash
python scripts/train.py --config configs/qwen25vl/qwen25_lora_a100.yaml
```

```bash
python scripts/train.py --config configs/qwen35vl/qwen35_08b_lora.yaml
```

```bash
python scripts/train.py --config configs/gemma4/gemma4_e2b_lora_nodes-first.yaml
```

```bash
python scripts/train.py --config configs/internvl35/internvl35_1b_lora_nodes-first.yaml
```

```bash
python scripts/train.py --config configs/llama32vision/llama32_vision_11b_lora_nodes-first.yaml
```

```bash
python scripts/train.py --config configs/lfm25vl/lfm25_vl_450m_lora_nodes-first.yaml
```

```bash
python scripts/train.py --config configs/smolvlm2/smolvlm2_500m_lora.yaml --overrides wandb.project=quicktest-blah
```

```bash
python scripts/train.py --config configs/fastvlm/fastvlm_hf-native_0-5b_lora_nodes-first.yaml
```

Environment variables:
- `WANDB_API_KEY` must be set if logging to Weights & Biases.

## Reproducibility

Reproducibility in both research and implementation is essential. To support this, we ask that any published models include the training `.yaml` configuration file to improve traceability. An example can be found [here](https://huggingface.co/lewiswatson/frame2kg-qwen2_5vl-3b-qkvo-lora/blob/main/frame2kg_training_config.yaml).

In addition, the otherwise unused `training_repo` key should be updated to point to a fork or repository containing your changes.

### Evaluation

For evaluation see the [Frame2KG Evaluation Toolkit](https://anonymous.4open.science/r/frame2kg_eval_toolkit/)

## Configuration and overrides

- All training/runtime options are driven by a YAML config (see `configs/qwen25vl/qwen25_lora_a100.yaml`, `configs/gemma4/gemma4_e2b_lora_nodes-first.yaml`, `configs/internvl35/internvl35_1b_lora_nodes-first.yaml`, `configs/llama32vision/llama32_vision_11b_lora_nodes-first.yaml`, `configs/lfm25vl/lfm25_vl_450m_lora_nodes-first.yaml`, `configs/smolvlm2/smolvlm2_500m_lora.yaml`, or `configs/fastvlm/fastvlm_hf-native_0-5b_lora_nodes-first.yaml`).
- You can override any config value at the CLI using `--overrides key=value` pairs (supports dot-notation):

```bash
python scripts/train.py \
  --config configs/smolvlm2/smolvlm2_500m_lora.yaml \
  --overrides per_device_train_bs=2 lr=1e-4 eval_sample_k=5 wandb.project=my-project
```

To train on nodes-first graph JSON without changing the dataset:

```bash
python scripts/train.py \
  --config configs/qwen25vl/qwen25_lora_a100.yaml \
  --overrides graph_key_order=nodes_first
```

Common keys:
- `backend`: which implementation to use (e.g., `qwen25_vl`, `qwen3_vl`, `qwen35_vl`, `internvl35_vl`, `gemma4_vl`, `llama32_vision`, `lfm25_vl`, `smolvlm2`, or `fastvlm`).
- `model_id`: HF model identifier.
- `load_in_4bit`: optional 4-bit loading path (CUDA only).
- `attn_implementation`: optional attention backend override (for example `flash_attention_2`).
- `disable_thinking`: when true (default), best-effort disables reasoning tokens for model families whose chat templates support it.
- `lora`: optional LoRA block with `r`, `alpha`, `dropout`, and `target_modules`.
- `graph_key_order`: target JSON top-level key order. Defaults to `dataset`; use `nodes_first` to train on `{"nodes":...,"edges":...}` even when the dataset stores edges first.
- Training knobs: `epochs`, `lr`, `per_device_train_bs`, `grad_acc_steps`, `eval_steps`, `save_steps`, etc.
- Eval sampling: `eval_sample_k` and `eval_sample_mode` (`random` or `first`).
- WandB: `wandb.project` and `wandb.entity`.

## Project layout and responsibilities

- `frame2kg_train/backends/`
  - `base.py`: Defines the `VLMBackend` Protocol, `BackendArtifacts` container, and `Collator` signature used by the Trainer.
  - `fastvlm.py`: Transformers-native FastVLM conversion backend for `KamilaMila/FastVLM-0.5B` / `KamilaMila/FastVLM-1.5B` / `KamilaMila/FastVLM-7B` (via AutoProcessor + FastVlmForConditionalGeneration), with optional 4-bit quant, decoder-scoped attention overrides, and text-decoder-scoped LoRA.
  - `gemma4_vl.py`: Gemma 4 backend for `google/gemma-4-E2B-it` / `google/gemma-4-E4B-it` (via AutoProcessor + AutoModelForMultimodalLM), with optional 4-bit quant, optional LoRA, and Gemma-specific target validation.
  - `internvl35_vl.py`: InternVL3.5 backend for `OpenGVLab/InternVL3_5-1B-HF` / `OpenGVLab/InternVL3_5-2B-HF` / `OpenGVLab/InternVL3_5-4B-HF` (via AutoProcessor + AutoModelForImageTextToText), with optional 4-bit quant, text-decoder-scoped LoRA, and InternVL image-patch handling.
  - `internvl35_targets.py`: Exact InternVL3.5 text-decoder LoRA target resolution for Qwen3 language-model layers.
  - `llama32_vision.py`: Llama 3.2 Vision backend for `meta-llama/Llama-3.2-11B-Vision-Instruct` (via AutoProcessor + MllamaForConditionalGeneration), with optional 4-bit quant, text-decoder-scoped LoRA, and Mllama-specific chat/image handling.
  - `lfm25_vl.py`: LFM2.5-VL backend for `LiquidAI/LFM2.5-VL-450M`-style checkpoints (via AutoProcessor + AutoModelForImageTextToText), with optional 4-bit quant, optional LoRA, and an LFM-specific collator.
  - `qwen25_vl.py`: Qwen backend for Qwen2.5‑VL / Qwen3‑VL / Qwen3.5-family checkpoints (via AutoProcessor + AutoModelForImageTextToText), with optional 4‑bit quant, optional LoRA, and a matching collator.
  - `smolvlm2.py`: SmolVLM2 backend (AutoProcessor + AutoModelForImageTextToText, optional 4-bit quant, optional LoRA, and Smol-specific generation).

- `frame2kg_train/data/`
  - `datasets.py`: Loads the Frame2KG dataset from the HF Hub and ensures the `image` column is typed as `datasets.Image`.
  - `collators.py`: Qwen, InternVL3.5, Gemma 4, Llama 3.2 Vision, LFM, Smol, and FastVLM chat formatting (`QwenVLDataCollator`, `InternVL35DataCollator`, `Gemma4VLDataCollator`, `Llama32VisionDataCollator`, `LFMVLDataCollator`, `SmolVLMDataCollator`, `FastVLMDataCollator`) that build supervised chat samples and mask labels appropriately.

- `frame2kg_train/train/`
  - `args.py`: Safe construction of `Seq2SeqTrainingArguments` with sensible defaults for generation - bc version differences are pain incarnate.
  - `callbacks.py`:
    - `PeriodicEvalCallback`: Forces evaluation every N steps when using strategy "no".
    - `CustomWandbCallback`: Logs per‑example predictions, parsed JSON flag, and node/edge scores to WandB tables.
    - `SaveAdaptersCallback`: Saves LoRA adapters and (optionally) the processor; logs as WandB artifacts.
  - `registry.py`: Name → backend instance mapping via `get_backend(name)`.
  - `run.py`: `Runner` orchestrates loading the backend, datasets, metrics, Trainer, callbacks, training, evaluation, and artifact export.

- `scripts/train.py`: CLI entrypoint that loads YAML, applies overrides, and invokes `Runner`.
- `configs/<model-type>/*.yaml`: Example configs grouped by model family (`qwen25vl`, `qwen3vl`, `qwen35vl`, `internvl35`, `gemma4`, `llama32vision`, `lfm25vl`, `smolvlm2`, `fastvlm`).
- Packaging: `pyproject.toml` and `requirements.txt`.

## Currently working out of the box

- A fully wired Qwen2.5‑VL backend with optional 4‑bit loading and LoRA adapters (via PEFT).
- A fully wired Qwen3‑VL backend with optional 4‑bit loading and LoRA adapters (via PEFT).
- A fully wired InternVL3.5 backend for 1B/2B/4B HF checkpoints with optional 4-bit loading and text-decoder-scoped LoRA adapters (via PEFT).
- A fully wired Gemma 4 backend for E2B/E4B with optional 4-bit loading and LoRA adapters (via PEFT).
- A fully wired Llama 3.2 Vision backend for 11B Vision Instruct with optional 4-bit loading and text-decoder-scoped LoRA adapters (via PEFT).
- A fully wired LFM2.5‑VL backend with optional 4‑bit loading and LoRA adapters (via PEFT).
- A fully wired SmolVLM2 backend with optional 4‑bit loading and LoRA adapters (via PEFT).
- A fully wired Transformers-native FastVLM conversion backend for KamilaMila 0.5B/1.5B/7B checkpoints with optional 4-bit loading and text-decoder-scoped LoRA adapters (via PEFT).
- A Qwen‑compatible collator that constructs chat prompts and masks labels for supervised fine‑tuning.
- An InternVL3.5-compatible collator that uses InternVL image placeholders, processor patch expansion, and the same JSON extraction prompt contract.
- A Gemma 4-compatible collator that preserves the same JSON extraction prompt contract and masks multimodal special tokens.
- A Llama 3.2 Vision-compatible collator that uses Mllama image placeholders, passes images separately to the processor, and masks prompt/special tokens for supervised fine-tuning.
- An LFM-compatible collator that uses the same JSON extraction prompt contract and masking scheme.
- A SmolVLM-compatible collator that preserves the same system/user prompt contract and masking scheme.
- A FastVLM-compatible collator that uses multimodal `apply_chat_template` tokenization directly and masks prompt, padding, image, and special tokens.
- A thin Trainer wrapper (`Runner`) with evaluation, periodic eval triggers, WandB logging, and adapter export.
- Example configs for Qwen2.5‑VL, InternVL3.5, Llama 3.2 Vision, LFM2.5‑VL, SmolVLM2, and Transformers-native FastVLM LoRA fine‑tuning.

## Adding a new backend (example workflow)

1) Implement the backend class
- Create `frame2kg_train/backends/<name>.py` with a class `<Name>Backend(VLMBackend)`.
- Implement:
  - `name`: string key used in configs (e.g., `"my_backend"`).
  - `load(cfg) -> BackendArtifacts`: Load tokenizer/processor/model (and optional quant/LoRA), and return `BackendArtifacts(model, tokenizer, processor, collator)`.
  - `default_lora_target_modules() -> List[str]`: Reasonable defaults for your architecture (optional).
  - `generate_text(artifacts, image, prompt, max_new_tokens) -> str`: Single‑sample generation for logging/eval.

2) Provide a collator
- If a family needs bespoke formatting, add a collator under `frame2kg_train/data/` implementing the `Collator` protocol (callable returning a dict of PyTorch tensors, including `labels` for training). Otherwise, reuse an existing collator if compatible.

3) Register the backend
- Add your backend class mapping to `frame2kg_train/train/registry.py`.

4) Add a config
- Create `configs/<model-type>/my_backend_lora.yaml` (or similar) with keys for `backend`, `model_id`, optional `lora`, and training hyperparameters.

5) Run training
```bash
python scripts/train.py --config configs/<model-type>/my_backend_lora.yaml \
  --overrides run_name=my-backend-test eval_sample_k=3
```

Tips:
- Run names are automatically suffixed with a UTC timestamp to keep repeated runs distinct on Weights & Biases.
- For quantised loading, use BitsAndBytes on Linux (`bitsandbytes` is optional and currently is only installed on Linux in `pyproject.toml`).
- For LoRA, use `peft.prepare_model_for_kbit_training` when training in 4‑bit.
- Ensure your tokenizer has a `pad_token` and set `padding_side` appropriately (often `left` for generation‑based training).

## System requirements and setup notes

- GPU is strongly recommended. The Qwen2.5‑VL backend can run in 4‑bit on CUDA. On macOS, MPS is used automatically for non‑quantized paths.
- The current repo pin is `transformers==5.9.0` with `peft==0.19.1`, which covers Gemma 4, SmolVLM2, LFM2.5-VL, and native FastVLM support. SmolVLM2 also requires `num2words`.
- The `fastvlm` backend is explicitly for the Transformers-native FastVLM conversion checkpoints under `KamilaMila/FastVLM-*`. It does not claim official Apple checkpoint provenance; official `apple/FastVLM-*` checkpoints would require a separate remote-code backend.
- Llama 3.2 Vision uses the Mllama classes exposed by Transformers and requires approved access to `meta-llama/Llama-3.2-11B-Vision-Instruct` on Hugging Face.
- InternVL3.5-HF uses `AutoModelForImageTextToText` with `trust_remote_code: true`. Use the `*-HF` checkpoints with `internvl35_vl`; the `*-Instruct` and bare InternVL3.5 repos expose the older `internvl_chat`/`AutoModel` interface instead. Official HF-format checkpoints currently include 1B, 2B, 4B, 8B, 14B, 30B-A3B, 38B, 241B-A28B, and GPT-OSS-20B-A4B variants; there is not an official `OpenGVLab/InternVL3_5-3B-HF` checkpoint.
- Gemma 4 support relies on `AutoModelForMultimodalLM`; older Transformers builds will fail to load Gemma 4 checkpoints.
- Some Qwen3.5 checkpoints currently require a very recent `transformers` build. If loading fails on model type `qwen3_5`, install from source:
  - `pip install git+https://github.com/huggingface/transformers.git`
- Authenticate for gated resources:
  - Hugging Face Hub: `huggingface-cli login` or set `HF_TOKEN`.
  - Weights & Biases: set `WANDB_API_KEY`.
- If you experience out‑of‑memory issues, lower `per_device_train_bs`, raise `grad_acc_steps`, or reduce `max_new_tokens`.

## Notes
- Dataset loader expects approved access to the gated dataset `lewiswatson/Frame2KG-YC2` on the Hugging Face Hub.

## Cite

```bibtex
@inproceedings{watson2026frame2kg,
  title = {Frame2KG: A Benchmark and Evaluation Toolkit for Interpretable Frame-to-Graph Generation},
  author = {Watson, Lewis N. and Strathearn, Carl and Mitchell, Kenny and Yu, Yanchao},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  month = {May},
  year = {2026},
  pages = {10912--10926},
  address = {Palma, Mallorca, Spain},
  publisher = {European Language Resources Association (ELRA)},
  editor = {Piperidis, Stelios and Bel, Núria and van den Heuvel, Henk and Ide, Nancy and Krek, Simon and Toral, Antonio},
  doi = {10.63317/4ys6kofrzoc5},
  url = {https://doi.org/10.63317/4ys6kofrzoc5}
```