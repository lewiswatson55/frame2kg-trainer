# Frame2KG Trainer

Scaffold for Frame→KG training with pluggable VLM backends.

## Install

```bash
python -m venv .venv && source .venv/bin/activate && pip install -e .
```

## Usage

### Training

This scaffold includes a training entrypoint and example config for Qwen2.5-VL with LoRA. You must have access to the referenced model checkpoints.

```bash
python scripts/train.py --config configs/qwen25_lora.yaml
```

```bash
python scripts/train.py --config configs/qwen25_lora.yaml --overrides wandb.project=quicktest-blah
```

Environment variables:
- `WANDB_API_KEY` must be set if logging to Weights & Biases.

### Evaluation

For evaluation see the [Frame2KG Evaluation Toolkit](https://anonymous.4open.science/r/frame2kg_eval_toolkit/)

## Configuration and overrides

- All training/runtime options are driven by a YAML config (see `configs/qwen25_lora.yaml`).
- You can override any config value at the CLI using `--overrides key=value` pairs (supports dot-notation):

```bash
python scripts/train.py \
  --config configs/qwen25_lora.yaml \
  --overrides per_device_train_bs=2 lr=1e-4 eval_sample_k=5 wandb.project=my-project
```

Common keys:
- `backend`: which implementation to use (e.g., `qwen25_vl`).
- `model_id`: HF model identifier.
- `lora`: optional LoRA block with `r`, `alpha`, `dropout`, and `target_modules`.
- Training knobs: `epochs`, `lr`, `per_device_train_bs`, `grad_acc_steps`, `eval_steps`, `save_steps`, etc.
- Eval sampling: `eval_sample_k` and `eval_sample_mode` (`random` or `first`).
- WandB: `wandb.project` and `wandb.entity`.

## Project layout and responsibilities

- `frame2kg/backends/`
  - `base.py`: Defines the `VLMBackend` Protocol, `BackendArtifacts` container, and `Collator` signature used by the Trainer.
  - `qwen25_vl.py`: Working backend for Qwen2.5‑VL (loads model/processor, optional 4‑bit quant, optional LoRA, and provides a matching collator).
  - `blip2.py`, `florence2.py`: Placeholders showing how additional backends would slot in.

- `frame2kg/data/`
  - `datasets.py`: Loads the Frame2KG dataset from the HF Hub and ensures the `image` column is typed as `datasets.Image`.
  - `collators.py`: Qwen‑specific chat formatting (`QwenVLDataCollator`) that builds supervised chat samples and masks labels appropriately.

- `frame2kg/train/`
  - `args.py`: Safe construction of `Seq2SeqTrainingArguments` with sensible defaults for generation - bc version differences are pain incarnate.
  - `callbacks.py`:
    - `PeriodicEvalCallback`: Forces evaluation every N steps when using strategy "no".
    - `CustomWandbCallback`: Logs per‑example predictions, parsed JSON flag, and node/edge scores to WandB tables.
    - `SaveAdaptersCallback`: Saves LoRA adapters and (optionally) the processor; logs as WandB artifacts.
  - `registry.py`: Name → backend instance mapping via `get_backend(name)`.
  - `run.py`: `Runner` orchestrates loading the backend, datasets, metrics, Trainer, callbacks, training, evaluation, and artifact export.

- `scripts/train.py`: CLI entrypoint that loads YAML, applies overrides, and invokes `Runner`.
- `configs/*.yaml`: Example configs (Qwen2.5‑VL working example; BLIP‑2/Florence2 placeholders).
- Packaging: `pyproject.toml` and `requirements.txt`.

## Currently working out of the box

- A fully wired Qwen2.5‑VL backend with optional 4‑bit loading and LoRA adapters (via PEFT).
- A Qwen‑compatible collator that constructs chat prompts and masks labels for supervised fine‑tuning.
- A thin Trainer wrapper (`Runner`) with evaluation, periodic eval triggers, WandB logging, and adapter export.
- Example config for our Qwen2.5‑VL LoRA fine‑tuning.

## Adding a new backend (example workflow)

1) Implement the backend class
- Create `frame2kg/backends/<name>.py` with a class `<Name>Backend(VLMBackend)`.
- Implement:
  - `name`: string key used in configs (e.g., `"blip2"`).
  - `load(cfg) -> BackendArtifacts`: Load tokenizer/processor/model (and optional quant/LoRA), and return `BackendArtifacts(model, tokenizer, processor, collator)`.
  - `default_lora_target_modules() -> List[str]`: Reasonable defaults for your architecture (optional).
  - `generate_text(artifacts, image, prompt, max_new_tokens) -> str`: Single‑sample generation for logging/eval.

2) Provide a collator
- If a family needs bespoke formatting, add a collator under `frame2kg/data/` implementing the `Collator` protocol (callable returning a dict of PyTorch tensors, including `labels` for training). Otherwise, reuse an existing collator if compatible.

3) Register the backend
- Add your instance to `frame2kg/train/registry.py`:
  - `"my_backend": MyBackend(),`

4) Add a config
- Create `configs/my_backend_lora.yaml` (or similar) with keys for `backend`, `model_id`, optional `lora`, and training hyperparameters.

5) Run training
```bash
python scripts/train.py --config configs/my_backend_lora.yaml \
  --overrides run_name=my-backend-test eval_sample_k=3
```

Tips:
- Run names are automatically suffixed with a UTC timestamp to keep repeated runs distinct on Weights & Biases.
- For quantised loading, use BitsAndBytes on Linux (`bitsandbytes` is optional and currently is only installed on Linux in `pyproject.toml`).
- For LoRA, use `peft.prepare_model_for_kbit_training` when training in 4‑bit.
- Ensure your tokenizer has a `pad_token` and set `padding_side` appropriately (often `left` for generation‑based training).

## System requirements and setup notes

- GPU is strongly recommended. The Qwen2.5‑VL backend can run in 4‑bit on CUDA. On macOS, MPS is used automatically for non‑quantized paths.
- Authenticate for gated resources:
  - Hugging Face Hub: `huggingface-cli login` or set `HF_TOKEN`.
  - Weights & Biases: set `WANDB_API_KEY`.
- If you experience out‑of‑memory issues, lower `per_device_train_bs`, raise `grad_acc_steps`, or reduce `max_new_tokens`.

## Notes
- `blip2` and `florence2` backends are placeholders and raise NotImplementedError.
- Dataset loader expects approved access to the gated dataset `lewiswatson/Frame2KG-YC2` on the Hugging Face Hub.
