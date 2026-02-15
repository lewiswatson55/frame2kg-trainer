from __future__ import annotations
import os
import inspect
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
from datasets import Dataset
from transformers import Seq2SeqTrainer, set_seed, EarlyStoppingCallback

from frame2kg_train.train.args import build_seq2seq_training_args
from frame2kg_train.train.callbacks import PeriodicEvalCallback, CustomWandbCallback, SaveAdaptersCallback
from frame2kg_train.train.registry import get_backend
from frame2kg_train.data.datasets import load_frame2kg
from frame2kg_train.eval.metrics import make_compute_metrics
from frame2kg_train.data.collators import SYSTEM_PROMPT


class Runner:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(self) -> None:
        import wandb
        seed = int(self.cfg.get("seed", 42))
        set_seed(seed); np.random.seed(seed)
        model_id = self.cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        run_ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        base_run_name = self.cfg.get("run_name", "frame2kg")
        run_name = f"{base_run_name}-{run_ts}"
        out_dir = os.path.abspath(self.cfg.get("output_dir", "../output"))
        run_dir = os.path.join(out_dir, run_name)
        ckpt_dir = os.path.join(run_dir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
        final_dir = os.path.join(run_dir, "final"); os.makedirs(final_dir, exist_ok=True)

        wandb.init(project=self.cfg["wandb"]["project"], entity=self.cfg["wandb"].get("entity"), name=run_name, config=self.cfg)

        backend_name = self.cfg.get("backend", "qwen25_vl")
        backend = get_backend(backend_name)
        artifacts = backend.load(self.cfg)

        compute_metrics = make_compute_metrics(artifacts.tokenizer)

        train_ds, eval_ds, test_ds = load_frame2kg(seed)
        if eval_ds is None: raise RuntimeError("Validation split not found.")
        # subset for quick eval
        k = int(self.cfg.get("eval_sample_k", 3))
        mode = str(self.cfg.get("eval_sample_mode", "random")).lower()
        if k > 0:
            k = min(k, len(eval_ds))
            if mode == "first":
                idx = list(range(k))
            else:
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(eval_ds), size=k, replace=False).tolist()
            eval_ds = eval_ds.select(idx)
        # tag modes for collators
        train_ds = train_ds.add_column("mode", ["train"] * len(train_ds))
        eval_ds = eval_ds.add_column("mode", ["eval"] * len(eval_ds))
        # Attach pre‑stringified GT for metrics
        import json as _json
        def _to_json(x): return x if isinstance(x,str) else _json.dumps(x, ensure_ascii=False, separators=(",", ":"))
        compute_metrics._eval_label_texts = [_to_json(r["graph"]) for r in eval_ds]  # type: ignore[attr-defined]

        # Trainer bits
        max_new = int(self.cfg.get("max_new_tokens", 4098))
        best_metric = str(self.cfg.get("early_stopping_metric", "edge_F1"))
        common = {
            "output_dir": ckpt_dir,
            "run_name": run_name,
            "learning_rate": float(self.cfg.get("lr", 5e-5)),
            "num_train_epochs": int(self.cfg.get("epochs", 3)),
            "per_device_train_batch_size": int(self.cfg.get("per_device_train_bs", 1)),
            "per_device_eval_batch_size": int(self.cfg.get("per_device_eval_bs", 2)),
            "gradient_accumulation_steps": int(self.cfg.get("grad_acc_steps", 8)),
            "warmup_ratio": float(self.cfg.get("warmup_ratio", 0.03)),
            "weight_decay": float(self.cfg.get("weight_decay", 0.01)),
            "logging_dir": os.path.join(run_dir, "logs"),
            "logging_steps": int(self.cfg.get("logging_steps", 10)),
            "logging_first_step": True,
            "save_steps": int(self.cfg.get("save_steps", 20)),
            "save_total_limit": int(self.cfg.get("save_total_limit", 2)),
            "load_best_model_at_end": True,
            "metric_for_best_model": best_metric,
            "greater_is_better": True,
            "bf16": bool(self.cfg.get("bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported())),
            "fp16": bool(self.cfg.get("fp16", torch.cuda.is_available() and not torch.cuda.is_bf16_supported())),
            "report_to": ["wandb"],
            "remove_unused_columns": False,
            "seed": seed,
            "data_seed": seed,
            "dataloader_num_workers": int(self.cfg.get("num_workers", 0 if os.name=="nt" else 4)),
            "dataloader_pin_memory": True,
            "max_grad_norm": float(self.cfg.get("max_grad_norm", 1.0)),
            "gradient_checkpointing": True,
            "eval_accumulation_steps": int(self.cfg.get("eval_accumulation_steps", 4)),
            "eval_steps": int(self.cfg.get("eval_steps", 20)),
        }
        training_args = build_seq2seq_training_args(common, max_new_tokens=max_new)

        periodic_eval = PeriodicEvalCallback(every_steps=int(self.cfg.get("eval_steps", 20)))
        wandb_cb = CustomWandbCallback(backend=backend, artifacts=artifacts, eval_ds=eval_ds, system_prompt=SYSTEM_PROMPT, max_new_tokens=max_new)
        save_adapters_cb = SaveAdaptersCallback(proc=artifacts.processor, max_new_tokens=max_new)

        callbacks = [periodic_eval, wandb_cb, save_adapters_cb]
        patience = int(self.cfg.get("early_stopping_patience", 0))
        if patience > 0:
            threshold = float(self.cfg.get("early_stopping_threshold", 0.0))
            callbacks.append(EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=threshold,
            ))

        trainer_kwargs = {
            "model": artifacts.model,
            "args": training_args,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "data_collator": artifacts.collator,
            "compute_metrics": compute_metrics,
            "callbacks": callbacks,
        }
        trainer_init_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
        if "processing_class" in trainer_init_params:
            trainer_kwargs["processing_class"] = artifacts.processor or artifacts.tokenizer
        else:
            trainer_kwargs["tokenizer"] = artifacts.tokenizer

        trainer = Seq2SeqTrainer(**trainer_kwargs)

        print("Starting fine‑tuning …")
        trainer.train()
        trainer.save_state()

        # Save adapters + processor
        adapters_dir = os.path.join(final_dir, "adapters"); os.makedirs(adapters_dir, exist_ok=True)
        artifacts.model.save_pretrained(adapters_dir)
        if artifacts.processor is not None:
            artifacts.processor.save_pretrained(adapters_dir)

        # Final eval
        metrics = trainer.evaluate()
        metrics = {f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
        wandb.log(metrics)

        # Optional merge
        if bool(self.cfg.get("export_merged", False)):
            try:
                merged_dir = os.path.join(final_dir, "merged"); os.makedirs(merged_dir, exist_ok=True)
                merged = artifacts.model.merge_and_unload()
                pad_id = getattr(artifacts.tokenizer, "pad_token_id", None)
                if pad_id is not None:
                    merged.generation_config.pad_token_id = pad_id
                    merged.config.pad_token_id = pad_id
                merged.config.use_cache = True
                merged.save_pretrained(merged_dir)
                if artifacts.processor is not None:
                    artifacts.processor.save_pretrained(merged_dir)
            except Exception as e:
                print(f"Merge failed: {e}")

        # Log artifacts
        adapters_art = wandb.Artifact(f"{run_name}-adapters", type="model"); adapters_art.add_dir(adapters_dir); wandb.log_artifact(adapters_art)
        if bool(self.cfg.get("export_merged", False)):
            merged_dir = os.path.join(final_dir, "merged")
            if os.path.isdir(merged_dir):
                merged_art = wandb.Artifact(f"{run_name}-merged", type="model"); merged_art.add_dir(merged_dir); wandb.log_artifact(merged_art)
        print(f"All done. Run directory: {run_dir}")
