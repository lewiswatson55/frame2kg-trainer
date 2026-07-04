from __future__ import annotations

import os
import select
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
from transformers import Seq2SeqTrainer, set_seed, EarlyStoppingCallback

from frame2kg_train.train.args import build_seq2seq_training_args
from frame2kg_train.train.callbacks import PeriodicEvalCallback, CustomWandbCallback, SaveAdaptersCallback
from frame2kg_train.train.registry import get_backend
from frame2kg_train.data.datasets import load_frame2kg
from frame2kg_train.eval.metrics import make_compute_metrics
from frame2kg_train.backends.tokenizer_extension import save_tokenizer_extension_manifest
from frame2kg_train.data.graph_formats import (
    graph_to_target_text,
    normalise_graph_key_order,
    normalise_target_format,
    system_prompt_for_target_format,
    user_prompt_for_target_format,
)

def _post_run_wait_and_shutdown(cfg: Dict[str, Any], *, run_failed: bool) -> None:
    shutdown_cfg = cfg.get("auto_shutdown", True)
    if shutdown_cfg in (False, None):
        print("Auto-shutdown disabled; exiting immediately.")
        return

    wait_minutes = 30
    skip_token = "skip"
    command: Any = ["sudo", "shutdown", "now"]

    if isinstance(shutdown_cfg, dict):
        wait_minutes = shutdown_cfg.get("wait_minutes", wait_minutes)
        skip_token = shutdown_cfg.get("skip_token", skip_token)
        command = shutdown_cfg.get("command", command)

    try:
        wait_seconds = max(0, int(float(wait_minutes) * 60))
    except (TypeError, ValueError):
        wait_seconds = 30 * 60

    skip_token = str(skip_token).strip().lower() or "skip"

    if os.name == "nt":
        print("Auto-shutdown wait is only supported on Unix-like systems; exiting without shutdown.")
        return

    status = "failed or was interrupted" if run_failed else "complete"
    print(
        f"Training {status}. The machine will shut down in {wait_seconds // 60} minutes unless you type "
        f"'{skip_token}' and press Enter. This is to save on cloud GPU costs."
    )
    sys.stdout.flush()

    try:
        ready, _, _ = select.select([sys.stdin], [], [], wait_seconds)
        if ready:
            response = sys.stdin.readline().strip().lower()
            if response == skip_token:
                print("Auto-shutdown skipped by user input.")
                return
    except KeyboardInterrupt:
        print("Auto-shutdown cancelled via keyboard interrupt.")
        return
    except Exception as exc:
        print(f"Input wait failed ({exc}); proceeding with shutdown.")

    print("No skip command received; attempting to shut down the machine now…")
    try:
        if isinstance(command, str):
            subprocess.run(command, shell=True, check=True)
        else:
            subprocess.run(command, check=True)
    except Exception as exc:
        print(f"Failed to trigger shutdown ({exc}). Please shut down the machine manually.")


class Runner:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def run(self) -> None:
        import wandb
        run_failed = True
        seed = int(self.cfg.get("seed", 42))
        set_seed(seed); np.random.seed(seed)
        skip_eval = bool(self.cfg.get("skip_eval", False))
        model_id = self.cfg.get("model_id", "Qwen/Qwen2.5-VL-3B-Instruct")
        graph_key_order = normalise_graph_key_order(self.cfg.get("graph_key_order", "dataset"))
        target_format = normalise_target_format(self.cfg.get("target_format", "json"))
        system_prompt = self.cfg.get("system_prompt") or system_prompt_for_target_format(target_format)
        user_prompt = self.cfg.get("user_prompt") or user_prompt_for_target_format(target_format)
        run_ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        base_run_name = self.cfg.get("run_name", "frame2kg")
        run_name = f"{base_run_name}-{run_ts}"
        out_dir = os.path.abspath(self.cfg.get("output_dir", "../output"))
        run_dir = os.path.join(out_dir, run_name)
        ckpt_dir = os.path.join(run_dir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
        final_dir = os.path.join(run_dir, "final"); os.makedirs(final_dir, exist_ok=True)

        try:
            wandb_cfg = dict(self.cfg)
            wandb_cfg["graph_key_order"] = graph_key_order # adding to show order used for reproducability
            wandb_cfg["target_format"] = target_format
            wandb.init(
                project=self.cfg["wandb"]["project"],
                entity=self.cfg["wandb"].get("entity"),
                name=run_name,
                config=wandb_cfg,
            )

            backend_name = self.cfg.get("backend", "qwen25_vl")
            backend = get_backend(backend_name)
            artifacts = backend.load(self.cfg)
            if hasattr(artifacts.collator, "graph_key_order"):
                setattr(artifacts.collator, "graph_key_order", graph_key_order)
            if hasattr(artifacts.collator, "target_format"):
                setattr(artifacts.collator, "target_format", target_format)
            if hasattr(artifacts.collator, "system_prompt"):
                setattr(artifacts.collator, "system_prompt", system_prompt)
            if hasattr(artifacts.collator, "user_prompt"):
                setattr(artifacts.collator, "user_prompt", user_prompt)

            compute_metrics = None

            train_ds, eval_ds, test_ds = load_frame2kg(
                seed,
                dataset_id=str(self.cfg.get("dataset_id", self.cfg.get("dataset_name", "lewiswatson/Frame2KG-YC2"))),
                dataset_config=self.cfg.get("dataset_config"),
                dataset_token=self.cfg.get("dataset_token"),
            )
            train_ds = train_ds.add_column("mode", ["train"] * len(train_ds))

            if not skip_eval:
                if eval_ds is None:
                    raise RuntimeError("Validation split not found.")
                compute_metrics = make_compute_metrics(artifacts.tokenizer, target_format=target_format)
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
                eval_ds = eval_ds.add_column("mode", ["eval"] * len(eval_ds))
                # Attach pre‑stringified GT for metrics
                compute_metrics._eval_label_texts = [
                    graph_to_target_text(r["graph"], graph_key_order, target_format) for r in eval_ds
                ]  # type: ignore[attr-defined]
            else:
                eval_ds = None

            # Trainer bits
            max_new = int(self.cfg.get("max_new_tokens", 4098))
            generation_prompt_buffer = int(self.cfg.get("generation_prompt_buffer", 2048))
            generation_max_length = int(self.cfg.get("generation_max_length", max_new + generation_prompt_buffer))
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
                "save_steps": int(self.cfg.get("save_steps", 100)),
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
                "generation_prompt_buffer": generation_prompt_buffer,
                "generation_max_length": generation_max_length,
            }
            if "max_steps" in self.cfg:
                common["max_steps"] = int(self.cfg["max_steps"])
            training_args = build_seq2seq_training_args(common, max_new_tokens=max_new, do_eval=not skip_eval)

            wandb_cb = CustomWandbCallback(
                backend=backend,
                artifacts=artifacts,
                eval_ds=eval_ds,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new,
                graph_key_order=graph_key_order,
                target_format=target_format,
            )
            include_processor_in_checkpoints = bool(
                self.cfg.get("include_processor_in_adapter_checkpoints", target_format != "json")
            )
            save_adapters_cb = SaveAdaptersCallback(
                proc=artifacts.processor,
                tokenizer=artifacts.tokenizer,
                max_new_tokens=max_new,
                include_processor=include_processor_in_checkpoints,
            )
            callbacks = [wandb_cb, save_adapters_cb]
            if not skip_eval:
                periodic_eval = PeriodicEvalCallback(every_steps=int(self.cfg.get("eval_steps", 20)))
                callbacks.insert(0, periodic_eval)
            patience = int(self.cfg.get("early_stopping_patience", 0))
            if patience > 0:
                threshold = float(self.cfg.get("early_stopping_threshold", 0.0))
                callbacks.append(EarlyStoppingCallback(
                    early_stopping_patience=patience,
                    early_stopping_threshold=threshold,
                ))

            trainer = Seq2SeqTrainer(
                model=artifacts.model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=artifacts.collator,
                compute_metrics=compute_metrics,
                processing_class=artifacts.tokenizer,
                callbacks=callbacks,
            )

            print("Starting fine‑tuning …")
            trainer.train()
            trainer.save_state()

            # Save adapters + processor
            adapters_dir = os.path.join(final_dir, "adapters"); os.makedirs(adapters_dir, exist_ok=True)
            artifacts.model.save_pretrained(adapters_dir)
            if artifacts.processor is not None:
                artifacts.processor.save_pretrained(adapters_dir)
            if artifacts.tokenizer is not None:
                artifacts.tokenizer.save_pretrained(adapters_dir)
            save_tokenizer_extension_manifest(artifacts.tokenizer, adapters_dir)

            # Final eval
            if not skip_eval:
                metrics = trainer.evaluate()
                metrics = {f"final/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
                wandb.log(metrics)
            else:
                print("Skipping final evaluation (skip_eval=true).")
                wandb.log({"final/eval_skipped": 1})

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
                    if artifacts.tokenizer is not None:
                        artifacts.tokenizer.save_pretrained(merged_dir)
                    save_tokenizer_extension_manifest(artifacts.tokenizer, merged_dir)
                except Exception as e:
                    print(f"Merge failed: {e}")

            # Log artifacts
            adapters_art = wandb.Artifact(f"{run_name}-adapters", type="model"); adapters_art.add_dir(adapters_dir); wandb.log_artifact(adapters_art)
            if bool(self.cfg.get("export_merged", False)):
                merged_dir = os.path.join(final_dir, "merged")
                if os.path.isdir(merged_dir):
                    merged_art = wandb.Artifact(f"{run_name}-merged", type="model"); merged_art.add_dir(merged_dir); wandb.log_artifact(merged_art)
            print(f"All done. Run directory: {run_dir}")
            run_failed = False
        except Exception as exc:
            print(f"Training run raised an exception: {exc}")
            raise
        finally:
            try:
                wandb_run = getattr(wandb, "run", None)
                if wandb_run is not None:
                    exit_code = 0 if not run_failed else 1
                    try:
                        wandb.finish(exit_code=exit_code)
                    except TypeError:
                        # Older wandb versions do not accept exit_code
                        wandb.finish()
            except Exception as finish_exc:
                print(f"Failed to close WandB run cleanly: {finish_exc}")

            _post_run_wait_and_shutdown(self.cfg, run_failed=run_failed)
