from __future__ import annotations
from dataclasses import dataclass
from html import escape
from typing import Any

import torch
from datasets import Dataset
from PIL import Image
from transformers import TrainerCallback

from frame2kg.eval.json_utils import first_json_object
from frame2kg.eval.metrics import calc_node_scores, calc_edge_scores


class PeriodicEvalCallback(TrainerCallback):
    def __init__(self, every_steps: int):
        self.every_steps = every_steps
    def on_step_end(self, args, state, control, **kwargs):
        strategy = getattr(args, "evaluation_strategy", "no")
        if strategy == "no" and state.global_step and state.global_step % self.every_steps == 0:
            control.should_evaluate = True
        return control


@dataclass
class SaveAdaptersCallback(TrainerCallback):
    proc: Any | None
    max_new_tokens: int
    def on_save(self, args, state, control, model=None, **kwargs):
        import os, json, wandb
        if model is None: return control
        step=int(getattr(state, "global_step", 0) or 0)
        tmp_dir=os.path.join(args.output_dir, f"adapters-step-{step}")
        os.makedirs(tmp_dir, exist_ok=True)
        model.save_pretrained(tmp_dir)
        metrics=kwargs.get("metrics") or {}
        with open(os.path.join(tmp_dir, "checkpoint_info.json"), "w", encoding="utf-8") as f:
            json.dump({"global_step": step, "event": "on_save", "max_new_tokens": self.max_new_tokens, "metrics": metrics}, f, ensure_ascii=False, indent=2)
        art=wandb.Artifact(name=f"{args.run_name}-adapters", type="model", metadata={"global_step": step, "event": "on_save", "max_new_tokens": self.max_new_tokens})
        art.add_dir(tmp_dir)
        wandb.log_artifact(art, aliases=[f"step-{step}", "latest-save"])
        if (self.proc is not None) and (os.environ.get("SAVE_PROCESSOR_ONCE","0")=="1"):
            proc_dir=os.path.join(args.output_dir, "processor")
            os.makedirs(proc_dir, exist_ok=True)
            self.proc.save_pretrained(proc_dir)
            proc_art=wandb.Artifact(name=f"{args.run_name}-processor", type="processor")
            proc_art.add_dir(proc_dir)
            wandb.log_artifact(proc_art, aliases=["processor-latest"])
        return control


@dataclass
class CustomWandbCallback(TrainerCallback):
    backend: Any
    artifacts: Any
    eval_ds: Dataset
    system_prompt: str
    max_new_tokens: int

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        import wandb
        if model is None: return
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        total=sum(p.numel() for p in model.parameters())
        print(f"Training {trainable} parameters out of {total} total ({100*trainable/total:.2f}% trainable)")
        tok=self.artifacts.tokenizer
        wandb.config.update({
            "trainable_params": trainable,
            "all_params": total,
            "trainable_percentage": 100*trainable/total,
            "pad_token_id": getattr(tok, "pad_token_id", None),
        }, allow_val_change=True)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        import wandb
        if model is None or len(self.eval_ds)==0: return control
        columns=["global_step","index","image","gt_json","pred_raw","json_parsed","node_P","node_R","node_F1","node_bbox_iou","edge_P","edge_R","edge_F1"]
        table=wandb.Table(columns=columns)
        first_pred=None; first_gt=None
        for i in range(len(self.eval_ds)):
            ex=self.eval_ds[i]
            img=ex["image"]
            from PIL import Image as _Image
            if not isinstance(img,_Image.Image):
                from PIL import Image as _Image2
                img=_Image2.open(img).convert("RGB")
            gt_raw=ex["graph"]
            import json as _json
            gt_str=gt_raw if isinstance(gt_raw,str) else _json.dumps(gt_raw, ensure_ascii=False, separators=(",", ":"))
            prompt=self.system_prompt.replace("The first character must be \"{\"", "The first character must be \"{\"")  # no-op; keep prompt stable
            # Build chat using the processor's template to ensure image placeholders match
            chat = [
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Extract data in JSON."},
                    ],
                },
            ]
            user_prompt = self.artifacts.processor.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            pred_txt = self.backend.generate_text(
                self.artifacts, img, user_prompt, self.max_new_tokens
            )
            pjson=first_json_object(pred_txt)
            nP=nR=nF=nIoU=eP=eR=eF=0.0
            try:
                gt_json = gt_raw if isinstance(gt_raw,(dict,list)) else _json.loads(gt_str)
                if pjson is not None and isinstance(gt_json, dict):
                    gt_nodes=gt_json.get("nodes",[]); pr_nodes=pjson.get("nodes",[])
                    gt_edges=gt_json.get("edges",[]); pr_edges=pjson.get("edges",[])
                    P,R,F,m,iou=calc_node_scores(gt_nodes, pr_nodes)
                    p,r,f=calc_edge_scores(gt_edges, pr_edges, m, gt_nodes, pr_nodes)
                    nP,nR,nF,nIoU,eP,eR,eF=P,R,F,iou,p,r,f
            except Exception:
                pass
            if first_pred is None:
                first_pred=pred_txt; first_gt=gt_str
            def _clip(s: str, limit: int = 20000) -> str:
                return s if len(s)<=limit else s[:limit]+"\n...[truncated]"
            table.add_data(int(getattr(state,"global_step",0) or 0), i, wandb.Image(img), _clip(gt_str), _clip(pred_txt), int(pjson is not None), float(nP), float(nR), float(nF), float(nIoU), float(eP), float(eR), float(eF))
        wandb.log({"eval/sample_table": table})
        if first_pred is not None and first_gt is not None:
            wandb.log({"eval/sample_pred_raw": wandb.Html(f"<pre>{escape(first_pred)}</pre>"), "eval/sample_gt_json": wandb.Html(f"<pre>{escape(first_gt)}</pre>")})
        return control
