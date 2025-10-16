from __future__ import annotations
from dataclasses import fields
from typing import Any, Dict
from transformers import Seq2SeqTrainingArguments


def build_seq2seq_training_args(common: Dict[str, Any], max_new_tokens: int) -> Seq2SeqTrainingArguments:
    arg_names = {f.name for f in fields(Seq2SeqTrainingArguments)}
    kw = {k: v for k, v in common.items() if k in arg_names}
    if "predict_with_generate" in arg_names: kw["predict_with_generate"] = True
    if "generation_max_new_tokens" in arg_names: kw["generation_max_new_tokens"] = max_new_tokens
    if "generation_num_beams" in arg_names: kw["generation_num_beams"] = 1
    if "return_dict_in_generate" in arg_names: kw["return_dict_in_generate"] = False
    if "output_scores" in arg_names: kw["output_scores"] = False
    eval_key = "evaluation_strategy" if "evaluation_strategy" in arg_names else ("eval_strategy" if "eval_strategy" in arg_names else None)
    if eval_key is not None:
        kw[eval_key] = "steps"
        if "save_strategy" in arg_names: kw["save_strategy"] = "steps"
        if "load_best_model_at_end" in arg_names: kw["load_best_model_at_end"] = True
        if "metric_for_best_model" in arg_names: kw["metric_for_best_model"] = "edge_F1"
        if "greater_is_better" in arg_names: kw["greater_is_better"] = True
    else:
        if "load_best_model_at_end" in kw: kw["load_best_model_at_end"] = False
        kw.pop("metric_for_best_model", None); kw.pop("greater_is_better", None)
    args = Seq2SeqTrainingArguments(**kw)
    if not hasattr(args, "predict_with_generate"): setattr(args, "predict_with_generate", True)
    if not hasattr(args, "evaluation_strategy"): setattr(args, "evaluation_strategy", "no")
    return args
