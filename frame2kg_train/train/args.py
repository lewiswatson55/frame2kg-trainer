from __future__ import annotations
from dataclasses import fields
from typing import Any, Dict
from transformers import Seq2SeqTrainingArguments


def build_seq2seq_training_args(
    common: Dict[str, Any],
    max_new_tokens: int,
    do_eval: bool = True,
) -> Seq2SeqTrainingArguments:
    arg_names = {f.name for f in fields(Seq2SeqTrainingArguments)}
    kw = {k: v for k, v in common.items() if k in arg_names}
    prompt_buffer = int(common.get("generation_prompt_buffer", 2048))
    generation_max_length = int(common.get("generation_max_length", max_new_tokens + prompt_buffer))

    kw["predict_with_generate"] = True
    if "generation_max_new_tokens" in arg_names:
        kw["generation_max_new_tokens"] = max_new_tokens
    if "generation_max_length" in arg_names:
        kw["generation_max_length"] = generation_max_length
    if "generation_num_beams" in arg_names:
        kw["generation_num_beams"] = 1

    eval_key = "evaluation_strategy" if "evaluation_strategy" in arg_names else ("eval_strategy" if "eval_strategy" in arg_names else None)
    if eval_key is not None:
        if do_eval:
            kw[eval_key] = "steps"
            kw["save_strategy"] = "steps"
            kw["load_best_model_at_end"] = True
            kw["metric_for_best_model"] = "edge_F1"
            kw["greater_is_better"] = True
        else:
            kw[eval_key] = "no"
            kw["load_best_model_at_end"] = False
            kw.pop("metric_for_best_model", None); kw.pop("greater_is_better", None)
    else:
        if "load_best_model_at_end" in kw: kw["load_best_model_at_end"] = False
        kw.pop("metric_for_best_model", None); kw.pop("greater_is_better", None)
    return Seq2SeqTrainingArguments(**kw)
