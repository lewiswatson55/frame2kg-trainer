from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from datasets import Dataset, Image as HFImage, load_dataset


def _ensure_image_column(ds: Dataset) -> Dataset:
    if ds.features["image"].__class__.__name__ != "Image":
        return ds.cast_column("image", HFImage())
    return ds


def load_frame2kg(seed: int = 42) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    raw = load_dataset("lewiswatson/Frame2KG-YC2")
    train = raw.get("training") or raw.get("train")
    if train is None:
        raise RuntimeError("No training split found.")
    eval_ = raw.get("validation")
    test = raw.get("testing")

    train = _ensure_image_column(train)
    if eval_ is not None:
        eval_ = _ensure_image_column(eval_)
    if test is not None:
        test = _ensure_image_column(test)

    train = train.shuffle(seed=seed)
    return train, eval_, test

def load_frame2kg_preferences(cfg: dict, seed: int = 42) -> Tuple[Dataset, Optional[Dataset]]:
    dpo_cfg = cfg.get("dpo", {})
    dataset_name = dpo_cfg.get("dataset_name")
    if not dataset_name:
        raise RuntimeError("DPO training requires dpo.dataset_name in the config.")

    dataset_config_name = dpo_cfg.get("dataset_config")
    train_split = dpo_cfg.get("train_split", "train")
    eval_split = dpo_cfg.get("eval_split")

    if dataset_config_name:
        raw = load_dataset(dataset_name, dataset_config_name)
    else:
        raw = load_dataset(dataset_name)

    train = raw.get(train_split)
    if train is None:
        raise RuntimeError(f"No train split '{train_split}' found in DPO dataset.")
    eval_ = raw.get(eval_split) if eval_split else None

    train = _ensure_image_column(train).shuffle(seed=seed)
    if eval_ is not None:
        eval_ = _ensure_image_column(eval_)

    required = {"image", "chosen", "rejected"}
    missing = [c for c in required if c not in train.column_names]
    if missing:
        raise RuntimeError(f"DPO dataset is missing required columns: {missing}")

    return train, eval_