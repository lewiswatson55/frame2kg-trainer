from __future__ import annotations
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from datasets import Dataset, Image as HFImage, load_dataset

DEFAULT_DATASET_ID = "lewiswatson/Frame2KG-YC2"


def _ensure_image_column(ds: Dataset) -> Dataset:
    if ds.features["image"].__class__.__name__ != "Image":
        return ds.cast_column("image", HFImage())
    return ds


def _dataset_token_arg(value: Any) -> Any:
    if value in (None, False):
        return None
    if value is True:
        return True
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "false", "none", "null", "0"}:
            return None
        if lowered in {"true", "1", "yes", "auto"}:
            return True
        if lowered == "env":
            return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or True
    return value


def load_frame2kg(
    seed: int = 42,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str | None = None,
    dataset_token: Any = None,
) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    load_kwargs: Dict[str, Any] = {}
    token = _dataset_token_arg(dataset_token)
    if token is not None:
        load_kwargs["token"] = token
    if dataset_config:
        raw = load_dataset(dataset_id, dataset_config, **load_kwargs)
    else:
        raw = load_dataset(dataset_id, **load_kwargs)
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
