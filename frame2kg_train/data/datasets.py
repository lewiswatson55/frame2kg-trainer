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
