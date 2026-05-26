from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "build_seq2seq_training_args": ("frame2kg_train.train.args", "build_seq2seq_training_args"),
    "PeriodicEvalCallback": ("frame2kg_train.train.callbacks", "PeriodicEvalCallback"),
    "CustomWandbCallback": ("frame2kg_train.train.callbacks", "CustomWandbCallback"),
    "SaveAdaptersCallback": ("frame2kg_train.train.callbacks", "SaveAdaptersCallback"),
    "get_backend": ("frame2kg_train.train.registry", "get_backend"),
    "Runner": ("frame2kg_train.train.run", "Runner"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
