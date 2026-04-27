from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

TEXT_BACKBONE_CANDIDATE_PATHS: tuple[str, ...] = (
    "model.language_model",
    "language_model",
)
TEXT_ATTN_BRANCHES = frozenset({"self_attn", "cross_attn"})
TEXT_ATTN_TARGETS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
TEXT_MLP_TARGETS = frozenset({"gate_proj", "down_proj", "up_proj"})
TEXT_TARGETS = TEXT_ATTN_TARGETS | TEXT_MLP_TARGETS


def get_attr_path(root: Any, path: str) -> Any | None:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def locate_text_backbone(model: Any) -> Tuple[str, Any]:
    for candidate in TEXT_BACKBONE_CANDIDATE_PATHS:
        text_backbone = get_attr_path(model, candidate)
        if text_backbone is not None:
            return candidate, text_backbone
    raise RuntimeError(
        "Mllama text backbone not found. Expected one of: "
        + ", ".join(TEXT_BACKBONE_CANDIDATE_PATHS)
        + "."
    )


def is_allowed_text_decoder_relative_path(path: str) -> bool:
    parts = path.split(".")
    if len(parts) != 4 or parts[0] != "layers" or not parts[1].isdigit():
        return False

    branch, leaf = parts[2], parts[3]
    if branch in TEXT_ATTN_BRANCHES:
        return leaf in TEXT_ATTN_TARGETS
    if branch == "mlp":
        return leaf in TEXT_MLP_TARGETS
    return False


def is_allowed_text_decoder_target_path(path: str, text_backbone_prefix: str) -> bool:
    prefix = f"{text_backbone_prefix}."
    if not path.startswith(prefix):
        return False
    return is_allowed_text_decoder_relative_path(path[len(prefix) :])


def resolve_text_decoder_target_modules(
    model: Any,
    target_modules: Sequence[str],
) -> Tuple[str, Dict[str, List[str]]]:
    text_backbone_prefix, text_backbone = locate_text_backbone(model)
    matches = {str(name): [] for name in target_modules}

    for module_name, module in text_backbone.named_modules():
        if not module_name or not hasattr(module, "weight"):
            continue
        if not is_allowed_text_decoder_relative_path(module_name):
            continue

        short_name = module_name.rsplit(".", 1)[-1]
        if short_name not in matches:
            continue
        matches[short_name].append(f"{text_backbone_prefix}.{module_name}")

    for module_paths in matches.values():
        module_paths.sort()
    return text_backbone_prefix, matches


def flatten_resolved_target_modules(
    target_modules: Sequence[str],
    grouped_target_modules: Dict[str, List[str]],
) -> List[str]:
    return [
        module_name
        for target_name in target_modules
        for module_name in grouped_target_modules.get(str(target_name), [])
    ]


def group_exact_target_module_paths(
    module_paths: Iterable[str],
    target_modules: Sequence[str],
) -> Dict[str, List[str]]:
    grouped = {str(name): [] for name in target_modules}
    for module_path in sorted(module_paths):
        short_name = module_path.rsplit(".", 1)[-1]
        if short_name in grouped:
            grouped[short_name].append(module_path)
    return grouped
