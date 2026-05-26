from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple


JSON_TARGET_FORMAT = "json"
COMPRESSED_TOKENS_TARGET_FORMAT = "compressed_tokens"

COMPRESSED_GRAPH_TOKENS = [
    "<|graph|>",
    "<|nodes|>",
    "<|node|>",
    "<|id|>",
    "<|label|>",
    "<|bbox|>",
    "<|conf|>",
    "<|attrs|>",
    "<|edges|>",
    "<|edge|>",
    "<|src|>",
    "<|pred|>",
    "<|tgt|>",
    "<|end_graph|>",
]

SYSTEM_PROMPT = (
    'You are a VLM that outputs ONLY a single, strict JSON object with exactly the keys "nodes" and "edges". '
    'Use valid JSON: double quotes for all keys and string values, no single quotes, no trailing commas. '
    'Schema — "nodes": [{"id":"str","label":"str","location":"x1,y1,x2,y2,confidence","attributes":{...}}], '
    '"edges":[{"predicate":"str","source":"node.id","target":"node.id"}]. '
    'Output the JSON object only - no code fences, no role tags, no prefixes/suffixes, no prose. '
    'The first character must be "{", and the last must be "}".'
)

USER_PROMPT = "Extract data in JSON."

COMPRESSED_TOKENS_SYSTEM_PROMPT = (
    "You are a VLM that outputs ONLY one Frame2KG compressed graph string. "
    "Use the added structural tokens exactly as literal delimiters: "
    "<|graph|>, <|nodes|>, <|node|>, <|id|>, <|label|>, <|bbox|>, <|conf|>, <|attrs|>, "
    "<|edges|>, <|edge|>, <|src|>, <|pred|>, <|tgt|>, <|end_graph|>. "
    "Do not output JSON, code fences, role tags, prefixes, suffixes, or prose. "
    "The first token must be <|graph|>, and the last token must be <|end_graph|>."
)

COMPRESSED_TOKENS_USER_PROMPT = "Extract data as a compressed graph string."


def normalise_target_format(value: Any = JSON_TARGET_FORMAT) -> str:
    fmt = str(value or JSON_TARGET_FORMAT).strip().lower().replace("-", "_")
    aliases = {
        "json": JSON_TARGET_FORMAT,
        "json_graph": JSON_TARGET_FORMAT,
        "compressed": COMPRESSED_TOKENS_TARGET_FORMAT,
        "compressed_token": COMPRESSED_TOKENS_TARGET_FORMAT,
        "compressed_tokens": COMPRESSED_TOKENS_TARGET_FORMAT,
        "special_tokens": COMPRESSED_TOKENS_TARGET_FORMAT,
        "token_graph": COMPRESSED_TOKENS_TARGET_FORMAT,
    }
    if fmt not in aliases:
        raise ValueError(
            "target_format must be one of: json, compressed_tokens"
        )
    return aliases[fmt]


def system_prompt_for_target_format(target_format: Any) -> str:
    if normalise_target_format(target_format) == COMPRESSED_TOKENS_TARGET_FORMAT:
        return COMPRESSED_TOKENS_SYSTEM_PROMPT
    return SYSTEM_PROMPT


def user_prompt_for_target_format(target_format: Any) -> str:
    if normalise_target_format(target_format) == COMPRESSED_TOKENS_TARGET_FORMAT:
        return COMPRESSED_TOKENS_USER_PROMPT
    return USER_PROMPT


def normalise_graph_key_order(value: Any = "dataset") -> str:
    order = str(value or "dataset").strip().lower()
    if order not in {"dataset", "nodes_first", "edges_first"}:
        raise ValueError("graph_key_order must be one of: dataset, nodes_first, edges_first")
    return order


def _reorder_graph_keys(graph: Any, graph_key_order: str) -> Any:
    order = normalise_graph_key_order(graph_key_order)
    if order == "dataset" or not isinstance(graph, dict):
        return graph

    preferred = ("nodes", "edges") if order == "nodes_first" else ("edges", "nodes")
    reordered: Dict[str, Any] = {}
    for key in preferred:
        if key in graph:
            reordered[key] = graph[key]
    for key, value in graph.items():
        if key not in reordered:
            reordered[key] = value
    return reordered


def graph_to_json_text(graph: Any, graph_key_order: str = "dataset") -> str:
    if isinstance(graph, str):
        if normalise_graph_key_order(graph_key_order) == "dataset":
            return graph
        try:
            graph = json.loads(graph)
        except Exception:
            return graph

    graph = _reorder_graph_keys(graph, graph_key_order)
    return json.dumps(graph, ensure_ascii=False, separators=(",", ":"))


def graph_to_target_text(
    graph: Any,
    graph_key_order: str = "dataset",
    target_format: Any = JSON_TARGET_FORMAT,
) -> str:
    if normalise_target_format(target_format) == COMPRESSED_TOKENS_TARGET_FORMAT:
        if isinstance(graph, str):
            return graph
        raise TypeError("compressed_tokens target_format expects dataset graph values to be strings")
    return graph_to_json_text(graph, graph_key_order)


_COMPRESSED_TOKEN_RE = re.compile(
    "(" + "|".join(re.escape(token) for token in COMPRESSED_GRAPH_TOKENS) + ")"
)


def _compressed_segments(text: str) -> List[Tuple[str, str]]:
    matches = list(_COMPRESSED_TOKEN_RE.finditer(text or ""))
    segments: List[Tuple[str, str]] = []
    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segments.append((match.group(1), text[match.end() : next_start].strip()))
    return segments


def _parse_attr(value: str, attrs: Dict[str, Any]) -> None:
    if not value:
        return
    if "=" in value:
        key, attr_value = value.split("=", 1)
        key = key.strip()
        if key:
            attrs[key] = attr_value.strip()
            return
    attrs[f"attr_{len(attrs) + 1}"] = value


def compressed_graph_to_json(text: str) -> Dict[str, Any] | None:
    if not text or "<|graph|>" not in text:
        return None

    start = text.find("<|graph|>")
    end = text.find("<|end_graph|>", start)
    if end != -1:
        text = text[start : end + len("<|end_graph|>")]
    else:
        text = text[start:]

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    current_node: Dict[str, Any] | None = None
    current_edge: Dict[str, Any] | None = None
    section: str | None = None

    def flush_node() -> None:
        nonlocal current_node
        if current_node is None:
            return
        if "attributes" not in current_node:
            current_node["attributes"] = {}
        if current_node.get("id") or current_node.get("label"):
            nodes.append(current_node)
        current_node = None

    def flush_edge() -> None:
        nonlocal current_edge
        if current_edge is None:
            return
        if current_edge.get("source") or current_edge.get("target") or current_edge.get("predicate"):
            edges.append(current_edge)
        current_edge = None

    for token, value in _compressed_segments(text):
        if token == "<|nodes|>":
            flush_edge()
            section = "nodes"
            continue
        if token == "<|edges|>":
            flush_node()
            section = "edges"
            continue
        if token == "<|end_graph|>":
            flush_node()
            flush_edge()
            section = None
            continue

        if section == "nodes":
            if token == "<|node|>":
                flush_node()
                current_node = {"attributes": {}}
            elif current_node is not None:
                if token == "<|id|>":
                    current_node["id"] = value
                elif token == "<|label|>":
                    current_node["label"] = value
                elif token == "<|bbox|>":
                    current_node["location"] = value
                elif token == "<|conf|>":
                    current_node["confidence"] = value
                elif token == "<|attrs|>":
                    _parse_attr(value, current_node.setdefault("attributes", {}))
            continue

        if section == "edges":
            if token == "<|edge|>":
                flush_edge()
                current_edge = {}
            elif current_edge is not None:
                if token == "<|src|>":
                    current_edge["source"] = value
                elif token == "<|pred|>":
                    current_edge["predicate"] = value
                elif token == "<|tgt|>":
                    current_edge["target"] = value

    flush_node()
    flush_edge()
    if not nodes and not edges:
        return None
    return {"nodes": nodes, "edges": edges}
