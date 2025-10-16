from __future__ import annotations
import json, re

SMART_QUOTES = {"\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"'}


def normalise_json_text(s: str) -> str:
    s = s.replace("\ufeff", "")
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE)
    s = s.replace("<|im_start|>", "").replace("<|im_end|>", "")
    s = s.replace("</s>", "")
    for k, v in SMART_QUOTES.items():
        s = s.replace(k, v)
    return s


def slice_from_assistant(s: str) -> str:
    cut = 0
    for m in re.finditer(r"(?im)^\s*assistant\s$", s):
        cut = m.end()
    return s[cut:] if cut else s


def first_json_object(text: str):
    """Extract the last JSON object containing graph keys from ``text``.

    The system prompt embeds schema examples such as
    ``{"predicate":"str","source":"node.id","target":"node.id"}``, which
    are valid JSON objects but not model predictions.  When a model fails to
    produce valid output, the previous implementation would fall back to these
    snippets.  To avoid scoring the schema, scan candidate objects from the end
    of the string and return the first one that parses and includes a "nodes" or
    "edges" key.  If none are found, return ``None``.
    """
    if not text:
        return None
    s = normalise_json_text(text).strip()
    s = slice_from_assistant(s)
    positions = [m.start() for m in re.finditer(r"\{", s)]
    for start in reversed(positions):
        depth = 0
        end = -1
        for i, ch in enumerate(s[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            continue
        cand = s[start:end]
        try:
            obj = json.loads(cand)
        except Exception:
            import re as _re
            cand2 = _re.sub(r",(\s*[}\]])", r"\1", cand)
            try:
                obj = json.loads(cand2)
            except Exception:
                continue
        if isinstance(obj, dict) and ("nodes" in obj or "edges" in obj):
            return obj
    # Debug helper when no valid JSON graph was found
    print("DEBUG: Using for prediction: None")
    return None
