"""Model-output parsing shared by the benchmark runners.

Responses are parsed with a three-stage fallback (direct JSON parse; then a
fenced ```json block; then the first-to-last {...} span). Reasoning traces such
as <think>...</think> (Qwen3, DeepSeek-R1) are removed first.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_RE = re.compile(r"<(think|thinking|reasoning)>.*?(</\1>|$)", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def parse_json_response(raw: str | None) -> dict[str, Any] | None:
    """Return the parsed JSON object, or None if no valid object can be extracted."""
    if not raw or not raw.strip():
        return None
    text = _THINK_RE.sub("", raw).strip()

    candidates = [text]
    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])

    for c in candidates:
        try:
            obj = json.loads(c)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj
    return None
