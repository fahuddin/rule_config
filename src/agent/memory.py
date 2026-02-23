# agent/memory.py  (replace the bottom half with this)

import json
import os
import time
from typing import Any, Dict, List

MEM_DIR = "memory"
USER_PROFILE = os.path.join(MEM_DIR, "user_profile.json")
MAPPINGS = os.path.join(MEM_DIR, "mappings.json")
REFLECT = os.path.join(os.path.dirname(__file__), "..", "memory.json")

MAX_MEM_ITEM_SIZE = 2000        # max chars for any single saved field
MAX_CONTEXT_CHARS = 8000        # cap for assembled context


def _now_ts() -> float:
    return time.time()


def _trim_text(s: str, limit: int) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s if len(s) <= limit else s[:limit] + "…[truncated]"


def save_memory_item(item: Dict[str, Any]) -> None:
    """
    Safely append a memory item with metadata.
    Expected item shape: {'type': 'reflection'|'glossary'|'mapping'|..., 'data': {...}}
    If `type` is missing, it will be set to 'misc'. Large text fields are truncated.
    """
    os.makedirs(os.path.dirname(REFLECT), exist_ok=True)

    # Normalize item
    safe_item = dict(item)  # shallow copy
    if "type" not in safe_item:
        safe_item["type"] = "misc"
    safe_item["ts"] = safe_item.get("ts", _now_ts())

    # Trim any string fields larger than MAX_MEM_ITEM_SIZE
    for k, v in list(safe_item.items()):
        if isinstance(v, str) and len(v) > MAX_MEM_ITEM_SIZE:
            safe_item[k] = _trim_text(v, MAX_MEM_ITEM_SIZE)

    data: List[Dict[str, Any]] = []
    if os.path.isfile(REFLECT):
        try:
            with open(REFLECT, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except Exception:
            data = []

    data.append(safe_item)

    # Optional: cap file size by dropping oldest entries if too many
    MAX_ITEMS = 2000
    if len(data) > MAX_ITEMS:
        data = data[-MAX_ITEMS:]

    with open(REFLECT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def load_memory() -> Dict[str, Any]:
    profile = _read_json(
        USER_PROFILE,
        {"tone": "non-technical", "style": "concise"},
    )

    mappings = _read_json(
        MAPPINGS,
        {"output_labels": {}, "field_definitions": {}},
    )


    return {"profile": profile, "mappings": mappings}


def format_context_from_memory(mem: Dict[str, Any], *,
                               include_reflect_types: List[str] = None,
                               max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Convert memory into human-readable text for prompts.

    - By default, only include mappings (output_labels, field_definitions).
    - Optionally, include reflect items of particular types (whitelist).
      E.g., include_reflect_types=['glossary','domain_terms'] will include only those types.
    """
    if include_reflect_types is None:
        include_reflect_types = ["glossary", "domain_terms", "mappings"]

    mappings = mem.get("mappings", {}) or {}
    output_labels = mappings.get("output_labels", {}) or {}
    field_definitions = mappings.get("field_definitions", {}) or {}

    parts: List[str] = []

    # Profile hints (short)
    profile = mem.get("profile", {})
    if profile:
        profile_frag = []
        if profile.get("tone"):
            profile_frag.append(f"Tone: {profile.get('tone')}")
        if profile.get("style"):
            profile_frag.append(f"Style: {profile.get('style')}")
        if profile_frag:
            parts.append(" | ".join(profile_frag))

    if field_definitions:
        parts.append("Field definitions:")
        for field, definition in field_definitions.items():
            parts.append(f"- {field}: {_trim_text(definition, 300)}")

    if output_labels:
        parts.append("Output label meanings:")
        for label, meaning in output_labels.items():
            parts.append(f"- {label}: {_trim_text(meaning, 300)}")

    # Optionally include selected reflect entries (whitelist)
    reflect_items = mem.get("reflect", []) or []
    if include_reflect_types:
        included = []
        for item in reflect_items:
            t = item.get("type", "misc")
            if t in include_reflect_types:
                # string-ify safely and trim
                text = item.get("data") or item.get("note") or item.get("text") or item
                # convert dicts to readable form
                if isinstance(text, (dict, list)):
                    try:
                        text = json.dumps(text, ensure_ascii=False)
                    except Exception:
                        text = str(text)
                text = _trim_text(str(text), 500)
                included.append(f"- [{t}] {text}")
        if included:
            parts.append("Selected memory entries:")
            parts.extend(included)

    # Join and cap length
    result = "\n".join(str(p) for p in parts)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n…[context truncated]"

    return result
