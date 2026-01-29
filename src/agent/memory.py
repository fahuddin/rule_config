# agent/memory.py
import json
import os
from typing import Any, Dict

MEM_DIR = "memory"
USER_PROFILE = os.path.join(MEM_DIR, "user_profile.json")
MAPPINGS = os.path.join(MEM_DIR, "mappings.json")
MEM_PATH = os.path.join(os.path.dirname(__file__), "..", "memory.json")


def save_memory_item(item: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(MEM_PATH), exist_ok=True)
    data = []
    print(MEM_PATH)
    if os.path.isfile(MEM_PATH):
        try:
            with open(MEM_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or []
        except Exception:
            data = []
    data.append(item)
    with open(MEM_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        


def _read_json(path: str, default: Any) -> Any:
    """
    Safely read a JSON file. If the file does not exist or is invalid,
    return the provided default value instead of raising an error.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def load_memory() -> Dict[str, Any]:
    """
    Load all long-term agent memory from disk.

    Returns:
        A dictionary containing:
        - profile: user preferences (tone, style, etc.)
        - mappings: domain-specific mappings (field definitions, output meanings)
    """
    profile = _read_json(
        USER_PROFILE,
        {
            "tone": "non-technical",
            "style": "concise",
        },
    )

    mappings = _read_json(
        MAPPINGS,
        {
            "output_labels": {},
            "field_definitions": {},
        },
    )

    return {
        "profile": profile,
        "mappings": mappings,
    }


def format_context_from_memory(mem: Dict[str, Any]) -> str:
    """
    Convert memory into human-readable text that can be injected
    into LLM prompts as context.
    """
    mappings = mem.get("mappings", {})
    output_labels = mappings.get("output_labels", {})
    field_definitions = mappings.get("field_definitions", {})

    parts = []

    if field_definitions:
        parts.append("Field definitions:")
        for field, definition in field_definitions.items():
            parts.append(f"- {field}: {definition}")

    if output_labels:
        parts.append("Output label meanings:")
        for label, meaning in output_labels.items():
            parts.append(f"- {label}: {meaning}")

    return "\n".join(parts).strip()
