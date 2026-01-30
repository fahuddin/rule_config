import json
from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser
from agent.prompts import VERIFY_PROMPT, REWRITE_PROMPT

def _parse_json_only(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        return json.loads(text[s:e+1])
    raise ValueError("Verifier did not return valid JSON.")

def verify_explanation(llm, extraction: dict, english: str) -> Dict[str, Any]:
    chain = VERIFY_PROMPT | llm | StrOutputParser()
    extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)
    raw = chain.invoke({"extraction_json": extraction_json, "english": english})
    print(raw)
    verdict = _parse_json_only(raw)
    # normalize keys
    verdict.setdefault("ok", True)
    verdict.setdefault("missing", [])
    # Ensure missing entries are strings (LLM may return structured items)
    cleaned_missing = []
    for m in verdict.get("missing", []):
        if isinstance(m, str):
            cleaned_missing.append(m)
        else:
            try:
                cleaned_missing.append(json.dumps(m, ensure_ascii=False))
            except Exception:
                cleaned_missing.append(str(m))
    verdict["missing"] = cleaned_missing
    verdict.setdefault("rewrite_needed", verdict.get("ok") is False)
    return verdict

def rewrite_explanation(llm, extraction: dict, english: str, missing: List[str]) -> str:
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)
    # Ensure missing list elements are strings
    missing_parts = []
    for m in missing or []:
        if isinstance(m, str):
            missing_parts.append(m)
        else:
            try:
                missing_parts.append(json.dumps(m, ensure_ascii=False))
            except Exception:
                missing_parts.append(str(m))

    return chain.invoke({
        "extraction_json": extraction_json,
        "english": english,
        "missing": "\n".join(missing_parts) if missing_parts else "(none)"
    }).strip()
