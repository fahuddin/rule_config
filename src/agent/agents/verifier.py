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
    verdict = _parse_json_only(raw)
    # normalize keys
    verdict.setdefault("ok", True)
    verdict.setdefault("missing", [])
    verdict.setdefault("rewrite_needed", verdict.get("ok") is False)
    return verdict

def rewrite_explanation(llm, extraction: dict, english: str, missing: List[str]) -> str:
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)
    return chain.invoke({
        "extraction_json": extraction_json,
        "english": english,
        "missing": "\n".join(missing) if missing else "(none)"
    }).strip()
