import json
from typing import Any, List
from langchain_core.output_parsers import StrOutputParser
from agent.prompts import TESTS_PROMPT

def _parse_json_array(text: str) -> List[Any]:
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        return json.loads(text)
    s = text.find("[")
    e = text.rfind("]")
    if s != -1 and e != -1 and e > s:
        return json.loads(text[s:e+1])
    raise ValueError("Tests agent did not return valid JSON array.")

def generate_tests(llm, extraction: dict) -> List[dict]:
    chain = TESTS_PROMPT | llm | StrOutputParser()
    extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)
    raw = chain.invoke({"extraction_json": extraction_json})
    return _parse_json_array(raw)
