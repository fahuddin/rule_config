import json
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .types import AgentResult
from agent.prompts import REFLECT_PROMPT

logging.basicConfig(level=logging.DEBUG)

def reflect(llm, extraction: dict, english: str) -> AgentResult:
    chain = REFLECT_PROMPT | llm | StrOutputParser()
    extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)
    raw = chain.invoke({"extraction_json": extraction_json, "english": english}).strip()

    # minimal json recovery
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        raw = raw[s:e+1]
    obj = json.loads(raw)
    logging.DEBUG("REFLECT:", obj)
    

    return AgentResult(
        ok=bool(obj.get("ok", False)),
        issues=list(obj.get("issues", []))
    )