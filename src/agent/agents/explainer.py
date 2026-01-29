import json
from langchain_core.output_parsers import StrOutputParser
from agent.prompts import EXPLAIN_PROMPT

def explain_rule(llm, extraction: dict, context: str) -> str:
    chain = EXPLAIN_PROMPT | llm | StrOutputParser()
    extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)
    return chain.invoke({"extraction_json": extraction_json, "context": context}).strip()
