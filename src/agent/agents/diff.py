import json
from langchain_core.output_parsers import StrOutputParser
from agent.prompts import DIFF_PROMPT

def diff_rules(llm, old_extraction: dict, new_extraction: dict) -> str:
    chain = DIFF_PROMPT | llm | StrOutputParser()
    old_json = json.dumps(old_extraction, ensure_ascii=False, indent=2)
    new_json = json.dumps(new_extraction, ensure_ascii=False, indent=2)
    return chain.invoke({"old_json": old_json, "new_json": new_json}).strip()
