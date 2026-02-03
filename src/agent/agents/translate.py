import json
import logging
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from agent.prompts import ENGLISH_TO_MVEL_PROMPT
from typing import List, Dict, Any

def generate_mvel(llm, extraction: dict) -> List[dict]:
    chain = ENGLISH_TO_MVEL_PROMPT | llm | StrOutputParser()
    extraction_json = json.dumps(extraction, ensure_ascii=False, indent=2)
    raw = chain.invoke({"extraction_json": extraction_json})
    return raw
