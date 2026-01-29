import json
import logging
from typing import List
from langchain_core.output_parsers import StrOutputParser
from agent.prompts import PLANNER_PROMPT

logging.basicConfig(level=logging.DEBUG)

def _parse_json_only(text: str) -> dict:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    # recover first {...}
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        return json.loads(text[s:e+1])
    raise ValueError("Planner did not return valid JSON.")

def plan_steps(llm, mode: str) -> List[str]:
    chain = PLANNER_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"mode": mode})
    obj = _parse_json_only(raw)
    logging.debug("Check obj", obj)
    steps = obj.get("steps", [])
    logging.debug("Check steps", steps)
    logging.debug("Check mode", mode)
    if mode == "diff":
        return ["parse", "parse", "diff"]
    if mode == "tests":
        return ["parse", "generate_tests"]
    if mode == "verify":
        logging.debug("Check mode", steps)
        return ["parse", "static_checks", "retrieve_context", "reflect", "verify", "rewrite"]
    if mode == "explain":
        logging.debug("Reached explain", steps)
        return ["parse", "retrieve_context", "explain", "reflect", "rewrite"]
    return ["parse", "static_checks", "retrieve_context", "explain", "verify", "rewrite"]
