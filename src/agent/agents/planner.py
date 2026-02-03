import json
import logging
from typing import List
from langchain_core.output_parsers import StrOutputParser
from agent.prompts import PLANNER_PROMPT

logging.basicConfig(level=logging.DEBUG)

def plan_steps(mode: str) -> List[str]:
    if mode == "diff":
        return ["parse", "parse", "diff"]
    if mode == "tests":
        return ["parse", "generate_tests"]
    if mode == "verify":
        return ["parse", "static_checks", "retrieve_context", "explain", "reflect", "verify", "rewrite"]
    if mode == "explain":
        return ["parse", "retrieve_context", "explain", "reflect", "rewrite"]
    return ["parse", "static_checks", "retrieve_context", "explain", "verify", "rewrite"]
