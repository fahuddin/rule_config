# agent/runner.py
import json
from typing import List

from agent.agents.reflect import reflect
from agent.llm import get_llm
from agent.memory import load_memory, format_context_from_memory,save_memory_item
from agent.tracing import Trace

from agent.tools.mvel_parser_tool import parse_mvel_branches
from agent.tools.static_checker_tool import run_static_checks
from agent.tools.rag import retrieve_context

from agent.agents.planner import plan_steps
from agent.agents.explainer import explain_rule
from agent.agents.verifier import verify_explanation, rewrite_explanation
from agent.agents.diff import diff_rules
from agent.agents.tests import generate_tests
from hashlib import sha256
from agent.agents.redis_mini import MiniRedis
import traceback



def run(mode: str, mvel_texts: List[str], model: str, enable_trace: bool) -> str:
    """
    Orchestrates the agent system.

    Args:
        mode: "explain" | "verify" | "tests" | "diff" | "agentic"
        mvel_texts: list of raw MVEL strings (1 item for most modes, 2 for diff)
        model: Ollama model name
        enable_trace: whether to write a run trace JSON file under runs/

    Returns:
        Final output string (English explanation, diff explanation, or JSON test cases)
    """
    # 1) Initialize shared resources
    llm = get_llm(model=model, temperature=0.0)  # deterministic
    mem = load_memory() #load user settings and domain mapping and reflection
    mem_context = format_context_from_memory(mem) 
    
    r = MiniRedis(host="127.0.0.1", port=6379)

    def hash_text(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()
    
    def get_cached_explanation(rule_hash: str) -> str | None:
        key = f"mvel:cache:explain:{rule_hash}"
        raw = r.get(key)
        if raw is None:
            return None
        return raw.decode("utf-8")
    
    def get_cached_parse(rule_hash: str) -> dict | None:
        key = f"mvel:cache:parse:{rule_hash}"
        raw = r.get(key)
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))
    
    def set_cached_parse(rule_hash: str, parsed: dict, ttl_seconds: int = 7 * 24 * 3600) -> None:
        key = f"mvel:cache:parse:{rule_hash}"
        r.setex(key, ttl_seconds, json.dumps(parsed))
        
    def set_cached_explanation(rule_hash: str, explanation: str, ttl_seconds: int = 24 * 3600) -> None:
        key = f"mvel:cache:explain:{rule_hash}"
        r.setex(key, ttl_seconds, explanation)
    

    trace = Trace(enabled=enable_trace)
    trace.log_step("start", {"mode": mode, "model": model, "inputs": len(mvel_texts)})

    # 2) Ask planner for the execution plan
    steps = plan_steps(llm, mode)
    trace.log_step("plan", {"steps": steps})

    # 3) Shared working state (short-term memory for this run)
    extractions: List[dict] = []
    context: str = ""
    english: str = ""           # holds the current natural-language output
    verdict: dict = {}          # holds verifier output when used
    static_issues: List[str] = []

    # 4) Execute the plan
    rule_hash: str | None = None
    for step in steps:
        if step == "parse":
            idx = len(extractions)
            if idx >= len(mvel_texts):
                trace.log_step("parse_skipped", {"reason": "no more inputs", "idx": idx})
                continue

            rule_hash = hash_text(mvel_texts[idx])
            extraction = parse_mvel_branches(mvel_texts[idx])

            # parse cache
            parsed = get_cached_parse(rule_hash)
            if parsed is None:
                parsed = extraction
                set_cached_parse(rule_hash, parsed)
            extractions.append(parsed)
            
            trace.log_step("parse", {
                "index": idx,
                "branches": len(extraction.get("branches", [])),
                "outputs": extraction.get("outputs", []),
            })

        elif step == "static_checks":
            if not extractions:
                static_issues = ["static_checks: no extraction available (parse not run yet)."]
            else:
                static_issues = run_static_checks(extractions[-1])

            trace.log_step("static_checks", {"issues": static_issues})

        elif step == "retrieve_context":
            # Simple RAG + memory context
            # Uses first MVEL text as query signal (good enough for POC)
            rag = retrieve_context(mvel_texts[0] if mvel_texts else "", kb_dir="dir")
            pieces = []
            if mem_context:
                pieces.append(mem_context)
            if rag:
                pieces.append(rag)
            if static_issues:
                pieces.append("Static check notes:\n" + "\n".join(f"- {x}" for x in static_issues))

            context = "\n\n".join(pieces).strip()
            trace.log_step("retrieve_context", {"context_chars": len(context)})

        elif step == "explain":
            if not extractions:
                english = "could not parse any rule branches from the provided MVEL."
                trace.log_step("explain_fallback", {"reason": "no extraction"})
                continue

            if rule_hash:
                cached = get_cached_explanation(rule_hash)
                if cached:
                    trace.log_step("explain_cache_hit", {"rule_hash": rule_hash})
                    return cached

            english = explain_rule(llm, extractions[-1], context)

            if rule_hash:
                set_cached_explanation(rule_hash, english)

            trace.log_step("explain", {"english_chars": len(english)})
            return english

            

        elif step == "verify":
            if not extractions or not english:
                verdict = {"ok": False, "missing": ["verify: missing extraction or english"], "rewrite_needed": True}
            else:
                verdict = verify_explanation(llm, extractions[-1], english)

            trace.log_step("verify", verdict)

        elif step == "rewrite":
            # Only rewrite if verifier says it's not OK
            if verdict.get("ok") is False and extractions and english:
                english = rewrite_explanation(llm, extractions[-1], english, verdict.get("missing", []))
                set_cached_explanation(rule_hash, english)
                trace.log_step("rewrite", {"english_chars": len(english)})
                return english
            else:
                trace.log_step("rewrite_skipped", {"ok": verdict.get("ok", True)})
        elif step == "reflect":
            try:
                r = reflect(llm, extractions[-1], english)
                trace.log_step("reflect", {"issues": len(r.issues)})
                save_memory_item({"type": "reflection_issue", "issues": r.issues})
                r.issues = "".join(r.issues)
                return r.issues
            
            except Exception:
                traceback.print_exc()
        elif step == "generate_tests":
            if not extractions:
                tests_json = [{"name": "error", "input": {}, "expected": {}, "note": "No extraction available"}]
            else:
                tests_json = generate_tests(llm, extractions[-1])

            english = json.dumps(tests_json, ensure_ascii=False, indent=2)
            trace.log_step("generate_tests", {"count": len(tests_json)})

        elif step == "diff":
            if len(extractions) < 2:
                english = "Diff requires two parsed rules, but fewer were available."
                trace.log_step("diff_fallback", {"reason": "need 2 extractions", "got": len(extractions)})
            else:
                english = diff_rules(llm, extractions[0], extractions[1])
                trace.log_step("diff", {"english_chars": len(english)})

        else:
            trace.log_step("unknown_step", {"step": step})

    # 5) Finalize and write trace
    if not english:
        english = "No output was produced. Check the plan and earlier steps."

    trace.finish(english)
    trace.write()
    return english
