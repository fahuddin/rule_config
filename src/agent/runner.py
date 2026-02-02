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
    
    redis_client = MiniRedis(host="127.0.0.1", port=6379)

    def hash_text(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()
    
    def get_cached_explanation(rule_hash: str) -> str | None:
        key = f"mvel:cache:explain:{rule_hash}"
        raw = redis_client.get(key)
        if raw is None:
            return None
        return raw.decode("utf-8")
    
    def get_cached_parse(rule_hash: str) -> dict | None:
        key = f"mvel:cache:parse:{rule_hash}"
        raw = redis_client.get(key)
        if raw is None:
            return None
        return json.loads(raw.decode("utf-8"))
    
    def set_cached_parse(rule_hash: str, parsed: dict, ttl_seconds: int = 7 * 24 * 3600) -> None:
        key = f"mvel:cache:parse:{rule_hash}"
        redis_client.setex(key, ttl_seconds, json.dumps(parsed))
        
    def set_cached_explanation(rule_hash: str, explanation: str, ttl_seconds: int = 24 * 3600) -> None:
        key = f"mvel:cache:explain:{rule_hash}"
        redis_client.setex(key, ttl_seconds, explanation)
        
    def get_cached_context(rule_hash: str) -> str | None:
        key = f"mvel:cache:context:{rule_hash}"
        raw = redis_client.get(key)
        if raw is None:
            return None
        return raw.decode("utf-8")

    def set_cached_context(rule_hash: str, ctx: str, ttl_seconds: int) -> None:
        key = f"mvel:cache:context:{rule_hash}"
        redis_client.setex(key, ttl_seconds, ctx)
    

    trace = Trace(enabled=enable_trace)
    trace.log_step("start", {"mode": mode, "model": model, "inputs": len(mvel_texts)})
    trace.log_reason(
            "start",
            "Initialized run with deterministic LLM and loaded memory context.",
            {"mode": mode, "model": model, "inputs": len(mvel_texts), "has_mem_context": bool(mem_context)},
    )
    # 2) Ask planner for the execution plan
    steps = plan_steps(mode)
    trace.log_step("plan", {"steps": steps})
    trace.log_reason("plan", "Planner selected steps based on requested mode.", {"mode": mode, "steps": steps})

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
                trace.log_reason("parse", "Skipped parse because there were no more inputs.", {"idx": idx})
                continue

            rule_hash = hash_text(mvel_texts[idx])
            parsed = get_cached_parse(rule_hash)
            if parsed is None:
                parsed = parse_mvel_branches(mvel_texts[idx])
                set_cached_parse(rule_hash, parsed)
                cache_hit = False
                trace.log_reason(
                    "parse",
                    "No cached parse found; parsed MVEL and stored result in cache.",
                    {"rule_hash": rule_hash, "index": idx},
                )
            else:
                cache_hit = True
                trace.log_reason(
                    "parse",
                    "Used cached parse to avoid re-parsing MVEL.",
                    {"rule_hash": rule_hash, "index": idx},
                )
                
            extractions.append(parsed)
            
            trace.log_step("parse", {
                "index": idx,
                "branches": len(parsed.get("branches", [])),
                "outputs": parsed.get("outputs", []),
                "cache_hit": cache_hit
            })

        elif step == "static_checks":
            if not extractions:
                static_issues = ["static_checks: no extraction available (parse not run yet)."]
                trace.log_reason(
                    "static_checks",
                    "Static checks could not run because parse output was missing.",
                    {"issues": static_issues},
                )
            else:
                trace.log_reason(
                    "static_checks",
                    "Static checks could not run because parse output was missing.",
                    {"issues": static_issues},
                )
                static_issues = run_static_checks(extractions[-1])

            trace.log_step("static_checks", {"issues": static_issues})

        elif step == "retrieve_context":
            # Simple RAG + memory context
            # Uses first MVEL text as query signal (good enough for POC)
            if rule_hash:
                cached_ctx = get_cached_context(rule_hash)
                if cached_ctx:
                    context = cached_ctx
                    trace.log_step("retrieve_context_cache_hit", {"context_chars": len(context)})
                    trace.log_reason(
                        "retrieve_context",
                        "Skipped RAG because cached context exists for this rule.",
                        {"rule_hash": rule_hash, "context_chars": len(context)},
                    )
                    continue
                else:
                    trace.log_reason(
                        "retrieve_context",
                        "No cached context found; will build context using memory + RAG + static notes.",
                        {"rule_hash": rule_hash, "has_mem_context": bool(mem_context), "has_static_issues": bool(static_issues)},
                    )
                
                    rag = retrieve_context(mvel_texts[0] if mvel_texts else "", kb_dir="dir")
                    pieces = []
                    if mem_context:
                        pieces.append(mem_context)
                    if rag:
                        pieces.append(rag)
                    if static_issues:
                        pieces.append("Static check notes:\n" + "\n".join(f"- {x}" for x in static_issues))

                    context = "\n\n".join(pieces).strip()
                    if rule_hash:
                        set_cached_context(rule_hash, context, ttl_seconds=6 * 3600)
                        trace.log_reason(
                            "retrieve_context",
                            "Cached assembled context to avoid repeated RAG + formatting on future runs.",
                            {"rule_hash": rule_hash, "context_chars": len(context), "ttl_seconds": 6 * 3600},
                        )

                    trace.log_step("retrieve_context", {"context_chars": len(context)})

        elif step == "explain":
            if not extractions:
                english = "could not parse any rule branches from the provided MVEL."
                trace.log_reason(
                    "explain",
                    "Generated fallback message because no parsed extraction was available.",
                    {"fallback": True},
                )
                continue

            if rule_hash:
                cached = get_cached_explanation(rule_hash)
                if cached:
                    # use cached explanation but continue pipeline so downstream steps can run
                    english = cached
                    trace.log_step("explain_cache_hit", {"rule_hash": rule_hash})
                    trace.log_reason(
                        "explain",
                        "Used cached explanation to avoid an LLM call.",
                        {"rule_hash": rule_hash, "english_chars": len(english)},
                    )
                    continue
                else:
                    trace.log_reason(
                        "explain",
                        "No cached explanation found; calling LLM to generate explanation.",
                        {"rule_hash": rule_hash, "has_context": bool(context), "context_chars": len(context)},
                    ) 

            english = explain_rule(llm, extractions[-1], context)

            if rule_hash:
                set_cached_explanation(rule_hash, english)
                trace.log_reason(
                    "explain",
                    "Cached LLM-generated explanation for faster repeat requests.",
                    {"rule_hash": rule_hash, "english_chars": len(english)},
                )


            trace.log_step("explain", {"english_chars": len(english)})

            

        elif step == "verify":
            if not extractions or not english:
                verdict = {"ok": False, "missing": ["verify: missing extraction or english"], "rewrite_needed": True}
                trace.log_reason(
                    "verify",
                    "Verification failed because required inputs were missing.",
                    {"missing": verdict.get("missing", [])},
                )
            else:
                verdict = verify_explanation(llm, extractions[-1], english)
                if verdict.get("ok") is False:
                    trace.log_reason(
                        "verify",
                        "Verifier reported missing coverage; rewrite recommended.",
                        {"missing": verdict.get("missing", [])},
                    )
                else:
                    trace.log_reason("verify", "Verifier passed the explanation.", {})


            trace.log_step("verify", verdict)

        elif step == "rewrite":
            # Only rewrite if verifier says it's not OK
            if verdict.get("ok") is False and extractions and english:
                trace.log_reason(
                    "rewrite",
                    "Rewriting explanation to address verifier feedback.",
                    {"missing": verdict.get("missing", [])},
                )
                english = rewrite_explanation(llm, extractions[-1], english, verdict.get("missing", []))
                if rule_hash:
                    set_cached_explanation(rule_hash, english)
                    trace.log_reason(
                        "rewrite",
                        "Cached rewritten explanation for future runs.",
                        {"rule_hash": rule_hash, "english_chars": len(english)},
                    )
                trace.log_step("rewrite", {"english_chars": len(english)})
            else:
                trace.log_reason(
                    "rewrite",
                    "Skipped rewrite because verifier passed or prerequisites were missing.",
                    {"ok": verdict.get("ok", True), "has_extraction": bool(extractions), "has_english": bool(english)},
                )
                trace.log_step("rewrite_skipped", {"ok": verdict.get("ok", True)})
        elif step == "reflect":
            try:
                refl = reflect(llm, extractions[-1], english)
                trace.log_step("reflect", {"issues": len(refl.issues)})
                trace.log_reason(
                    "reflect",
                    "Generated reflection issues and saved them to memory.",
                    {"issue_count": len(refl.issues)},
                )
                save_memory_item({"type": "reflection_issue", "issues": refl.issues})
            except Exception as e:
                trace.log_reason("reflect", "Reflection step failed; continuing without reflection.", {"error": str(e)})
                traceback.print_exc()

        elif step == "generate_tests":
            if not extractions:
                tests_json = [{"name": "error", "input": {}, "expected": {}, "note": "No extraction available"}]
                trace.log_reason("generate_tests", "Generated error test case because parse output was missing.", {})
            else:
                tests_json = generate_tests(llm, extractions[-1])
                trace.log_reason("generate_tests", "Generated tests from parsed extraction using LLM.", {"count": len(tests_json)})

            english = json.dumps(tests_json, ensure_ascii=False, indent=2)
            trace.log_step("generate_tests", {"count": len(tests_json)})

        elif step == "diff":
            if len(extractions) < 2:
                english = "Diff requires two parsed rules, but fewer were available."
                trace.log_step("diff_fallback", {"reason": "need 2 extractions", "got": len(extractions)})
                trace.log_reason(
                    "diff",
                    "Diff skipped because fewer than two parsed extractions were available.",
                    {"got": len(extractions)},
                )
            else:
                english = diff_rules(llm, extractions[0], extractions[1])
                trace.log_step("diff", {"english_chars": len(english)})
                trace.log_reason("diff", "Computed diff explanation from two parsed rules.", {"english_chars": len(english)})

        else:
            trace.log_step("unknown_step", {"step": step})
            trace.log_reason("unknown_step", "Encountered an unrecognized planner step; recorded and continued.", {"step": step})

    # 5) Finalize and write trace
    if not english:
        english = "No output was produced. Check the plan and earlier steps."
        trace.log_reason("finish", "No output was produced; returning generic fallback message.", {})

    trace.finish(english)
    trace.write()
    return english