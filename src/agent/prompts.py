from langchain_core.prompts import ChatPromptTemplate


# -------------------------
# PLANNER PROMPT
# -------------------------
# Decides WHICH STEPS the agent should run.
# Must return JSON only.
PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a planning agent. Output STRICT JSON only. "
        "Do not explain your reasoning."
    ),
    (
        "user",
        "You are given a user mode and a list of allowed steps.\n\n"
        "Allowed steps:\n"
        "- parse\n"
        "- static_checks\n"
        "- retrieve_context\n"
        "- explain\n"
        "- verify\n"
        "- rewrite\n"
        "- generate_tests\n"
        "- diff\n\n"
        "Rules:\n"
        "- mode=explain -> [parse, retrieve_context, explain]\n"
        "- mode=verify -> [parse, static_checks, retrieve_context, explain, verify, rewrite]\n"
        "- mode=tests -> [parse, generate_tests]\n"
        "- mode=diff -> [parse, parse, diff]\n"
        "- mode=agentic -> choose the safest default (like verify)\n\n"
        "Return JSON only in this format:\n"
        "{{ \"steps\": [\"step1\", \"step2\", ...] }}\n\n"
        "mode: {mode}"
    )
])


# -------------------------
# EXPLAINER PROMPT
# -------------------------
# Turns structured rule extraction into English
EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You write for non-technical stakeholders. "
        "Do not mention code, syntax, or programming terms."
    ),
    (
        "user",
        "Convert the following extracted rule structure into clear English.\n\n"
        "Requirements:\n"
        "- Start with a 1–2 sentence Summary\n"
        "- Then list Decision logic as bullet points\n"
        "- One bullet per branch, in order\n"
        "- Use 'Otherwise,' for the DEFAULT branch\n"
        "- Use provided context to define business terms if relevant\n\n"
        "Context:\n"
        "{context}\n\n"
        "Rule extraction (JSON):\n"
        "{extraction_json}"
    )
])


# -------------------------
# VERIFIER PROMPT
# -------------------------
# Checks if the English explanation matches the rule extraction
VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict QA verifier. "
        "Be precise. Output STRICT JSON only."
    ),
    (
        "user",
        "Compare the English explanation against the rule extraction.\n\n"
        "Check for:\n"
        "- Missing branches\n"
        "- Missing outputs\n"
        "- Incorrect conditions\n\n"
        "Return JSON only in this format:\n"
        "{{\n"
        "  \"ok\": true | false,\n"
        "  \"missing\": [\"description of missing item\", ...],\n"
        "  \"rewrite_needed\": true | false\n"
        "}}\n\n"
        "Rule extraction:\n"
        "{extraction_json}\n\n"
        "English explanation:\n"
        "{english}"
    )
])


# -------------------------
# REWRITE PROMPT
# -------------------------
# Fixes an explanation based on verifier feedback
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You rewrite explanations for non-technical stakeholders."
    ),
    (
        "user",
        "Rewrite the explanation so that it fully matches the rule extraction.\n\n"
        "Rules:\n"
        "- Cover ALL branches and outputs\n"
        "- Keep it concise and clear\n"
        "- Do not mention verification or errors\n\n"
        "Rule extraction:\n"
        "{extraction_json}\n\n"
        "Current explanation:\n"
        "{english}\n\n"
        "Missing or incorrect items:\n"
        "{missing}"
    )
])


# -------------------------
# DIFF PROMPT
# -------------------------
# Explains behavioral differences between two rules
DIFF_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a rules comparison analyst for non-technical stakeholders."
    ),
    (
        "user",
        "Compare the OLD rule and the NEW rule and explain how behavior changed.\n\n"
        "Output format:\n"
        "- Short summary of changes\n"
        "- Bullet points describing behavioral differences\n"
        "- Mention added, removed, or modified conditions\n\n"
        "OLD rule extraction:\n"
        "{old_json}\n\n"
        "NEW rule extraction:\n"
        "{new_json}"
    )
])


# -------------------------
# TEST GENERATION PROMPT
# -------------------------
# Generates test cases that cover all branches
TESTS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You generate test cases for business rules. Output STRICT JSON only (no markdown, no commentary)."
    ),
    (
        "user",
        "Given the rule extraction below, generate test cases that cover ALL branches.\n\n"
        "Requirements:\n"
        "2) Inputs must use the same field paths referenced in the rule conditions/actions (e.g., applicant.age).\n"
        "3) Expected must include ALL output assignments performed by the branch taken (do NOT return an empty expected if the rule assigns outputs).\n"
        "4) Prefer boundary values (e.g., equals threshold) where relevant.\n\n"
        "Return STRICT JSON array with this schema:\n"
        "[\n"
        "  {{\n"
        "    \"name\": \"...\",\n"
        "    \"input\": {{\"...\": \"...\"}},\n"
        "    \"expected\": {{\"...\": \"...\"}}\n"
        "  }}\n"
        "]\n\n"
        "Rule extraction:\n"
        "{extraction_json}"
    )
])

REFLECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a strict reviewer. Output STRICT JSON only."),
    ("user",
     "Check whether the English explanation fully matches the rule extraction.\n"
     "Return JSON like:\n"
     "{{\n"
     "  \"ok\": true,\n"
     "  \"issues\": [\"...\"]\n"
     "}}\n\n"
     "Rule extraction:\n{extraction_json}\n\n"
     "English explanation:\n{english}\n")
])