import re
from typing import Any, Dict, List

# --- Comments ---
RE_LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

# --- Tokens / identifiers ---
IDENT_RE = re.compile(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*\b")

# --- Assignments (single statement) ---
ASSIGN_RE = re.compile(r"^\s*([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\s*=\s*(.+?)\s*$")

# --- Strip string literals for brace counting ---
RE_STRINGS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')

KEYWORDS = {
    "if", "else", "return", "true", "false", "null", "new",
    "for", "while", "switch", "case", "break", "continue", "def",
}

def strip_comments(src: str) -> str:
    src = RE_BLOCK_COMMENT.sub("", src)
    src = RE_LINE_COMMENT.sub("", src)
    return src

def _count_braces(s: str) -> int:
    # Remove string literals so braces inside quotes don't break depth
    s = RE_STRINGS.sub("", s)
    return s.count("{") - s.count("}")

def _split_statements(line: str) -> List[str]:
    """
    Split a line into statements by ';' without trying to fully parse MVEL.
    Good enough for most POC rules. Trims whitespace and ignores empties.
    """
    # NOTE: This does not handle semicolons inside strings perfectly,
    # but is usually fine for business-rule style scripts.
    parts = [p.strip() for p in line.split(";")]
    return [p for p in parts if p]

def parse_mvel_branches(mvel_text: str) -> Dict[str, Any]:
    """
    Lightweight MVEL-ish parser:
    - Detects if / else if / else branches (including patterns like "} else if")
    - Tracks brace depth to handle nested blocks
    - Captures actions inside each branch:
        * assignments (tracks outputs)
        * non-assignment statements (e.g., addReason(...), list.add(...))
    - Captures top-level statements outside branches as "globals"
    """
    src = strip_comments(mvel_text)

    branches: List[Dict[str, Any]] = []
    variables = set()
    outputs = set()
    globals_actions: List[str] = []

    lines = src.splitlines()
    i = 0

    def record_idents(text: str) -> None:
        for ident in IDENT_RE.findall(text):
            if ident not in KEYWORDS:
                variables.add(ident)

    def record_statement(stmt: str, actions_list: List[str]) -> None:
        # Track outputs for assignments and keep the statement either way
        m = ASSIGN_RE.match(stmt)
        if m:
            outputs.add(m.group(1).strip())
        actions_list.append(stmt)

    while i < len(lines):
        raw = lines[i].strip()
        # normalize: allow patterns like "} else if" / "} else"
        line = raw.lstrip("}").strip()

        # Record identifiers for visibility
        record_idents(line)

        is_if = line.startswith("if")
        is_elif = line.startswith("else if")
        is_else = line.startswith("else")

        if is_if or is_elif:
            condition = ""
            if "(" in line and ")" in line:
                condition = line[line.find("(") + 1 : line.rfind(")")].strip()

            actions: List[str] = []

            brace_depth = _count_braces(line)
            i += 1

            # Collect until this block closes
            while i < len(lines) and brace_depth > 0:
                lraw = lines[i].strip()
                brace_depth += _count_braces(lraw)

                # Remove outer braces, keep content
                content = lraw.strip().strip("{}").strip()
                if content:
                    # Support multiple statements on one line
                    for stmt in _split_statements(content):
                        record_idents(stmt)
                        record_statement(stmt, actions)

                i += 1

            branches.append({"condition": condition, "actions": actions})
            continue

        if is_else:
            actions = []
            brace_depth = _count_braces(line)
            i += 1

            while i < len(lines) and brace_depth > 0:
                lraw = lines[i].strip()
                brace_depth += _count_braces(lraw)

                content = lraw.strip().strip("{}").strip()
                if content:
                    for stmt in _split_statements(content):
                        record_idents(stmt)
                        record_statement(stmt, actions)

                i += 1

            branches.append({"condition": "DEFAULT", "actions": actions})
            continue

        # Capture top-level statements outside branches (globals/defaults/helpers)
        # Example: decision="DENY"; riskTier="HIGH"; reasonCodes=[];
        if line and not line.startswith(("def ", "def\t")):
            # Strip braces and split statements
            content = line.strip().strip("{}").strip()
            if content:
                for stmt in _split_statements(content):
                    record_idents(stmt)
                    record_statement(stmt, globals_actions)

        i += 1

    # Optional: remove outputs from variables to reduce noise
    for out in outputs:
        if out in variables:
            variables.remove(out)

    return {
        "globals": globals_actions,     # statements outside any if/else blocks
        "branches": branches,           # parsed conditional branches
        "variables": sorted(variables), # referenced identifiers (approx)
        "outputs": sorted(outputs),     # assignment LHS fields
    }


if __name__ == "__main__":
    demo = r'''
    // globals
    decision = "REVIEW"; reasons = [];

    if (applicant.age < 18) {
        decision = "DENY";
        reasons.add("UNDERAGE");
    } else if (fraudScore >= 80) {
        decision="DENY"; reasons.add("HIGH_FRAUD_SCORE");
    } else {
        decision = "APPROVE";
    }
    '''
    out = parse_mvel_branches(demo)
    print(out)
