import re
from typing import Any, Dict, List

# --- Comments ---
RE_LINE_COMMENT = re.compile(r"//.*?$", re.MULTILINE)
RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

# --- Tokens / identifiers ---
IDENT_RE = re.compile(r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*\b")

# --- Assignments (single statement) ---
ASSIGN_RE = re.compile(r"^\s*([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)\s*=\s*(.+?)\s*$")

# --- Strip string literals for brace counting / identifier extraction ---
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

def _split_statements(text: str) -> List[str]:
    """
    Split text into statements by ';' (lightweight).
    Note: does not perfectly handle semicolons inside strings, but OK for most business rules.
    """
    parts = [p.strip() for p in text.split(";")]
    return [p for p in parts if p]

def parse_mvel_branches(mvel_text: str) -> Dict[str, Any]:
    """
    Lightweight MVEL-ish parser:
    - Detects if / else if / else branches (including patterns like "} else if")
    - Tracks brace depth to handle nested blocks
    - Captures actions inside each branch:
        * assignments (tracks outputs)
        * non-assignment statements (e.g., addReason(...), list.add(...), return false)
    - Captures top-level statements outside branches as "globals"
    - FIXES:
        * Handles one-line blocks: if (...) { a; b; } tail;
        * Strips string literals before extracting identifiers (prevents "velocity" from quotes)
        * Captures trailing statements after inline block close as globals
    """
    src = strip_comments(mvel_text)

    branches: List[Dict[str, Any]] = []
    variables = set()
    outputs = set()
    globals_actions: List[str] = []

    lines = src.splitlines()
    i = 0

    def record_idents(text: str) -> None:
        # Remove string literals so words inside quotes don't become "variables"
        text = RE_STRINGS.sub("", text)
        for ident in IDENT_RE.findall(text):
            if ident not in KEYWORDS:
                variables.add(ident)

    def record_statement(stmt: str, actions_list: List[str]) -> None:
        m = ASSIGN_RE.match(stmt)
        if m:
            outputs.add(m.group(1).strip())
        actions_list.append(stmt)

    def parse_inline_block(line_text: str, actions_list: List[str]) -> None:
        """
        Extract and parse statements inside the first {...} on the same line,
        and also capture any trailing statements after the closing '}' as globals.
        """
        # Inside {...}
        inner = line_text[line_text.find("{") + 1 : line_text.rfind("}")].strip()
        if inner:
            for stmt in _split_statements(inner):
                record_idents(stmt)
                record_statement(stmt, actions_list)

        # After the closing '}': treat as top-level statements
        tail = line_text[line_text.rfind("}") + 1 :].strip()
        if tail:
            for stmt in _split_statements(tail):
                record_idents(stmt)
                record_statement(stmt, globals_actions)

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

            # If block is inline: if (...) { ... } (possibly with tail statements)
            if "{" in line and "}" in line:
                brace_depth = _count_braces(line)
                if brace_depth == 0:
                    parse_inline_block(line, actions)
                    branches.append({"condition": condition, "actions": actions})
                    i += 1
                    continue

            # Multi-line block
            brace_depth = _count_braces(line)
            i += 1

            while i < len(lines) and brace_depth > 0:
                lraw = lines[i].strip()
                brace_depth += _count_braces(lraw)

                # Remove outer braces, keep content
                content = lraw.strip().strip("{}").strip()
                if content:
                    for stmt in _split_statements(content):
                        record_idents(stmt)
                        record_statement(stmt, actions)

                i += 1

            branches.append({"condition": condition, "actions": actions})
            continue

        if is_else:
            actions: List[str] = []

            # Inline else: else { ... } tail;
            if "{" in line and "}" in line:
                brace_depth = _count_braces(line)
                if brace_depth == 0:
                    parse_inline_block(line, actions)
                    branches.append({"condition": "DEFAULT", "actions": actions})
                    i += 1
                    continue

            # Multi-line else block
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
        if line and not line.startswith(("def ", "def\t")):
            content = line.strip().strip("{}").strip()
            if content:
                for stmt in _split_statements(content):
                    record_idents(stmt)
                    record_statement(stmt, globals_actions)

        i += 1

    # Optional: remove outputs from variables to reduce noise
    for out in outputs:
        variables.discard(out)

    return {
        "globals": globals_actions,      # statements outside any if/else blocks
        "branches": branches,            # parsed conditional branches
        "variables": sorted(variables),  # referenced identifiers (approx)
        "outputs": sorted(outputs),      # assignment LHS fields
    }
