from typing import Any, Dict, List

def run_static_checks(extraction: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    branches = extraction.get("branches", [])

    if not branches:
        issues.append("No branches detected (parser may have failed).")
        return issues

    has_default = any(b.get("condition") == "DEFAULT" for b in branches)
    if not has_default:
        issues.append("No DEFAULT/else branch found; rule may be partial or relies on fallthrough.")

    for idx, b in enumerate(branches):
        actions = b.get("actions", [])
        if not actions:
            issues.append(f"Branch {idx} has no actions.")

    outputs = extraction.get("outputs", [])
    if not outputs:
        issues.append("No output assignments detected.")

    return issues
