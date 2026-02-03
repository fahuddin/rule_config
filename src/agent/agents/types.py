from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Step:
    name: str
    why: str 
    inputs: List[str]
    output: List[str]
    

@dataclass
class StepResult:
    step: Step
    status: str                 # "pass" | "fail" | "error"
    summary: str                # short explanation (UI-safe)
    evidence: Dict[str, Any]    # small facts (e.g., values used)
    outputs: Dict[str, Any]     # fields set (errorCode, errorMessage, etc.)
    
@dataclass
class AgentResult:
    ok: bool
    issues: List[str] = field(default_factory=list)
    
    
    