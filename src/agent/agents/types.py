from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Step:
    name: str
    why: str 
    inputs: List[str]
    output: List[str]
    
@dataclass
class AgentResult:
    ok: bool
    issues: List[str] = field(default_factory=list)
    
    
    