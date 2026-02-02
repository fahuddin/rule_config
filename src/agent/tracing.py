import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional

RUNS_DIR = "runs"

class Trace:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.run_id = str(uuid.uuid4())
        self.started_at = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.cache_hit: bool = False
        self.final_output: Optional[str] = None

    def log_step(self, name: str, data: Dict[str, Any], cache_hit: bool):
        self.steps.append({
            "name": name,
            "ts": time.time(),
            "cache_hit": cache_hit,
            "data": data
        })
    
    def reasons_ui(self):
        out = []
        for s in self.steps:
            name = s.get("name", "")
            if not name.endswith("_reason"):
                continue
            data = s.get("data", {}) or {}
            out.append({
                "step": name.replace("_reason", ""),
                "summary": data.get("summary", ""),
                "evidence": data.get("evidence", {}) or {},
                "ts": s.get("ts"),
            })
        return out
        
    def log_reason(self, step: str, summary: str, evidence: Dict[str, Any] | None = None):
        self.steps.append({
            "name": f"{step}_reason",
            "ts": time.time(),
            "cache_hit": None,
            "data": {
                "summary": summary,
                "evidence": evidence or {}
            }
        })
    def to_dict(self):
         dict(self.steps)

    def finish(self, output: str):
        self.final_output = output

    def write(self):
        if not self.enabled:
            return
        os.makedirs(RUNS_DIR, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": time.time(),
            "steps": self.steps,
            "cache_hit": self.cache_hit,
            "final_output": self.final_output,
        }
        path = os.path.join(RUNS_DIR, f"run_{self.run_id}.json")
        print(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            print(payload)
