import uuid
from typing import Any, Dict, List, Optional
from tracing import Trace



def log(trace, name: str, status: str = "ok", summary: str = "", **data):
    payload = {"status": status}
    if summary:
        payload["summary"] = summary
    payload.update(data)
    trace.log_step(name, payload)

