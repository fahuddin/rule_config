# MVEL → English Agent (Ollama + LangChain)

An agentic AI system that parses MVEL business rules, explains them in plain English, verifies correctness, generates test cases and self-corrects using reflection 

## What it does
- Parses MVEL `if / else if / else` rules
- Extracts decision branches
- Uses a local LLM via Ollama
- Outputs a human-readable explanation


🧠 Why This Is Agentic AI

This system demonstrates true agentic behavior:

🧭 Planning – dynamically selects execution steps

🛠️ Tool use – parser, RAG, checker etc.

🔍 Self-verification – checks its own explanations

🔁 Self-correction – rewrites when wrong

🪞 Reflection – critiques final output

🧠 Memory – persists lessons across runs

📊 Observability – full execution traces

This goes far beyond “prompt → response”.


## Project Structure
agentic_ai/
├── main.py                       # CLI entry point
├── agent/
│   ├── runner.py                 # Orchestrator (agent loop)
│   ├── llm.py                    # LLM loader (Ollama)
│   ├── memory.py                 # Persistent agent memory
│   ├── tracing.py                # Execution tracing
│   ├── types.py                  # Agent schemas / dataclasses
│   ├── agents/
│   │   ├── planner.py            # Planning agent
│   │   ├── explainer.py          # Rule explainer
│   │   ├── verifier.py           # Explanation verifier
│   │   ├── reflect.py            # Reflection / critique agent
│   │   ├── tests.py              # Test generation agent
│   │   └── diff.py               # Rule diff agent
│   └── tools/
│       ├── mvel_parser_tool.py   # MVEL rule parser
│       ├── static_checker_tool.py# Static rule checks
│       ├── rag.py                # RAG retrieval logic
│       └── dir/                  # Knowledge base for RAG
│           └── rules.md
├── runs/                         # Execution traces (JSON)
├── examples/
│   └── rule.mvel                 # Sample MVEL rule
└── README.md


s

## Run
```bash
python main.py examples/sample.mvel


python main.py --mode explain examples/rule.mvel
Verify explanation fidelity
python main.py --mode verify examples/rule.mvel
Generate test cases
python main.py --mode tests examples/rule.mvel
Diff two rules
python main.py --mode diff old.mvel new.mvel