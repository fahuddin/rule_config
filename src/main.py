import argparse
from agent.runner import run

def main():
    parser = argparse.ArgumentParser(description="MVEL -> English agentic system")
    parser.add_argument("--mode", default="agentic",
                        choices=["explain", "verify", "tests", "diff", "agentic", "reflect"],
                        help="Run mode")
    parser.add_argument("--model", default="llama3.1", help="Ollama model name")
    parser.add_argument("--trace", action="store_true", help="Write trace log to runs/")
    parser.add_argument("files", nargs="+", help="One MVEL file (or two for diff)")
    args = parser.parse_args()

    if args.mode == "diff" and len(args.files) != 2:
        raise SystemExit("diff mode requires two files: old.mvel new.mvel")
    if args.mode != "diff" and len(args.files) != 1:
        raise SystemExit(f"{args.mode} mode requires exactly one file")

    texts = []
    for p in args.files:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            texts.append(f.read())

    result = run(mode=args.mode, mvel_texts=texts, model=args.model, enable_trace=args.trace)
    print(result)

if __name__ == "__main__":
    main()
