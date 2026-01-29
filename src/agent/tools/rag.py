import os 
from typing import List, Tuple 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def read_files(kb_dir: str):
    kb_path = os.path.join(BASE_DIR, kb_dir)
    docs = []
    if not os.path.isdir(kb_path):
        return docs

    for name in os.listdir(kb_path):
        path = os.path.join(kb_path, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                docs.append((name, f.read()))
    return docs


#return context to llm
def retrieve_context(query_text: str, kb_dir: str = "dir", max_snippets: int = 5) -> str: 
    """ Simple keyword RAG (no embeddings): find lines containing query keywords. """ 
    docs = read_files(kb_dir) 
    if not docs: 
        return "" # naive keywords: long-ish tokens 
    tokens = [t.strip("(){}[];,.") for t in query_text.split()] 
    tokens = [t for t in tokens if len(t) >= 6] # reduce noise 
    hits = [] 
    for name, text in docs: 
        lines = text.splitlines()
        for ln in lines: 
            for tok in tokens: 
                if tok in ln: 
                    hits.append((name, ln.strip())) 
                    break # de-dupe 
                seen = set() 
                uniq = [] 
                for name, ln in hits: 
                    key = (name, ln) 
                    if key not in seen: 
                        seen.add(key) 
                        uniq.append((name, ln)) 
                uniq = uniq[:max_snippets] 
                if not uniq: 
                    return "" 
                out = ["Knowledge base snippets:"] 
                for name, ln in uniq: 
                    out.append(f"- ({name}) {ln}") 
                    return "\n".join(out)