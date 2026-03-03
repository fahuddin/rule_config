from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import re
import numpy as np
from langchain_ollama import OllamaEmbeddings

def normalize_english(s: str) -> str:
    s = s.replace("“", "'").replace("”", "'").replace("’", "'").replace("‘", "'")
    s = " ".join(s.split())
    s = re.sub(r"[.]+$", "", s.strip())
    return s

def cosine_sim(a: list[float], b: list[float]) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Choose an embedding model you have pulled in Ollama.
# Common choices: "nomic-embed-text", "mxbai-embed-large"
emb = OllamaEmbeddings(model="nomic-embed-text")  # or "mxbai-embed-large"

def english_similarity_embedding(a: str, b: str) -> float:
    a = normalize_english(a)
    b = normalize_english(b)
    va = emb.embed_query(a)
    vb = emb.embed_query(b)
    return cosine_sim(va, vb)

# ----------------------------
# 1) Load data from your CSV
# ----------------------------

@dataclass
class Example:
    id: str
    family_id: str          # rule_name
    human_english: str      # english
    gold_mvel: str          # rule_definition

df = pd.read_csv("dataset.csv")  # or "/mnt/data/dataset.csv" in your environment
examples: List[Example] = [
    Example(
        id=str(row["id"]),
        family_id=row["rule_name"],
        human_english=row["english"],
        gold_mvel=row["rule_definition"],
    )
    for _, row in df.iterrows()
]


# ----------------------------
# 2) Split: hold out whole families (OOD split)
# ----------------------------

def split_by_family_ood(
    data: List[Example],
    test_family_fraction: float = 0.34,
    seed: int = 42
) -> Tuple[List[Example], List[Example]]:
    rnd = random.Random(seed)
    families = sorted({x.family_id for x in data})
    rnd.shuffle(families)

    n_test = max(1, int(round(len(families) * test_family_fraction)))
    test_families = set(families[:n_test])

    train = [x for x in data if x.family_id not in test_families]
    test  = [x for x in data if x.family_id in test_families]
    return train, test



llm = ChatOllama(model="llama3.1", temperature=0.0)

english_to_mvel_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You convert English business rules into MVEL boolean expressions.\n"
        "Return ONLY the MVEL expression.\n"
        "No backticks. No explanations.\n"
        "Always use null checks before calling methods.\n"
        "Use equalsIgnoreCase for string comparison unless explicitly specified otherwise.\n"
        "Preserve AND/OR logic exactly.\n"
    ),

    # ---------------------------
    # Few-shot examples
    # ---------------------------

    (
        "human",
        "English rule:\n"
        "The rule passes when the message is 'MT', the estimate is 'Actual', "
        "and the client ID is either missing or set to 'N'.\n\n"
        "Output MVEL:"
    ),
    (
        "assistant",
        "(input.message != null && input.message.equalsIgnoreCase('MT')) && "
        "(input.estimate != null && input.estimate.equalsIgnoreCase('Actual')) && "
        "(input.clientId == null || input.clientId.equals('N'))"
    ),

    (
        "human",
        "English rule:\n"
        "The rule passes when the message is 'CASH' and the estimate code is 'Actual'.\n\n"
        "Output MVEL:"
    ),
    (
        "assistant",
        "(input.message != null && input.message.equalsIgnoreCase('CASH')) && "
        "(input.estimateCode != null && input.estimateCode.equalsIgnoreCase('Actual'))"
    ),

    (
        "human",
        "English rule:\n"
        "The rule passes when the message is 'ABC', the money code is 'Money', "
        "the work type is 'regular', and the transaction flag is 'Y'.\n\n"
        "Output MVEL:"
    ),
    (
        "assistant",
        "(input.message != null && input.message.equalsIgnoreCase('ABC')) && "
        "(input.moneyCode != null && input.moneyCode.equalsIgnoreCase('Money')) && "
        "(input.workType != null && input.workType.equalsIgnoreCase('regular')) && "
        "(input.transaction != null && input.transaction.equals('Y'))"
    ),

    (
        "human",
        "English rule:\n{english}\n\nOutput MVEL:"
    ),
])

english_to_mvel_chain = english_to_mvel_prompt | llm | StrOutputParser()

def english_to_mvel(english: str) -> str:
    return english_to_mvel_chain.invoke({"english": english}).strip()


mvel_to_english_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You translate MVEL boolean rules into plain English.\n"
        "Write 1-3 short sentences.\n"
        "Be precise about AND/OR logic, null checks, and negations.\n"
        "Do not mention MVEL.\n"
        "Do not add extra assumptions."
    ),

    # ---------------------------
    # Few-shot examples
    # ---------------------------

    (
        "human",
        "Rule:\n"
        "(input.message != null && input.message.equalsIgnoreCase('MT')) && "
        "(input.estimate != null && input.estimate.equalsIgnoreCase('Actual')) && "
        "(input.clientId == null || input.clientId.equals('N'))\n\n"
        "Plain English:"
    ),
    (
        "assistant",
        "The rule passes when the message is 'MT', the estimate is 'Actual', "
        "and the client ID is either missing or set to 'N'."
    ),

    (
        "human",
        "Rule:\n"
        "(input.message != null && input.message.equalsIgnoreCase('CASH')) && "
        "(input.estimateCode != null && input.estimateCode.equalsIgnoreCase('Actual'))\n\n"
        "Plain English:"
    ),
    (
        "assistant",
        "The rule passes when the message is 'CASH' and the estimate code is 'Actual'."
    ),

    (
        "human",
        "Rule:\n{mvel}\n\nPlain English:"
    ),
])

mvel_to_english_chain = mvel_to_english_prompt | llm | StrOutputParser()

def mvel_to_english(mvel: str) -> str:
    return mvel_to_english_chain.invoke({"mvel": mvel}).strip()



def evaluate_backtranslation(
    eval_examples: List[Example],
    gen_mvel: Callable[[str], str],
    translate: Callable[[str], str],
) -> List[Dict]:
    rows: List[Dict] = []
    for ex in eval_examples:
        pred_mvel = gen_mvel(ex.human_english)
        pred_english = translate(pred_mvel)
        sim = english_similarity_embedding(ex.human_english, pred_english)

        rows.append({
            "id": ex.id,
            "family_id": ex.family_id,
            "human_english": ex.human_english,
            "gold_mvel": ex.gold_mvel,
            "pred_mvel": pred_mvel,
            "pred_english": pred_english,
            "tfidf_cosine": round(sim, 3),
        })
    return rows


if __name__ == "__main__":
    train, test = split_by_family_ood(examples, test_family_fraction=0.34, seed=1)

    print("TRAIN families:", sorted({x.family_id for x in train}))
    print("TEST  families:", sorted({x.family_id for x in test}))
    print("Train size:", len(train), "Test size:", len(test))
    print()

    results = evaluate_backtranslation(test, english_to_mvel, mvel_to_english)

    # Pretty-print a few traces
    for r in results[:10]:
        print("----", r["id"], r["family_id"])
        print("HUMAN ENGLISH:", r["human_english"])
        print("GOLD MVEL:", r["gold_mvel"])
        print("PRED MVEL:", r["pred_mvel"])
        print("PRED ENGLISH:", r["pred_english"])
        print("Embeddings:", r["tfidf_cosine"])
        print()