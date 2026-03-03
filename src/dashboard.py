import time
import difflib
import streamlit as st
from train_test import (
    mvel_to_english,
    english_to_mvel,
    english_similarity_embedding,
    normalize_english,
)

def html_word_diff(a: str, b: str) -> str:
    """Simple word-level diff with HTML highlights."""
    a_words = a.split()
    b_words = b.split()
    out = []
    for token in difflib.ndiff(a_words, b_words):
        if token.startswith("- "):
            out.append(f"<span style='background:#ffdddd'>{token[2:]}</span>")
        elif token.startswith("+ "):
            out.append(f"<span style='background:#ddffdd'>{token[2:]}</span>")
        elif token.startswith("? "):
            continue
        else:
            out.append(token[2:])
    return " ".join(out)

st.set_page_config(page_title="English ↔ MVEL Playground", layout="wide")
st.title("English ↔ MVEL Playground")

with st.sidebar:
    st.subheader("Models")
    llm_model = st.text_input("LLM model (Ollama)", value="llama3.1")
    embed_model = st.text_input("Embedding model (Ollama)", value="nomic-embed-text")
    show_debug = st.checkbox("Show debug", value=False)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("English rule")
    english = st.text_area(
        "Paste an English business rule",
        height=260,
        placeholder="The rule passes when ...",
        label_visibility="collapsed",
    )
    run = st.button("Run", type="primary", disabled=not english.strip())

with col2:
    st.subheader("Results")
    pred_mvel_box = st.empty()
    pred_english_box = st.empty()
    sim_box = st.empty()
    timing_box = st.empty()
    compare_box = st.empty()

if run:
    eng = english.strip()

    t0 = time.time()
    pred_mvel = english_to_mvel(eng)  # uses your ChatOllama chain
    t1 = time.time()
    pred_english = mvel_to_english(pred_mvel)
    t2 = time.time()
    sim = english_similarity_embedding(eng, pred_english)
    t3 = time.time()

    pred_mvel_box.code(pred_mvel, language="java")  # close enough for MVEL highlighting
    pred_english_box.write(pred_english)

    sim_box.metric("Similarity (English vs back-translation)", f"{sim:.3f}")
    st.progress(min(max(sim, 0.0), 1.0))

    timing_box.caption(
        f"Timings: to_mvel={int((t1 - t0) * 1000)}ms | "
        f"to_english={int((t2 - t1) * 1000)}ms | "
        f"embed={int((t3 - t2) * 1000)}ms"
    )

    # --- English comparison (human vs predicted) ---
    compare_box.subheader("English comparison")

    c1, c2 = compare_box.columns(2, gap="large")
    c1.markdown("**Human English (input)**")
    c1.write(eng)

    c2.markdown("**Predicted English (back-translation)**")
    c2.write(pred_english)

    compare_box.subheader("Differences (word-level)")
    compare_box.markdown(html_word_diff(eng, pred_english), unsafe_allow_html=True)

    if show_debug:
        compare_box.divider()
        compare_box.subheader("Debug")
        compare_box.write("Normalized English:", normalize_english(eng))
        compare_box.write("Normalized Pred English:", normalize_english(pred_english))