import time
import difflib
import html
import json
from datetime import datetime, timezone

import streamlit as st

from train_test import (
    mvel_to_english,
    english_to_mvel,
    english_similarity_embedding,
    normalize_english,
)


# -----------------------------
# Utilities
# -----------------------------
def html_word_diff(a: str, b: str) -> str:
    """Simple word-level diff with HTML highlights (HTML-escaped)."""
    a_words = a.split()
    b_words = b.split()
    out = []

    for token in difflib.ndiff(a_words, b_words):
        tag = token[:2]
        word = html.escape(token[2:])  # prevent HTML injection / broken markup

        if tag == "- ":
            out.append(f"<span style='background:#ffdddd'>{word}</span>")
        elif tag == "+ ":
            out.append(f"<span style='background:#ddffdd'>{word}</span>")
        elif tag == "? ":
            continue
        else:
            out.append(word)

    return " ".join(out)


def _call_maybe_with_models(fn, *args, llm_model=None, embed_model=None, **kwargs):
    """
    Calls fn with model kwargs if supported; otherwise retries without them.
    This lets the UI model inputs work without requiring you to change train_test signatures.
    """
    try:
        return fn(*args, llm_model=llm_model, embed_model=embed_model, **kwargs)
    except TypeError:
        return fn(*args, **kwargs)


def confidence_label(sim: float) -> tuple[str, str]:
    """
    Returns (label, help_text)
    """
    if sim >= 0.90:
        return "✅ High confidence", "Meaning appears preserved; safe to proceed with normal review."
    if sim >= 0.75:
        return "⚠️ Review required", "Potential meaning drift; manual review recommended."
    return "❌ Low confidence", "Meaning likely changed; do not approve without edits/tests."


def clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="English ↔ MVEL Playground", layout="wide")
st.title("English ↔ MVEL Playground")

# -----------------------------
# Sidebar: Models + Governance metadata
# -----------------------------
with st.sidebar:
    st.subheader("Models")
    llm_model = st.text_input("LLM model (Ollama)", value="llama3.1")
    embed_model = st.text_input("Embedding model (Ollama)", value="nomic-embed-text")
    show_debug = st.checkbox("Show debug", value=False)

    st.divider()
    st.subheader("Governance metadata")
    rule_name = st.text_input("Rule name", value="Untitled rule")
    rule_id = st.text_input("Rule ID", value="rule-001")
    owner = st.text_input("Owner", value="(your name)")
    policy_version = st.text_input("Policy version", value="v1")
    require_human_approval = st.checkbox("Require human approval before export", value=True)

    st.divider()
    st.subheader("Confidence thresholds")
    hi_thr = st.slider("High confidence ≥", 0.0, 1.0, 0.90, 0.01)
    review_thr = st.slider("Review required ≥", 0.0, 1.0, 0.75, 0.01)

# Ensure thresholds make sense
if review_thr > hi_thr:
    st.warning("Review threshold is higher than high-confidence threshold. Adjusting automatically.")
    review_thr = hi_thr


def confidence_label_custom(sim: float) -> tuple[str, str]:
    if sim >= hi_thr:
        return "✅ High confidence", "Meaning appears preserved; safe to proceed with normal review."
    if sim >= review_thr:
        return "⚠️ Review required", "Potential meaning drift; manual review recommended."
    return "❌ Low confidence", "Meaning likely changed; do not approve without edits/tests."


# -----------------------------
# Top “manager-friendly” narrative components
# -----------------------------
st.markdown(
    """
### What this demo shows
Convert **plain-English business rules** into **executable MVEL**, then validate meaning using:
- **Back-translation** (MVEL → English)
- **Similarity score** (English vs back-translation)
- **Word-level diff** (quick visual review)

**Why it matters:** faster rule authoring, fewer interpretation bugs, and better governance/auditability.
"""
)

with st.expander("Workflow (how it works)", expanded=True):
    st.markdown(
        """
1. **Input:** English business rule  
2. **Translate:** English → MVEL  
3. **Validate:** MVEL → English (back-translation)  
4. **Quality check:** Similarity score + word-level diff  
5. **Handoff:** Export MVEL + metadata (and optional tests)
"""
    )

# Demo examples (one-click)
EXAMPLES = {
    "Select an example…": "",
    "Simple threshold": "Approve when the applicant credit score is at least 720 and income is above 60000.",
    "Time window": "Reject when a chargeback occurs within 30 days of account opening.",
    "Multi-condition": "Pass when country is US or CA, user is over 18, and KYC status is VERIFIED.",
    "Exception": "Approve when the applicant has a guarantor, unless the applicant is on the sanctions list.",
}

demo_pick = st.selectbox("Quick demo examples", list(EXAMPLES.keys()), index=0)

# -----------------------------
# Main layout
# -----------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("English rule")

    # Let examples populate the text area without fighting user edits
    if "english_input" not in st.session_state:
        st.session_state.english_input = ""

    if demo_pick != "Select an example…" and EXAMPLES[demo_pick]:
        # Only overwrite if the box is empty OR the current content matches some prior example
        # (avoids clobbering user edits too aggressively)
        if not st.session_state.english_input.strip() or st.session_state.english_input.strip() in EXAMPLES.values():
            st.session_state.english_input = EXAMPLES[demo_pick]

    english = st.text_area(
        "Paste an English business rule",
        height=260,
        placeholder="The rule passes when ...",
        label_visibility="collapsed",
        key="english_input",
    )

    st.divider()
    run_clicked = st.button("Run", type="primary", disabled=not english.strip(), use_container_width=True)

with col2:
    st.subheader("Results")
    pred_mvel_box = st.container()
    pred_english_box = st.container()
    sim_box = st.container()
    timing_box = st.container()
    compare_box = st.container()
    export_box = st.container()

# -----------------------------
# Run + render
# -----------------------------
if run_clicked:
    eng = english.strip()

    try:
        t0 = time.time()
        pred_mvel = _call_maybe_with_models(
            english_to_mvel, eng, llm_model=llm_model, embed_model=embed_model
        )
        t1 = time.time()

        pred_english = _call_maybe_with_models(
            mvel_to_english, pred_mvel, llm_model=llm_model, embed_model=embed_model
        )
        t2 = time.time()

        sim = _call_maybe_with_models(
            english_similarity_embedding,
            eng,
            pred_english,
            llm_model=llm_model,
            embed_model=embed_model,
        )
        t3 = time.time()

    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

    # ---- MVEL output ----
    pred_mvel_box.markdown("**Predicted MVEL**")
    pred_mvel_box.code(pred_mvel, language="java")  # close enough highlighting for MVEL

    # ---- Back-translation ----
    pred_english_box.markdown("**Back-translated English**")
    pred_english_box.write(pred_english)

    # ---- Similarity + confidence ----
    try:
        sim_f = float(sim)
    except Exception:
        sim_f = 0.0

    sim_f01 = clamp01(sim_f)
    conf, conf_help = confidence_label_custom(sim_f)

    sim_box.metric("Similarity (English vs back-translation)", f"{sim_f:.3f}", conf, help=conf_help)
    sim_box.progress(sim_f01)

    timing_box.caption(
        f"Timings: to_mvel={int((t1 - t0) * 1000)}ms | "
        f"to_english={int((t2 - t1) * 1000)}ms | "
        f"embed={int((t3 - t2) * 1000)}ms"
    )

    # ---- English comparison ----
    compare_box.subheader("English comparison")
    c1, c2 = compare_box.columns(2, gap="large")
    c1.markdown("**Human English (input)**")
    c1.write(eng)

    c2.markdown("**Predicted English (back-translation)**")
    c2.write(pred_english)

    compare_box.subheader("Differences (word-level)")
    compare_box.markdown(html_word_diff(eng, pred_english), unsafe_allow_html=True)

    # ---- Optional: Debug normalization ----
    if show_debug:
        compare_box.divider()
        compare_box.subheader("Debug")
        compare_box.write("Normalized English:", normalize_english(eng))
        compare_box.write("Normalized Pred English:", normalize_english(pred_english))

    # -----------------------------
    # “Presentable” export / handoff components
    # -----------------------------
    export_box.subheader("Handoff / Export")

    # Simple gate: export disabled if low similarity OR human approval required and not checked.
    approve = export_box.checkbox("I approve this output for handoff", value=False)

    low_confidence = sim_f < review_thr
    blocked_by_conf = low_confidence
    blocked_by_approval = require_human_approval and not approve

    if blocked_by_conf:
        export_box.warning(
            f"Export is blocked because similarity {sim_f:.3f} is below the review threshold {review_thr:.2f}."
        )
    if blocked_by_approval:
        export_box.info("Export requires human approval (check the approval box above).")

    can_export = (not blocked_by_conf) and (not blocked_by_approval)

    # Create an export bundle (JSON) so it looks “real” in demos
    bundle = {
        "rule": {
            "id": rule_id,
            "name": rule_name,
            "owner": owner,
            "policy_version": policy_version,
        },
        "models": {"llm_model": llm_model, "embed_model": embed_model},
        "timestamps": {"generated_at_utc": datetime.now(timezone.utc).isoformat()},
        "inputs": {"english": eng},
        "outputs": {"mvel": pred_mvel, "back_translated_english": pred_english},
        "validation": {
            "similarity": sim_f,
            "confidence_band": conf,
            "thresholds": {"high_confidence": hi_thr, "review_required": review_thr},
        },
    }
    bundle_json = json.dumps(bundle, indent=2)

    b1, b2, b3 = export_box.columns([1, 1, 1], gap="small")

    b1.download_button(
        "⬇️ Download bundle (JSON)",
        data=bundle_json,
        file_name=f"{rule_id}.json",
        mime="application/json",
        disabled=not can_export,
        use_container_width=True,
    )

    b2.download_button(
        "⬇️ Download MVEL",
        data=pred_mvel,
        file_name=f"{rule_id}.mvel",
        mime="text/plain",
        disabled=not can_export,
        use_container_width=True,
    )

    b3.button(
        "📋 Copy MVEL (manual)",
        help="Streamlit can’t reliably copy to clipboard in all browsers. Use the code block copy button.",
        disabled=not can_export,
        use_container_width=True,
    )

    export_box.caption(
        "Tip: For demos, show that you can export a versioned JSON bundle for audit trails and downstream pipelines."
    )