"""
Model Evaluation: Claude vs Ollama
Compares English-to-MVEL translation quality using embedding similarity scores.
"""

import os
import json
import time
import sys
import traceback
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Callable, Optional, Tuple
from datetime import datetime

import anthropic
import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dotenv import load_dotenv
import os

load_dotenv()

# ===========================
# Data Models
# ===========================

@dataclass
class Example:
    id: str
    family_id: str
    human_english: str
    gold_mvel: str


@dataclass
class EvalResult:
    example_id: str
    family_id: str
    model_name: str
    human_english: str
    gold_mvel: str
    pred_mvel: str
    pred_english: str
    embedding_similarity: float
    generation_time: float


# ===========================
# Utilities
# ===========================

def normalize_english(s: str) -> str:
    """Normalize English text for comparison."""
    s = s.replace("“", "'").replace("”", "'").replace("’", "'").replace("‘", "'")
    s = " ".join(s.split())
    s = s.rstrip(".")
    return s.strip()


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ===========================
# Embedding & Language Models
# ===========================

class EmbeddingClient:
    """Wrapper for embedding operations."""
    
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        self.embedder = OllamaEmbeddings(model=model)
    
    def embed(self, text: str) -> list[float]:
        """Embed a text string."""
        normalized = normalize_english(text)
        return self.embedder.embed_query(normalized)
    
    def similarity(self, text_a: str, text_b: str) -> float:
        """Calculate similarity between two texts."""
        va = self.embed(text_a)
        vb = self.embed(text_b)
        return cosine_sim(va, vb)


class LLMClient:
    """Abstract base for LLM clients."""
    
    def english_to_mvel(self, english: str) -> str:
        raise NotImplementedError
    
    def mvel_to_english(self, mvel: str) -> str:
        raise NotImplementedError


class OllamaClient(LLMClient):
    """Ollama LLM client (via LangChain)."""
    
    def __init__(self, model: str = "llama3.1", temperature: float = 0.0):
        self.model_name = model
        self.llm = ChatOllama(model=model, temperature=temperature)
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup prompt templates for Ollama."""
        self.english_to_mvel_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You convert English business rules into MVEL boolean expressions.\n"
                "Return ONLY the MVEL expression.\n"
                "No backticks. No explanations.\n"
                "Always use null checks before calling methods.\n"
                "Use equalsIgnoreCase for string comparison unless explicitly specified otherwise.\n"
                "Preserve AND/OR logic exactly.\n"
            ),
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
                "English rule:\n{english}\n\nOutput MVEL:"
            ),
        ])
        
        self.mvel_to_english_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You translate MVEL boolean rules into plain English.\n"
                "Write 1-3 short sentences.\n"
                "Be precise about AND/OR logic, null checks, and negations.\n"
                "Do not mention MVEL.\n"
                "Do not add extra assumptions."
            ),
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
                "Rule:\n{mvel}\n\nPlain English:"
            ),
        ])
        
        self.english_to_mvel_chain = self.english_to_mvel_prompt | self.llm | StrOutputParser()
        self.mvel_to_english_chain = self.mvel_to_english_prompt | self.llm | StrOutputParser()
    
    def english_to_mvel(self, english: str) -> str:
        return self.english_to_mvel_chain.invoke({"english": english}).strip()
    
    def mvel_to_english(self, mvel: str) -> str:
        return self.mvel_to_english_chain.invoke({"mvel": mvel}).strip()


class ClaudeClient(LLMClient):
    """Claude LLM client (via Anthropic SDK)."""
    
    def __init__(self, model: str = "claude-sonnet-4-6", temperature: float = 0.0):
        self.model_name = model
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.temperature = temperature
    
    def english_to_mvel(self, english: str) -> str:
        """Convert English rules to MVEL using Claude."""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=self.temperature,
            system=(
                "You convert English business rules into MVEL boolean expressions.\n"
                "Return ONLY the MVEL expression.\n"
                "No backticks. No explanations.\n"
                "Always use null checks before calling methods.\n"
                "Use equalsIgnoreCase for string comparison unless explicitly specified otherwise.\n"
                "Preserve AND/OR logic exactly.\n"
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "English rule:\n"
                        "The rule passes when the message is 'MT', the estimate is 'Actual', "
                        "and the client ID is either missing or set to 'N'.\n\n"
                        "Output MVEL:"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "(input.message != null && input.message.equalsIgnoreCase('MT')) && "
                        "(input.estimate != null && input.estimate.equalsIgnoreCase('Actual')) && "
                        "(input.clientId == null || input.clientId.equals('N'))"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "English rule:\n"
                        "The rule passes when the message is 'CASH' and the estimate code is 'Actual'.\n\n"
                        "Output MVEL:"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "(input.message != null && input.message.equalsIgnoreCase('CASH')) && "
                        "(input.estimateCode != null && input.estimateCode.equalsIgnoreCase('Actual'))"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "English rule:\n"
                        "The rule passes when the message is 'ABC', the money code is 'Money', "
                        "the work type is 'regular', and the transaction flag is 'Y'.\n\n"
                        "Output MVEL:"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "(input.message != null && input.message.equalsIgnoreCase('ABC')) && "
                        "(input.moneyCode != null && input.moneyCode.equalsIgnoreCase('Money')) && "
                        "(input.workType != null && input.workType.equalsIgnoreCase('regular')) && "
                        "(input.transaction != null && input.transaction.equals('Y'))"
                    )
                },
                {
                    "role": "user",
                    "content": f"English rule:\n{english}\n\nOutput MVEL:"
                }
            ]
        )
        return message.content[0].text.strip()
    
    def mvel_to_english(self, mvel: str) -> str:
        """Translate MVEL to English using Claude."""
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=512,
            temperature=self.temperature,
            system=(
                "You translate MVEL boolean rules into plain English.\n"
                "Write 1-3 short sentences.\n"
                "Be precise about AND/OR logic, null checks, and negations.\n"
                "Do not mention MVEL.\n"
                "Do not add extra assumptions."
            ),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Rule:\n"
                        "(input.message != null && input.message.equalsIgnoreCase('MT')) && "
                        "(input.estimate != null && input.estimate.equalsIgnoreCase('Actual')) && "
                        "(input.clientId == null || input.clientId.equals('N'))\n\n"
                        "Plain English:"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "The rule passes when the message is 'MT', the estimate is 'Actual', "
                        "and the client ID is either missing or set to 'N'."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Rule:\n"
                        "(input.message != null && input.message.equalsIgnoreCase('CASH')) && "
                        "(input.estimateCode != null && input.estimateCode.equalsIgnoreCase('Actual'))\n\n"
                        "Plain English:"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "The rule passes when the message is 'CASH' and the estimate code is 'Actual'."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Rule:\n"
                        "(input.message != null && input.message.equalsIgnoreCase('ABC')) && "
                        "(input.moneyCode != null && input.moneyCode.equalsIgnoreCase('Money')) && "
                        "(input.workType != null && input.workType.equalsIgnoreCase('regular')) && "
                        "(input.transaction != null && input.transaction.equals('Y'))\n\n"
                        "Plain English:"
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "The rule passes when the message is 'ABC', the money code is 'Money', "
                        "the work type is 'regular', and the transaction flag is 'Y'."
                    )
                },
                {
                    "role": "user",
                    "content": f"Rule:\n{mvel}\n\nPlain English:"
                }
            ]
        )
        return message.content[0].text.strip()


# ===========================
# Evaluation Functions
# ===========================

def load_examples_from_csv(csv_path: str = "dataset.csv") -> List[Example]:
    """Load examples from CSV file."""
    df = pd.read_csv(csv_path)
    examples = [
        Example(
            id=str(row["id"]),
            family_id=row["rule_name"],
            human_english=row["english"],
            gold_mvel=row["rule_definition"],
        )
        for _, row in df.iterrows()
    ]
    return examples


def validate_model(model: LLMClient, model_name: str) -> bool:
    """Test if model works with a simple example."""
    test_english = "The rule passes when the message is MT."
    try:
        print(f"\n[{model_name}] Testing connectivity...")
        mvel = model.english_to_mvel(test_english)
        print(f"  ✓ english_to_mvel: {mvel[:50]}...")
        english = model.mvel_to_english(mvel)
        print(f"  ✓ mvel_to_english: {english[:50]}...")
        print(f"  ✓ {model_name} is ready.\n")
        return True
    except Exception as e:
        print(f"  ✗ {model_name} FAILED:")
        print(f"    {type(e).__name__}: {e}")
        traceback.print_exc()
        print()
        return False
        return True
    except Exception as e:
        print(f"  ✗ {model_name} FAILED:")
        print(f"    {type(e).__name__}: {e}")
        traceback.print_exc()
        print()
        return False


def evaluate_model(
    model: LLMClient,
    embedder: EmbeddingClient,
    examples: List[Example],
    model_name: str,
) -> List[EvalResult]:
    """Evaluate a single model on all examples."""
    results = []
    
    for i, example in enumerate(examples):
        print(f"  [{model_name}] Evaluating {i+1}/{len(examples)}: {example.id}")
        
        try:
            # English to MVEL
            t0 = time.time()
            pred_mvel = model.english_to_mvel(example.human_english)
            t_mvel = time.time() - t0
            
            # MVEL to English (back-translation)
            t0 = time.time()
            pred_english = model.mvel_to_english(pred_mvel)
            t_english = time.time() - t0
            
            # Calculate similarity
            t0 = time.time()
            sim = embedder.similarity(example.human_english, pred_english)
            t_embed = time.time() - t0
            
            total_time = t_mvel + t_english + t_embed
            
            result = EvalResult(
                example_id=example.id,
                family_id=example.family_id,
                model_name=model_name,
                human_english=example.human_english,
                gold_mvel=example.gold_mvel,
                pred_mvel=pred_mvel,
                pred_english=pred_english,
                embedding_similarity=round(sim, 4),
                generation_time=round(total_time, 3),
            )
            results.append(result)
        
        except Exception as e:
            print(f"    ERROR on {example.id}: {type(e).__name__}: {e}")
            traceback.print_exc()
            result = EvalResult(
                example_id=example.id,
                family_id=example.family_id,
                model_name=model_name,
                human_english=example.human_english,
                gold_mvel=example.gold_mvel,
                pred_mvel="ERROR",
                pred_english="ERROR",
                embedding_similarity=0.0,
                generation_time=0.0,
            )
            results.append(result)
    
    return results


def compute_metrics(results: List[EvalResult]) -> Dict:
    """Compute aggregate metrics."""
    if not results:
        return {}
    
    sims = [r.embedding_similarity for r in results if r.embedding_similarity > 0]
    times = [r.generation_time for r in results if r.generation_time > 0]
    
    if not sims:
        return {
            "count": len(results),
            "errors": len(results),
            "avg_similarity": 0.0,
            "median_similarity": 0.0,
            "std_similarity": 0.0,
            "min_similarity": 0.0,
            "max_similarity": 0.0,
        }
    
    return {
        "count": len(results),
        "errors": len(results) - len(sims),
        "avg_similarity": round(np.mean(sims), 4),
        "median_similarity": round(np.median(sims), 4),
        "std_similarity": round(np.std(sims), 4),
        "min_similarity": round(np.min(sims), 4),
        "max_similarity": round(np.max(sims), 4),
        "avg_time_seconds": round(np.mean(times), 3),
    }


def compare_models(
    results_by_model: Dict[str, List[EvalResult]]
) -> pd.DataFrame:
    """Create comparison dataframe."""
    rows = []
    
    for model_name, results in results_by_model.items():
        metrics = compute_metrics(results)
        metrics["model"] = model_name
        rows.append(metrics)
    
    return pd.DataFrame(rows)


def build_results_df(results: List[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "id": r.example_id,
            "family": r.family_id,
            "model": r.model_name,
            "human_english": r.human_english,
            "gold_mvel": r.gold_mvel,
            "pred_mvel": r.pred_mvel,
            "pred_english": r.pred_english,
            "embedding_similarity": r.embedding_similarity,
            "time_seconds": r.generation_time,
        }
        for r in results
    ])


def build_plotly_figures(
    comparison_df: pd.DataFrame,
    ollama_df: pd.DataFrame,
    claude_df: pd.DataFrame,
) -> Tuple[go.Figure, go.Figure]:
    combined = pd.concat([ollama_df, claude_df], ignore_index=True)

    metric_names = [
        "avg_similarity",
        "median_similarity",
        "min_similarity",
        "max_similarity",
    ]
    metric_labels = {
        "avg_similarity": "Average Similarity",
        "median_similarity": "Median Similarity",
        "min_similarity": "Minimum Similarity",
        "max_similarity": "Maximum Similarity",
    }

    bar_rows = []
    for _, row in comparison_df.iterrows():
        for metric in metric_names:
            bar_rows.append({
                "Model": row["model"],
                "Metric": metric_labels[metric],
                "Score": row.get(metric, 0.0),
            })
    bar_df = pd.DataFrame(bar_rows)

    dashboard_fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Overall Similarity Comparison",
            "Similarity Distribution by Model",
            "Average Response Time (seconds)",
            "Top 5 Lowest-Scoring Examples",
        ),
        specs=[[{"type": "bar"}, {"type": "box"}], [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    for model in bar_df["Model"].unique():
        mdf = bar_df[bar_df["Model"] == model]
        dashboard_fig.add_trace(
            go.Bar(x=mdf["Metric"], y=mdf["Score"], name=model, text=mdf["Score"].round(3), textposition="outside"),
            row=1,
            col=1,
        )

    for model in combined["model"].unique():
        mdf = combined[combined["model"] == model]
        dashboard_fig.add_trace(
            go.Box(
                y=mdf["embedding_similarity"],
                name=model,
                boxmean=True,
                boxpoints="outliers",
            ),
            row=1,
            col=2,
        )

    time_df = comparison_df[["model", "avg_time_seconds"]].copy()
    dashboard_fig.add_trace(
        go.Bar(
            x=time_df["model"],
            y=time_df["avg_time_seconds"],
            text=time_df["avg_time_seconds"].round(2),
            textposition="outside",
            name="Average Time",
            marker_color="#636EFA",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    low5 = (
        combined[["id", "model", "embedding_similarity"]]
        .sort_values("embedding_similarity", ascending=True)
        .head(5)
        .copy()
    )
    low5["label"] = low5["model"] + " | ID " + low5["id"].astype(str)
    dashboard_fig.add_trace(
        go.Bar(
            x=low5["label"],
            y=low5["embedding_similarity"],
            text=low5["embedding_similarity"].round(3),
            textposition="outside",
            marker_color="#EF553B",
            name="Lowest Scores",
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    dashboard_fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    dashboard_fig.update_yaxes(range=[0, 1.05], row=1, col=2)
    dashboard_fig.update_yaxes(title_text="Seconds", row=2, col=1)
    dashboard_fig.update_yaxes(range=[0, 1.05], row=2, col=2)
    dashboard_fig.update_layout(
        title_text="Model Evaluation Dashboard (Non-Technical View)",
        template="plotly_white",
        height=900,
        width=1400,
        legend_title_text="Model",
        margin=dict(t=80, l=50, r=30, b=40),
    )

    summary = comparison_df[["model", "avg_similarity", "median_similarity", "errors", "avg_time_seconds"]].copy()
    summary = summary.rename(columns={
        "model": "Model",
        "avg_similarity": "Avg Similarity",
        "median_similarity": "Median Similarity",
        "errors": "Errors",
        "avg_time_seconds": "Avg Time (s)",
    })

    summary_fig = px.bar(
        summary,
        x="Model",
        y="Avg Similarity",
        text=summary["Avg Similarity"].round(3),
        color="Model",
        title="Executive Summary: Average Similarity by Model",
    )
    summary_fig.update_layout(template="plotly_white")
    summary_fig.update_yaxes(range=[0, 1.05])

    return summary_fig, dashboard_fig


def generate_plotly_report(
    comparison_df: pd.DataFrame,
    ollama_df: pd.DataFrame,
    claude_df: pd.DataFrame,
    output_html: str,
) -> None:
    """Generate stakeholder-friendly Plotly HTML report."""
    summary_fig, fig = build_plotly_figures(comparison_df, ollama_df, claude_df)

    with open(output_html, "w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Model Evaluation Dashboard</title></head><body>")
        f.write("<h1>Claude vs Ollama Evaluation</h1>")
        f.write("<p>This dashboard summarizes model quality and consistency using embedding similarity scores.</p>")
        f.write(summary_fig.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write(fig.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")


def run_evaluation_pipeline(log_fn=print) -> Dict[str, object]:
    log_fn("=" * 70)
    log_fn("MODEL EVALUATION: Claude vs Ollama")
    log_fn("=" * 70)

    log_fn("Loading examples from dataset.csv...")
    examples = load_examples_from_csv()
    log_fn(f"  Loaded {len(examples)} examples\n")

    log_fn("Initializing LLM and embedding clients...")
    ollama_llm = OllamaClient(model="llama3.1", temperature=0.0)
    claude_llm = ClaudeClient(model="claude-sonnet-4-6", temperature=0.0)
    embedder = EmbeddingClient(model="nomic-embed-text")
    log_fn("  Clients ready.\n")

    log_fn("Validating model connectivity...")
    ollama_ok = validate_model(ollama_llm, "Ollama")
    claude_ok = validate_model(claude_llm, "Claude")

    results_by_model = {}

    if ollama_ok:
        log_fn("Evaluating Ollama...")
        results_ollama = evaluate_model(ollama_llm, embedder, examples, "Ollama")
        results_by_model["Ollama"] = results_ollama
    else:
        log_fn("⚠ Skipping Ollama (connection failed)")
        results_ollama = []

    if claude_ok:
        log_fn("Evaluating Claude...")
        results_claude = evaluate_model(claude_llm, embedder, examples, "Claude")
        results_by_model["Claude"] = results_claude
    else:
        log_fn("⚠ Skipping Claude (connection failed)")
        results_claude = []

    comparison_df = compare_models(results_by_model)
    return {
        "comparison_df": comparison_df,
        "results_ollama": results_ollama,
        "results_claude": results_claude,
    }


def run_streamlit_app() -> None:
    st.set_page_config(page_title="Claude vs Ollama Evaluation", layout="wide")
    st.title("Claude vs Ollama Evaluation Dashboard")
    st.caption("Live embedding-score comparison for non-technical stakeholders.")

    with st.sidebar:
        st.subheader("Controls")
        run_eval = st.button("▶️ Run Evaluation", type="primary", use_container_width=True)

    # Default view
    if not run_eval:
        st.divider()
        st.header("Getting Started")
        st.write("""
        This dashboard evaluates and compares **Claude** and **Ollama** models on their ability to:
        1. Convert English business rules → MVEL expressions
        2. Translate MVEL back to English
        3. Measure semantic similarity between original and back-translated text
        
        **Click the "▶️ Run Evaluation" button in the sidebar to start.**
        """)
        
        st.header("What You'll See")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📊 Metrics")
            st.write("- Similarity scores (avg, median, min, max)")
            st.write("- Response times")
            st.write("- Error counts")
        with col2:
            st.subheader("📈 Charts")
            st.write("- Model comparison bar chart")
            st.write("- Score distribution (box plot)")
            st.write("- Low-scoring examples")
        with col3:
            st.subheader("📋 Data")
            st.write("- Detailed results table")
            st.write("- CSV exports")
            st.write("- JSON metrics")
        
        st.divider()
        st.info("⚠️ Make sure your `.env` file has `ANTHROPIC_API_KEY` set for Claude to work.")
        return

    # Evaluation in progress
    with st.spinner("Running evaluation across both models..."):
        payload = run_evaluation_pipeline(log_fn=lambda _: None)

        comparison_df = payload["comparison_df"]
        results_ollama = payload["results_ollama"]
        results_claude = payload["results_claude"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ollama_df = build_results_df(results_ollama)
        claude_df = build_results_df(results_claude)

        ollama_csv = f"eval_ollama_{timestamp}.csv"
        claude_csv = f"eval_claude_{timestamp}.csv"
        metrics_file = f"eval_metrics_{timestamp}.json"
        report_file = f"eval_dashboard_{timestamp}.html"

        ollama_df.to_csv(ollama_csv, index=False)
        claude_df.to_csv(claude_csv, index=False)

        metrics_json = {
            "timestamp": timestamp,
            "comparison": comparison_df.to_dict(orient="records"),
            "ollama_metrics": compute_metrics(results_ollama),
            "claude_metrics": compute_metrics(results_claude),
        }
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, indent=2)

        generate_plotly_report(comparison_df, ollama_df, claude_df, report_file)
        summary_fig, dashboard_fig = build_plotly_figures(comparison_df, ollama_df, claude_df)

    st.success("✅ Evaluation complete. Charts and files exported.")

    # Display detailed metrics
    st.header("📊 Detailed Metrics")
    
    ollama_metrics = compute_metrics(results_ollama)
    claude_metrics = compute_metrics(results_claude)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🦙 Ollama (llama3.1)")
        st.metric("Examples Evaluated", ollama_metrics.get("count", 0))
        st.metric("Evaluation Errors", ollama_metrics.get("errors", 0))
        st.metric("Average Similarity", f"{ollama_metrics.get('avg_similarity', 0):.4f}")
        st.metric("Median Similarity", f"{ollama_metrics.get('median_similarity', 0):.4f}")
        st.metric("Std Dev Similarity", f"{ollama_metrics.get('std_similarity', 0):.4f}")
        st.metric("Min Similarity", f"{ollama_metrics.get('min_similarity', 0):.4f}")
        st.metric("Max Similarity", f"{ollama_metrics.get('max_similarity', 0):.4f}")
        st.metric("Avg Response Time (s)", f"{ollama_metrics.get('avg_time_seconds', 0):.3f}")

    with col2:
        st.subheader("🤖 Claude (Sonnet 3.5)")
        st.metric("Examples Evaluated", claude_metrics.get("count", 0))
        st.metric("Evaluation Errors", claude_metrics.get("errors", 0))
        st.metric("Average Similarity", f"{claude_metrics.get('avg_similarity', 0):.4f}")
        st.metric("Median Similarity", f"{claude_metrics.get('median_similarity', 0):.4f}")
        st.metric("Std Dev Similarity", f"{claude_metrics.get('std_similarity', 0):.4f}")
        st.metric("Min Similarity", f"{claude_metrics.get('min_similarity', 0):.4f}")
        st.metric("Max Similarity", f"{claude_metrics.get('max_similarity', 0):.4f}")
        st.metric("Avg Response Time (s)", f"{claude_metrics.get('avg_time_seconds', 0):.3f}")

    # Visualization
    st.header("📈 Visualizations")
    st.plotly_chart(summary_fig, use_container_width=True)
    st.plotly_chart(dashboard_fig, use_container_width=True)

    # Comparison table
    st.header("📋 Model Comparison Table")
    st.dataframe(comparison_df, use_container_width=True)

    # Exported files
    st.header("📁 Exported Files")
    st.write(f"**Ollama Results:** `{ollama_csv}`")
    st.write(f"**Claude Results:** `{claude_csv}`")
    st.write(f"**Metrics JSON:** `{metrics_file}`")
    st.write(f"**HTML Report:** `{report_file}`")


# ===========================
# Main Evaluation Script
# ===========================

def main():
    payload = run_evaluation_pipeline(log_fn=print)
    comparison_df = payload["comparison_df"]
    results_ollama = payload["results_ollama"]
    results_claude = payload["results_claude"]
    results_by_model = {
        "Ollama": results_ollama,
        "Claude": results_claude,
    }

    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    print(comparison_df.to_string(index=False))
    print()
    
    # Detailed per-model metrics
    for model_name, results in results_by_model.items():
        metrics = compute_metrics(results)
        print(f"\n{model_name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    # Export detailed results
    print("\n" + "=" * 70)
    print("EXPORTING RESULTS")
    print("=" * 70)
    print()
    
    # Export as CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ollama results
    ollama_df = build_results_df(results_ollama)
    ollama_csv = f"eval_ollama_{timestamp}.csv"
    ollama_df.to_csv(ollama_csv, index=False)
    print(f"  ✓ Exported Ollama results: {ollama_csv}")
    
    # Claude results
    claude_df = build_results_df(results_claude)
    claude_csv = f"eval_claude_{timestamp}.csv"
    claude_df.to_csv(claude_csv, index=False)
    print(f"  ✓ Exported Claude results: {claude_csv}")
    
    # Export comparison metrics
    metrics_json = {
        "timestamp": timestamp,
        "comparison": comparison_df.to_dict(orient="records"),
        "ollama_metrics": compute_metrics(results_ollama),
        "claude_metrics": compute_metrics(results_claude),
    }
    metrics_file = f"eval_metrics_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  ✓ Exported metrics: {metrics_file}")

    # Plotly report for non-technical stakeholders
    report_file = f"eval_dashboard_{timestamp}.html"
    generate_plotly_report(comparison_df, ollama_df, claude_df, report_file)
    print(f"  ✓ Exported Plotly dashboard: {report_file}")
    print()


if __name__ == "__main__":
    is_streamlit = "streamlit" in os.path.basename(sys.argv[0]).lower() or bool(os.environ.get("STREAMLIT_SERVER_PORT"))
    if is_streamlit:
        run_streamlit_app()
    else:
        main()
