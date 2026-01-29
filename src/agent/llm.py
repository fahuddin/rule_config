# agent/llm.py
from langchain_ollama import ChatOllama


def get_llm(model: str, temperature: float = 0.0) -> ChatOllama:
    """
    Create and return an Ollama-backed chat model instance.

    Args:
        model: Ollama model name (e.g., "llama3.1", "mistral", "phi3")
        temperature: Controls randomness. 0.0 = most deterministic.

    Returns:
        A configured ChatOllama instance that LangChain can call.
    """
    # ChatOllama will talk to the local Ollama server (default: http://localhost:11434)
    # No API keys needed.
    return ChatOllama(model=model, temperature=temperature)
