"""Pick any LLM provider with two lines in .env — the rest of the code never changes.

LangChain's init_chat_model gives every provider (Groq, OpenAI, Gemini, Claude,
Ollama...) the same interface, so .invoke() and .with_structured_output() work everywhere.
"""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# Sensible default model for each provider (override with LLM_MODEL in .env)
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "google_genai": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5",
    "ollama": "llama3.2",  # runs locally, no API key needed
}
# If a provider retires one of these models, just set LLM_MODEL in .env.


def get_llm(temperature: float = 0):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(DEFAULT_MODELS)}")
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]
    return init_chat_model(model, model_provider=provider, temperature=temperature)
