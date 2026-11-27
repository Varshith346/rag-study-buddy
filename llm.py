"""Returns a chat model for the provider set in .env."""
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# defaults; override with LLM_MODEL
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "google_genai": "gemini-2.5-flash",
    "anthropic": "claude-haiku-4-5",
    "ollama": "llama3.2",
}


def get_llm(temperature: float = 0):
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Use one of: {', '.join(DEFAULT_MODELS)}")
    model = os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]
    return init_chat_model(model, model_provider=provider, temperature=temperature)
