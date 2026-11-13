"""LLM configuration. Switch provider with LLM_PROVIDER in .env (gemini | groq | ollama)."""
import os

from dotenv import load_dotenv

load_dotenv()

DEFAULTS = {
    "gemini": ("GEMINI_MODEL", "gemini-2.5-flash", "GOOGLE_API_KEY"),
    "groq": ("GROQ_MODEL", "openai/gpt-oss-120b", "GROQ_API_KEY"),
    "ollama": ("OLLAMA_MODEL", "llama3.1", None),
}


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").strip().lower()


def llm_status():
    """Return (provider, model, api_key_present) for display in the UI."""
    p = provider()
    if p not in DEFAULTS:
        return p, "unknown", False
    model_env, default_model, key_env = DEFAULTS[p]
    model = os.getenv(model_env, default_model)
    ok = True if key_env is None else bool(os.getenv(key_env))
    return p, model, ok


def get_llm(temperature: float = 0.3):
    """Build the chat model. Imports are lazy so unused providers need not be installed."""
    p = provider()
    if p not in DEFAULTS:
        raise ValueError(f"Unknown LLM_PROVIDER '{p}'. Use gemini, groq or ollama.")
    model_env, default_model, _ = DEFAULTS[p]
    model = os.getenv(model_env, default_model)

    if p == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
    if p == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature)
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model, temperature=temperature)
