"""Environment-based configuration for the Foldback Service."""

import os
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # LLM provider selection
    llm_provider: Literal["ollama", "openai"] = os.getenv("FOLDBACK_LLM_PROVIDER", "ollama")  # type: ignore[assignment]

    # Ollama settings
    ollama_model: str = os.getenv("FOLDBACK_OLLAMA_MODEL", "qwen2.5:14b")

    # OpenAI-compatible settings
    openai_base_url: str = os.getenv("FOLDBACK_OPENAI_BASE_URL", "http://localhost:11434/v1")
    openai_api_key: str = os.getenv("FOLDBACK_OPENAI_API_KEY", "")
    openai_model: str = os.getenv("FOLDBACK_OPENAI_MODEL", "qwen2.5:14b")

    # Ollama generation parameters
    num_ctx: int = int(os.getenv("FOLDBACK_NUM_CTX", "16384"))
    num_predict: int = int(os.getenv("FOLDBACK_NUM_PREDICT", "1024"))

    # Embedding model (used by /embeddings endpoint)
    embedding_model: str = os.getenv("FOLDBACK_EMBEDDING_MODEL", "nomic-embed-text")

    # Server settings
    port: int = int(os.getenv("FOLDBACK_PORT", "8100"))
    host: str = os.getenv("FOLDBACK_HOST", "0.0.0.0")

    # Logging
    log_level: str = os.getenv("FOLDBACK_LOG_LEVEL", "INFO")

    # Request timeout (seconds)
    request_timeout: int = int(os.getenv("FOLDBACK_REQUEST_TIMEOUT", "120"))

    # STT settings
    stt_model: str = os.getenv("FOLDBACK_STT_MODEL", "large-v3")
    stt_device: str = os.getenv("FOLDBACK_STT_DEVICE", "cuda" if os.name != "nt" else "cpu")
    # Auto-detect compute type: float16 requires CUDA, so default to int8 on CPU
    _stt_compute_type_default = "float16" if os.getenv("FOLDBACK_STT_DEVICE", "cuda" if os.name != "nt" else "cpu") == "cuda" else "int8"
    stt_compute_type: str = os.getenv("FOLDBACK_STT_COMPUTE_TYPE", _stt_compute_type_default)

    # Embedding model settings
    embedding_model: str = os.getenv("FOLDBACK_EMBEDDING_MODEL", "nomic-embed-text")

    # Task manager settings
    batch_size_llm: int = int(os.getenv("FOLDBACK_BATCH_SIZE_LLM", "1"))
    batch_size_stt: int = int(os.getenv("FOLDBACK_BATCH_SIZE_STT", "1"))
    idle_unload_seconds: int = int(os.getenv("FOLDBACK_IDLE_UNLOAD_SECONDS", "30"))


settings = Settings()
